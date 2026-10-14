"""TSMOM 实盘化回测：按资金规模、整手、容量上限和三段式成本模拟。

信号与仓位规则同论文（tsmom.py）：sign(过去 k 月收益) × 40% / σ，月度调仓，品种等权。
在此基础上加入：
  1. 资金规模 C：净值逐日滚动；目标名义 = w × NAV / S，按整手四舍五入（不足半手不开仓）
  2. 流动性：过去 60 日成交额中位数 >= ADV_MIN；单品种持仓名义 <= POS_CAP × 日均成交额（超出按上限缩小）
  3. 成本 = 手续费 + 滑点（按流动性 1/2/4 跳）+ 冲击（平方根模型 Y·σ_日·√(成交手数/日均手数)）
     换月日平旧开新，两条腿都按上述成本计
  4. 执行：月末收盘出信号，lag 个交易日后收盘成交
  5. 保证金占用：商品/股指 12%，国债 TS 0.5%、TF 1.2%、T 2%、TL 3.5%
"""
import glob
import os

import numpy as np
import polars as pl

import tsmom as T
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = T.DATA
OUT = f'{RESEARCH}/common/output'   # 合约乘数、最小变动价位、价差、VWAP
COMMISSION = 1e-4
# 滑点改为数据估计的半个价差（见 microstructure.py），不再按流动性分档
IMPACT_Y = 0.5               # 平方根冲击系数（滑点已含约 1 跳的价差，冲击取 0.5 避免重复计算）
FALLBACK_PART = 1.0          # 日均成交手数缺失时假设的成交占比
ADV_MIN = 1e8                # 60 日成交额中位数门槛（元）
POS_CAP = 0.02               # 单品种持仓名义上限 = 2% × 日均成交额
MARGIN = {'TS': 0.005, 'TF': 0.012, 'T': 0.02, 'TL': 0.035}
MARGIN_DEFAULT = 0.12


def panel():
    """日频面板（交易日 × 品种）。"""
    mult = pl.read_parquet(f'{OUT}/multipliers.parquet').select('product', pl.col('year').cast(pl.Int32), 'mult')
    ticks = pl.read_parquet(f'{OUT}/ticks.parquet').select('product', pl.col('year').cast(pl.Int32), 'tick')
    spr = pl.read_parquet(f'{OUT}/spreads.parquet').select('product', pl.col('year').cast(pl.Int32), 'spread_ticks')             .with_columns(pl.when(pl.col('spread_ticks').is_finite()).then(pl.col('spread_ticks')).alias('spread_ticks'))
    vwap = pl.read_parquet(f'{OUT}/vwap.parquet')
    parts = []
    for f in glob.glob(f'{DATA}/*/*/daily_main.parquet'):
        p = os.path.basename(os.path.dirname(f))
        parts.append(pl.read_parquet(f, columns=['trading_day', 'symbol', 'close', 'close_adj', 'volume'])
                       .with_columns(pl.lit(p).alias('product')))
    d = pl.concat(parts).sort(['product', 'trading_day']).with_columns(pl.col('trading_day').dt.year().alias('year'))
    d = (d.join(mult, on=['product', 'year'], how='left').join(ticks, on=['product', 'year'], how='left')
           .join(spr, on=['product', 'year'], how='left').join(vwap, on=['product', 'trading_day'], how='left')
           .with_columns(pl.col('mult').forward_fill().backward_fill().over('product'),
                         pl.col('tick').forward_fill().backward_fill().over('product'),
                         # 价差估计失败的年份用该品种其他年份的中位数，仍缺则按 1 跳
                         pl.col('spread_ticks').fill_null(pl.col('spread_ticks').median().over('product')).fill_null(1.0)))
    d = d.with_columns(
        (pl.col('close_adj') / pl.col('close_adj').shift(1).over('product') - 1).alias('ret'),
        (pl.col('close') * pl.col('mult')).alias('lot_value'),
        (pl.col('symbol') != pl.col('symbol').shift(1).over('product')).fill_null(False).alias('roll'))
    d = d.with_columns((pl.col('volume') * pl.col('lot_value')).alias('dollar_vol'),
                       pl.col('symbol').shift(1).over('product').alias('prev_symbol'))
    # 换月日：旧合约（前一天的主力）当天的成交量，取自 daily_contracts
    dcs = []
    for f in glob.glob(f'{DATA}/*/*/daily_contracts.parquet'):
        dcs.append(pl.read_parquet(f, columns=['symbol', 'trading_day', 'volume'])
                     .with_columns(pl.lit(os.path.basename(os.path.dirname(f))).alias('product')))
    dc = pl.concat(dcs).rename({'symbol': 'prev_symbol', 'volume': 'old_vol'})
    d = d.join(dc, on=['product', 'prev_symbol', 'trading_day'], how='left')          .with_columns(pl.when(pl.col('roll')).then(pl.col('old_vol')).alias('old_vol'),
                       pl.when(pl.col('roll')).then(pl.col('volume')).alias('new_vol'))
    d = d.with_columns(
        pl.col('dollar_vol').rolling_median(60, min_periods=40).over('product').alias('adv_val'),
        pl.col('volume').rolling_median(60, min_periods=40).over('product').alias('adv_lots'),
        (pl.col('ret').ewm_var(alpha=1 - T.DELTA, adjust=True, min_periods=T.MIN_OBS, ignore_nulls=True)
           .over('product').sqrt()).alias('sigma_d'))
    # 线性成本 = 手续费 + 半个价差（价差由分钟数据 Roll 估计，至少 1 跳 → 半价差至少 0.5 跳）
    half_spread = pl.max_horizontal(pl.lit(0.5), pl.col('spread_ticks') / 2)
    d = d.with_columns((COMMISSION + half_spread * pl.col('tick') / pl.col('close')).alias('base_cost'),
                       (pl.col('vwap') * pl.col('mult')).alias('lot_value_vwap'),
                       pl.col('product').replace(MARGIN, default=MARGIN_DEFAULT).cast(pl.Float64).alias('margin_rate'))
    return d


def wide(d, col, days, prods):
    w = d.pivot(index='trading_day', columns='product', values=col, aggregate_function='first')
    w = pl.DataFrame({'trading_day': days}).join(w, on='trading_day', how='left')
    return w.select([c if c in w.columns else pl.lit(None).alias(c) for c in prods]).to_numpy().astype(float)


def run(d, k=12, lag=1, capital=1e8, y=IMPACT_Y, ticks_mult=1.0, roll_cost=True, cap=POS_CAP, adv_min=ADV_MIN, exec_px='vwap',
        weights=None):
    """exec_px='vwap': 月末收盘后按净值定目标手数，次日按 VWAP 成交（lag 固定为 1）；
    exec_px='close': lag 日后按收盘价成交（旧口径，用于对比）。
    weights: 可选，外部策略的目标权重表 (product, trading_day=信号日, w)，w = 名义金额占净值的比例；
             给定时不使用 TSMOM 信号，也不再除以品种数（供 Carry 等其他策略复用本框架）。"""
    if exec_px == 'vwap':
        lag = 1
    T.MIN_VOL_LOTS = 0                                    # 流动性改由成交额判断
    if weights is None:
        sig = T.monthly_signals(T.load_cache(), k)        # product, trading_day(月末), signal, sigma
        sig = sig.filter(pl.col('signal').is_not_null()).with_columns(
            (pl.col('signal') * T.TARGET_VOL / pl.col('sigma')).alias('w'))
        sig_days = T.monthly_signals(T.load_cache(), k)['trading_day'].unique()
    else:
        sig = weights
        sig_days = weights['trading_day'].unique()
    days = d['trading_day'].unique().sort()
    prods = sorted(d['product'].unique().to_list())
    ix = {p: i for i, p in enumerate(prods)}
    R = np.nan_to_num(wide(d, 'ret', days, prods))
    LV = wide(d, 'lot_value', days, prods)
    ADV = wide(d, 'adv_val', days, prods)
    ADVL = wide(d, 'adv_lots', days, prods)
    ROLL = np.nan_to_num(wide(d, 'roll', days, prods)).astype(bool)
    BC = wide(d, 'base_cost', days, prods)
    BC = 1e-4 + (BC - 1e-4) * ticks_mult
    SD = wide(d, 'sigma_d', days, prods)
    MR = wide(d, 'margin_rate', days, prods)
    CA = wide(d, 'close_adj', days, prods)
    VA = wide(d, 'vwap_adj', days, prods)
    LVW = wide(d, 'lot_value_vwap', days, prods)
    OLDV = wide(d, 'old_vol', days, prods)
    NEWV = wide(d, 'new_vol', days, prods)
    day_ix = {dd: i for i, dd in enumerate(days.to_list())}

    # 月末目标权重矩阵 → 执行日
    W = {}
    for r in sig.iter_rows(named=True):
        t = day_ix.get(r['trading_day'])
        if t is None or t + lag >= len(days) or r['product'] not in ix:
            continue
        W.setdefault(t + lag, {})[ix[r['product']]] = r['w']
    me_days = sorted({day_ix[x] + lag for x in sig_days.to_list() if x in day_ix and day_ix[x] + lag < len(days)})
    per_nav = weights is not None                         # True: w 直接是占净值比例；False: TSMOM 等权，再除以品种数

    n = len(days)
    lots = np.zeros(len(prods))
    nav = capital
    out = np.zeros((n, 9))   # ret, c_comm+slip rebal, c_impact rebal, c_roll, n_pos, margin, dropped, capped, gross_lev
    me_set = set(me_days)
    acc = np.zeros((len(prods), 3))          # 每个品种累计: 调仓线性成本, 调仓冲击, 换月成本（元）
    pending = None                            # VWAP 模式：月末算好的目标手数，次日执行
    for t in range(n):
        lv_prev = LV[t - 1] if t > 0 else LV[t]
        vwap_exec = exec_px == 'vwap' and pending is not None and pending[0] == t
        if vwap_exec:
            new = pending[1]
            va_ok = np.isfinite(VA[t]) & np.isfinite(CA[t - 1]) & np.isfinite(CA[t]) & (VA[t] > 0)
            r1 = np.where(va_ok, VA[t] / np.where(va_ok, CA[t - 1], 1) - 1, R[t])       # 昨收 → VWAP
            r2 = np.where(va_ok, CA[t] / np.where(va_ok, VA[t], 1) - 1, 0.0)            # VWAP → 今收
            lvw = np.where(np.isfinite(LVW[t]), LVW[t], LV[t])
            pnl = np.nansum(lots * np.nan_to_num(lv_prev) * np.nan_to_num(r1)) + np.nansum(new * np.nan_to_num(lvw) * np.nan_to_num(r2))
        else:
            pnl = np.nansum(lots * np.nan_to_num(lv_prev) * R[t])
        c_roll = 0.0
        if roll_cost and t > 0:
            m = ROLL[t] & (lots != 0)
            if m.any():
                # 两条腿分开：平旧仓对比旧合约当天成交量，开新仓对比新合约当天成交量
                q = np.abs(lots[m])
                sd = np.nan_to_num(SD[t, m])
                p_old = q / np.where(OLDV[t, m] > 0, OLDV[t, m], np.nan)
                p_new = q / np.where(NEWV[t, m] > 0, NEWV[t, m], np.nan)
                imp = y * sd * (np.sqrt(np.nan_to_num(p_old, nan=FALLBACK_PART)) + np.sqrt(np.nan_to_num(p_new, nan=FALLBACK_PART)))
                cr = q * LV[t, m] * (2 * BC[t, m] + imp)
                acc[np.where(m)[0], 2] += np.nan_to_num(cr)
                c_roll = np.nansum(cr)
        c_lin = c_imp = 0.0
        dropped = capped = 0
        if vwap_exec:
            tr = np.abs(new - lots)
            m = tr > 0
            if m.any():
                part = tr[m] / np.where(ADVL[t, m] > 0, ADVL[t, m], np.nan)
                cl = tr[m] * lvw[m] * BC[t, m]
                ci = tr[m] * lvw[m] * y * np.nan_to_num(SD[t, m]) * np.sqrt(np.nan_to_num(part, nan=FALLBACK_PART))
                acc[np.where(m)[0], 0] += np.nan_to_num(cl)
                acc[np.where(m)[0], 1] += np.nan_to_num(ci)
                c_lin, c_imp = np.nansum(cl), np.nansum(ci)
            dropped, capped = pending[2], pending[3]
            lots = new
            pending = None
        if exec_px == 'close' and t in me_set:
            tgt = W.get(t, {})
            new = np.zeros(len(prods))
            elig = [j for j, w in tgt.items() if np.isfinite(LV[t, j]) and LV[t, j] > 0 and np.nan_to_num(ADV[t, j]) >= adv_min]
            S = len(elig)
            for j in elig:
                notional = tgt[j] * (nav + pnl) / (1 if per_nav else S)
                lim = cap * ADV[t, j]
                if abs(notional) > lim:
                    notional = np.sign(notional) * lim
                    capped += 1
                q = np.round(notional / LV[t, j])
                if q == 0:
                    dropped += 1
                new[j] = q
            tr = np.abs(new - lots)
            m = tr > 0
            if m.any():
                part = tr[m] / np.where(ADVL[t, m] > 0, ADVL[t, m], np.nan)
                cl = tr[m] * LV[t, m] * BC[t, m]
                ci = tr[m] * LV[t, m] * y * np.nan_to_num(SD[t, m]) * np.sqrt(np.nan_to_num(part, nan=FALLBACK_PART))
                acc[np.where(m)[0], 0] += np.nan_to_num(cl)
                acc[np.where(m)[0], 1] += np.nan_to_num(ci)
                c_lin, c_imp = np.nansum(cl), np.nansum(ci)
            lots = new
        nav_new = nav + pnl - c_roll - c_lin - c_imp
        # me_set 是执行日；VWAP 模式下在执行日前一天（信号日）收盘后定目标手数
        if exec_px == 'vwap' and (t + 1) in me_set:
            tgt = W.get(t + 1, {})
            new_t = np.zeros(len(prods))
            elig = [j for j, w in tgt.items() if np.isfinite(LV[t, j]) and LV[t, j] > 0 and np.nan_to_num(ADV[t, j]) >= adv_min]
            S = len(elig)
            dr = cp = 0
            for j in elig:
                notional = tgt[j] * nav_new / (1 if per_nav else S)
                lim = cap * ADV[t, j]
                if abs(notional) > lim:
                    notional = np.sign(notional) * lim
                    cp += 1
                q = np.round(notional / LV[t, j])
                if q == 0:
                    dr += 1
                new_t[j] = q
            pending = (t + 1, new_t, dr, cp)
        notional_now = np.abs(lots) * np.nan_to_num(LV[t])
        out[t] = [(nav_new / nav - 1) if nav > 0 else 0, c_lin / nav, c_imp / nav, c_roll / nav, (lots != 0).sum(),
                  np.nansum(notional_now * MR[t]) / nav_new, dropped, capped, notional_now.sum() / nav_new]
        nav = nav_new
    res = pl.DataFrame({'trading_day': days}).with_columns(
        *[pl.Series(c, out[:, i]) for i, c in enumerate(['ret', 'c_lin', 'c_imp', 'c_roll', 'n_pos', 'margin', 'dropped', 'capped', 'gross_lev'])])
    run.by_product = pl.DataFrame({'product': prods, 'c_lin': acc[:, 0], 'c_imp': acc[:, 1], 'c_roll': acc[:, 2]})
    return res.filter(pl.col('trading_day') >= pl.date(2006, 1, 1))


def summary(r):
    s = T.stats(r['ret'])
    cost = (r['c_lin'] + r['c_imp'] + r['c_roll']).mean() * T.ANN
    return s, cost


def main():
    d = panel()
    miss = d.group_by('product').agg(pl.col('mult').is_null().all()).filter(pl.col('mult'))['product'].to_list()
    print('没有合约乘数、无法交易的品种:', miss)
    print(f'定稿方案: 资金 1 亿，次日 VWAP 成交，成交额门槛 {ADV_MIN/1e8:.0f} 亿/日，单品种持仓 <= {POS_CAP:.0%} 日均成交额，'
          f'手续费 1bp，滑点 = 数据估计的半个价差，冲击 Y={IMPACT_Y}', end='\n\n')

    r = run(d, k=12, capital=1e8)
    s, cost = summary(r)
    print('== 定稿方案结果（k=12，2006 年起）==')
    print(T.fmt(s))
    print(f"年化成本 {cost:.2%}：调仓手续费+价差 {r['c_lin'].mean()*T.ANN:.2%}，调仓冲击 {r['c_imp'].mean()*T.ANN:.2%}，换月 {r['c_roll'].mean()*T.ANN:.2%}")
    print(f"平均持仓 {r['n_pos'].mean():.0f} 个；每次调仓被容量上限缩小 {r.filter(pl.col('capped')>0)['capped'].mean() if r['capped'].sum() else 0:.1f} 个、"
          f"因不足半手放弃 {r.filter(pl.col('dropped')>0)['dropped'].mean() if r['dropped'].sum() else 0:.1f} 个")
    print(f"保证金占净值 中位数 {r['margin'].median():.1%}，最高 {r['margin'].max():.1%}；名义杠杆中位数 {r['gross_lev'].median():.1f} 倍")
    yr = r.group_by(pl.col('trading_day').dt.year().alias('y')).agg(((1 + pl.col('ret')).product() - 1).alias('ret')).sort('y')
    print('逐年: ' + '  '.join(f"{a}:{b:+.1%}" for a, b in yr.rows()))
    for start in (2010, 2016):
        q = r.filter(pl.col('trading_day').dt.year() >= start)
        print(f'{start} 年起: {T.fmt(T.stats(q["ret"]))}')

    print('\n== 从论文口径到定稿方案（k=12）==')
    d0 = d.with_columns(pl.lit(0.0).alias('base_cost'))
    ladder = [
        ('① 无摩擦，月末收盘成交，无门槛', d0, dict(exec_px='close', lag=0, capital=1e13, y=0, roll_cost=False, cap=1e9, adv_min=0)),
        ('② 改为次日 VWAP 成交', d0, dict(capital=1e13, y=0, roll_cost=False, cap=1e9, adv_min=0)),
        ('③ + 成交额门槛 1 亿/日', d0, dict(capital=1e13, y=0, roll_cost=False, cap=1e9)),
        ('④ + 资金 1 亿、整手、容量上限 2%', d0, dict(capital=1e8, y=0, roll_cost=False)),
        ('⑤ + 调仓手续费和半价差', d, dict(capital=1e8, y=0, roll_cost=False)),
        ('⑥ + 换月成本', d, dict(capital=1e8, y=0)),
        ('⑦ + 冲击成本（= 定稿方案）', d, dict(capital=1e8)),
    ]
    for name, dd, kw in ladder:
        rr = run(dd, k=12, **kw)
        ss, cc = summary(rr)
        print(f'{name:28s} 夏普 {ss["sharpe"]:5.2f}  年化收益 {ss["ann_ret"]:6.2%}  年化成本 {cc:5.2%}')

    print('\n== 敏感性（其余同定稿方案）==')
    for name, kw in [('资金 1000 万', {'capital': 1e7}), ('资金 10 亿', {'capital': 1e9}),
                     ('价差成本 ×0.5', {'ticks_mult': 0.5}), ('价差成本 ×2', {'ticks_mult': 2.0}),
                     ('冲击 Y=0', {'y': 0.0}), ('冲击 Y=1', {'y': 1.0}),
                     ('容量上限 1%', {'cap': 0.01}), ('容量上限 5%', {'cap': 0.05}),
                     ('无成交额门槛', {'adv_min': 0}), ('次日收盘成交（旧口径）', {'exec_px': 'close', 'lag': 1}),
                     ('回看期 k=3', {'k': 3}), ('回看期 k=6', {'k': 6})]:
        rr = run(d, **{'k': 12, 'capital': 1e8, **kw})
        ss, cc = summary(rr)
        print(f'{name:22s} 夏普 {ss["sharpe"]:5.2f}  年化收益 {ss["ann_ret"]:6.2%}  最大回撤 {ss["max_dd"]:7.2%}  年化成本 {cc:5.2%}')


if __name__ == '__main__':
    main()
