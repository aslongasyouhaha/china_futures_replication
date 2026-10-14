"""《The Skewness of Commodity Futures Returns》复现 —— Fernandez-Perez, Frijns, Fuertes & Miffre (2018, Journal of Banking & Finance)

论文方法：
  收益：近月合约，到期前一个月换到第二近月；日对数收益
  偏度：月末用过去 12 个月日收益计算 Pearson 偏度
  组合：按偏度分五组，Q1 偏度最低；多空 = 全额保证金口径 (Q1 − Q5)/2；持有 1 个月
  基准：EW（全部等权做多）、TS（过去 12 个月平均展期收益 ln F1 − ln F2）、Mom（过去 12 个月平均收益）、HP（CFTC 套保压力），
        均为五分组多空 (Q_top − Q_bottom)/2；增强模型另加流动性、持仓变化、通胀 beta、美元 beta、价值、波动率（变异系数）
  检验：Table III 组合特征；Table IV 五组表现与基准 α；Table V 稳健性（过滤后偏度、剔除低流动性、交易成本）；Table VI 截面 Fama-MacBeth
中国对应：
  paper：每月选“交割月 ≥ 当前月 + 2”的第一个合约（持有月内不进入交割月，即个人须退出的月份之前，对应论文“到期前一个月换月”）；
         合约当天持仓 ≥ 100 手且有成交（同 Basis-Momentum 复现）；数据来自 basis_momentum/output
  liquid：主力合约（中国化对照）
  HP 因子（CFTC 持仓）在中国没有对应数据，基准模型为 EW + TS + Mom 三因子；通胀 beta、美元 beta 暂缺（待下载 CPI、汇率）
用法：python skew.py [paper|liquid]
"""
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
BM = os.path.join(HERE, '..', 'basis_momentum', 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
from pricing import nw_cov  # noqa: E402

MIN_OI = 100
MIN_N = 10
MIN_DAYS = 150
SPLIT = pl.date(2016, 1, 1)
MULT = os.path.join(HERE, '..', 'common', 'output', 'multipliers.parquet')


def nw_t(x, L=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / np.sqrt(nw_cov(x, L)[0, 0] / len(x))


def nw_reg(y, X, L=None):
    X = np.column_stack([np.ones(len(y)), X])
    A = np.linalg.inv(X.T @ X)
    b = A @ X.T @ y
    e = y - X @ b
    V = A @ (nw_cov(X * e[:, None], L) * len(y)) @ A
    return b, b / np.sqrt(np.diag(V)), 1 - e.var() / y.var()


# ---------------------------------------------------------------- 数据与信号
def load(v):
    c = pl.read_parquet(os.path.join(BM, f'curve_{v}.parquet'))
    n1 = c.filter(pl.col('n') == 1).select('product', 'm', 'symbol', 'ret', 'open_interest', 'volume', 'px')
    d = pl.read_parquet(os.path.join(BM, f'daily_{v}.parquet')).filter(pl.col('n') <= 2)
    d = d.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('hm'))
    mult = pl.read_parquet(MULT).select('product', 'year', 'mult')
    d1 = d.filter((pl.col('n') == 1) & pl.col('r').is_not_null()).with_columns((pl.col('r') + 1).log().alias('lr'), pl.col('trading_day').dt.year().cast(pl.Int64).alias('year'))
    d1 = d1.join(mult, on=['product', 'year'], how='left').with_columns((pl.col('volume') * pl.col('px') * pl.col('mult').fill_null(1.0)).alias('dv'))
    px2 = d.filter(pl.col('n') == 2).select('product', 'trading_day', pl.col('px').alias('px2'))
    d1 = d1.join(px2, on=['product', 'trading_day'], how='left').with_columns((pl.col('px').log() - pl.col('px2').log()).alias('ry'))
    return n1, d1


def total_oi():
    import glob
    rows = []
    for f in glob.glob(f'{RAW_DATA}/*/*/daily_index.parquet'):
        p = os.path.basename(os.path.dirname(f))
        x = pl.read_parquet(f, columns=['trading_day', 'open_interest']).with_columns(pl.lit(p).alias('product'),
                                                                                     pl.col('trading_day').dt.truncate('1mo').alias('m'))
        rows.append(x.group_by(['product', 'm']).agg(pl.col('open_interest').sort_by('trading_day').last().alias('toi')))
    return pl.concat(rows)


def signals(n1, d1):
    """每个品种每个月末（m）的信号，窗口 = 持有月 m−11..m 的日数据。"""
    months = n1['m'].unique().sort().to_list()
    d1 = d1.sort(['product', 'trading_day'])
    byp = d1.partition_by(['product'], as_dict=True)
    rows = []
    for (p,), g in byp.items():
        hm = g['hm'].to_numpy()
        lr = g['lr'].to_numpy()
        ry = g['ry'].to_numpy()
        dv = g['dv'].to_numpy()
        px = g['px'].to_numpy()
        ums = np.unique(hm)
        for m in ums:
            m_pd = np.datetime64(m, 'M')
            w12 = (hm.astype('datetime64[M]') > m_pd - 12) & (hm.astype('datetime64[M]') <= m_pd)
            if w12.sum() < MIN_DAYS:
                continue
            x = lr[w12]
            sk = float(((x - x.mean()) ** 3).mean() / x.std(ddof=1) ** 3)
            w2 = (hm.astype('datetime64[M]') > m_pd - 2) & (hm.astype('datetime64[M]') <= m_pd)
            ok = w2 & (dv > 0)
            amihud = float(np.nanmean(np.abs(lr[ok]) / dv[ok]) * 1e9) if ok.sum() > 10 else np.nan
            w36 = (hm.astype('datetime64[M]') > m_pd - 36) & (hm.astype('datetime64[M]') <= m_pd)
            y36 = lr[w36]
            cv = float(y36.std(ddof=1) / abs(y36.mean())) if w36.sum() > 500 and y36.mean() != 0 else np.nan
            wv = (hm.astype('datetime64[M]') > m_pd - 66) & (hm.astype('datetime64[M]') <= m_pd - 54)
            val = float(np.log(np.nanmean(px[wv]) / px[hm.astype('datetime64[M]') <= m_pd][-1])) if wv.sum() > 150 else np.nan
            rows.append({'product': p, 'm': np.datetime64(m, 'D').item(), 'SK': sk, 'n_days': int(w12.sum()), 'RY': float(np.nanmean(ry[w12])) if np.isfinite(ry[w12]).sum() > 100 else np.nan,
                         'MOM': float(x.mean()), 'ILLIQ': amihud, 'CV': cv, 'VAL': val})
    s = pl.DataFrame(rows).with_columns(pl.col('m').cast(pl.Date))
    return s


def panel(v):
    n1, d1 = load(v)
    s = signals(n1, d1)
    uni = pl.read_parquet(os.path.join(BM, 'universe.parquet'))
    toi = total_oi().sort(['product', 'm']).with_columns((pl.col('toi') / pl.col('toi').shift(12).over('product')).log().alias('DOI'),
                                                         (pl.col('m').shift(12).over('product') == pl.col('m').dt.offset_by('-12mo')).alias('oi_ok'))
    x = n1.join(uni, on=['product', 'm']).join(s, on=['product', 'm']).join(toi.filter(pl.col('oi_ok')).select('product', 'm', 'DOI'), on=['product', 'm'], how='left')
    x = x.filter(pl.col('ret').is_not_null() & (pl.col('open_interest') >= MIN_OI) & (pl.col('volume') > 0))
    n = x.group_by('m').agg(pl.len().alias('N'))
    return x.join(n, on='m').filter(pl.col('N') >= MIN_N).sort(['m', 'product']), d1


# ---------------------------------------------------------------- 组合
def quint(g, col, high_long):
    """五分组：返回 (Q1..Q5 的成员列表)，Q1 = 信号最低。"""
    o = g.drop_nulls(col).sort([col, 'product'])
    n = o.height
    if n < MIN_N:
        return None
    idx = np.array_split(np.arange(n), 5)
    return [o[i.tolist()] for i in idx]


def factor_returns(x, col, high_long=True, members=False):
    """(Q_long − Q_short)/2 的月收益，以及成员。high_long：信号高的做多。"""
    rows, mem = [], []
    for (m,), g in x.group_by(['m'], maintain_order=True):
        q = quint(g, col, high_long)
        if q is None:
            continue
        lo, hi = q[0], q[4]
        lg, sh = (hi, lo) if high_long else (lo, hi)
        rows.append({'m': m, 'r': 0.5 * (lg['ret'].mean() - sh['ret'].mean()), **{f'Q{k+1}': q[k]['ret'].mean() for k in range(5)}})
        if members:
            mem += [{'m': m, 'product': p, 'leg': 'L'} for p in lg['product']] + [{'m': m, 'product': p, 'leg': 'S'} for p in sh['product']]
    out = pl.DataFrame(rows).sort('m')
    return (out, pl.DataFrame(mem)) if members else out


def benchmark(x):
    ew = x.group_by('m').agg(pl.col('ret').mean().alias('EW'))
    f = ew
    for name, col, hl in (('TS', 'RY', False), ('MOM', 'MOM', True), ('LIQ', 'ILLIQ', True), ('OI', 'DOI', True), ('VAL', 'VAL', True), ('VOL', 'CV', True)):
        # TS：RY = ln F1 − ln F2，越大越贴水 → 做多高 RY
        r = factor_returns(x, col, True if name == 'TS' else hl)
        f = f.join(r.select('m', pl.col('r').alias(name)), on='m', how='left')
    return f.sort('m')


# ---------------------------------------------------------------- 表
def table3(x):
    rows = []
    for (m,), g in x.group_by(['m'], maintain_order=True):
        q = quint(g, 'SK', False)
        if q is None:
            continue
        for k in range(5):
            rows.append({'m': m, 'Q': f'Q{k+1}', 'SK': q[k]['SK'].mean(), 'RY': q[k]['RY'].mean(), 'MOM': q[k]['MOM'].mean() * 243})
    d = pl.DataFrame(rows)
    out = d.group_by('Q').agg(pl.col('SK', 'RY', 'MOM').mean()).sort('Q')
    diff = d.pivot(values=['SK', 'RY', 'MOM'], index='m', columns='Q', aggregate_function='first')
    tt = {c: nw_t((diff[f'{c}_Q_Q1'] if f'{c}_Q_Q1' in diff.columns else diff[f'{c}_Q1']).to_numpy() - (diff[f'{c}_Q_Q5'] if f'{c}_Q_Q5' in diff.columns else diff[f'{c}_Q5']).to_numpy()) for c in ('SK', 'RY', 'MOM')}
    return out, tt


def perf(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    mu, sd = r.mean() * 12, r.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0].std(ddof=1) * np.sqrt(12) if (r < 0).sum() > 2 else np.nan
    om = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.nan
    eq = np.cumprod(1 + r)
    return {'年化%': mu * 100, 't(NW)': nw_t(r), '年化波动%': sd * 100, '夏普': mu / sd, 'Sortino': mu / dn, 'Omega': om,
            '偏度': st.skew(r), '最大回撤%': (eq / np.maximum.accumulate(eq) - 1).min() * 100, '月数': len(r)}


def table4(sk, f):
    rows = []
    d = sk.join(f, on='m', how='left')
    for k in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'r']:
        y = d[k].to_numpy()
        row = {'组合': '多空 (Q1−Q5)/2' if k == 'r' else k, **perf(y)}
        F = d.select('EW', 'TS', 'MOM').to_numpy()
        ok = ~np.isnan(y) & ~np.isnan(F).any(1)
        b, t, r2 = nw_reg(y[ok], F[ok])
        row.update({'α 年化%': b[0] * 1200, 't(α)': t[0], 'β_EW': b[1], 'β_TS': b[2], 't_TS': t[2], 'β_MOM': b[3], 't_MOM': t[3], 'R2': r2})
        rows.append(row)
    return pl.DataFrame(rows)


def alpha_vs(y, f, cols):
    d = y.join(f, on='m', how='left').drop_nulls(['r'] + cols)
    b, t, r2 = nw_reg(d['r'].to_numpy(), d.select(cols).to_numpy())
    return b[0] * 1200, t[0], r2, d.height


def daily_factors(x, d1, cols):
    """日度因子收益（成员按形成月固定），用于过滤后偏度与 Fama-MacBeth。"""
    out = None
    dd = d1.select('product', 'trading_day', pl.col('m_form').alias('m'), 'lr')
    ew = dd.join(x.select('product', 'm'), on=['product', 'm']).group_by('trading_day').agg(pl.col('lr').mean().alias('EW'))
    out = ew
    for name, col, hl in cols:
        _, mem = factor_returns(x, col, hl, members=True)
        j = dd.join(mem, on=['product', 'm'])
        a = j.group_by(['trading_day', 'leg']).agg(pl.col('lr').mean()).pivot(values='lr', index='trading_day', columns='leg')
        out = out.join(a.select('trading_day', (0.5 * (pl.col('L') - pl.col('S'))).alias(name)), on='trading_day', how='left')
    return out.sort('trading_day')


def filtered_skew(x, d1, fd, mode):
    """把过去 12 个月日收益对因子（及/或日历月虚拟变量）回归，用残差算偏度。"""
    dd = d1.join(fd, on='trading_day', how='left').sort(['product', 'trading_day'])
    res = []
    byp = dd.partition_by(['product'], as_dict=True)
    need = x.select('product', 'm').unique()
    for (p,), g in byp.items():
        ms = need.filter(pl.col('product') == p)['m'].to_list()
        hm = g['hm'].to_numpy().astype('datetime64[M]')
        y = g['lr'].to_numpy()
        F = g.select('EW', 'TS', 'MOM').to_numpy()
        mon = g['trading_day'].dt.month().to_numpy()
        for m in ms:
            w = (hm > np.datetime64(m, 'M') - 12) & (hm <= np.datetime64(m, 'M'))
            Xs = [np.ones(w.sum())]
            if 'factor' in mode:
                Xs.append(F[w])
            if 'season' in mode:
                Xs.append(np.column_stack([(mon[w] == k).astype(float) for k in range(2, 13)]))
            X = np.column_stack(Xs)
            ok = ~np.isnan(X).any(1) & ~np.isnan(y[w])
            if ok.sum() < MIN_DAYS:
                continue
            e = y[w][ok] - X[ok] @ np.linalg.lstsq(X[ok], y[w][ok], rcond=None)[0]
            res.append({'product': p, 'm': m, 'SKf': float(((e - e.mean()) ** 3).mean() / e.std(ddof=1) ** 3)})
    return x.join(pl.DataFrame(res).with_columns(pl.col('m').cast(pl.Date)), on=['product', 'm'], how='left')


def turnover_breakeven(x, col):
    w_prev, tos, rets = {}, [], []
    for (m,), g in x.group_by(['m'], maintain_order=True):
        q = quint(g, col, False)
        if q is None:
            continue
        w = {p: 0.5 / q[0].height for p in q[0]['product']}
        for p in q[4]['product']:
            w[p] = w.get(p, 0) - 0.5 / q[4].height
        keys = set(w) | set(w_prev)
        tos.append(sum(abs(w.get(k, 0) - w_prev.get(k, 0)) for k in keys))
        rets.append(0.5 * (q[0]['ret'].mean() - q[4]['ret'].mean()))
        w_prev = w
    return np.mean(rets) / np.mean(tos) * 100, np.mean(tos)


def fama_macbeth(x, d1, fd, models):
    """论文 Table VI：每月用当月日数据估计 beta（第一步），再对当月收益做截面回归（第二步）。"""
    dd = d1.select('product', 'trading_day', 'hm', 'lr').join(fd, on='trading_day', how='left')
    dd = dd.with_columns(pl.col('hm').dt.offset_by('-1mo').alias('m'))
    xs = x.select('product', 'm', 'ret')
    out = []
    for name, cols in models.items():
        lam, r2 = [], []
        for (m,), g in dd.join(xs, on=['product', 'm']).group_by(['m'], maintain_order=True):
            B, R = [], []
            for (p,), h in g.group_by(['product']):
                h = h.drop_nulls(['lr'] + cols)
                if h.height < len(cols) + 5:
                    continue
                X = np.column_stack([np.ones(h.height), h.select(cols).to_numpy()])
                B.append(np.linalg.lstsq(X, h['lr'].to_numpy(), rcond=None)[0][1:])
                R.append(h['ret'][0])
            if len(R) < len(cols) + 3:
                continue
            Z = np.column_stack([np.ones(len(R)), np.array(B)])
            co, *_ = np.linalg.lstsq(Z, np.array(R), rcond=None)
            e = np.array(R) - Z @ co
            lam.append(co)
            r2.append(1 - e.var() / np.var(R))
        L = np.array(lam)
        row = {'模型': name, '月数': len(L), '平均 R2%': np.mean(r2) * 100}
        for i, c in enumerate(['常数'] + cols):
            row[f'λ_{c}'] = L[:, i].mean() * 1200
            row[f't_{c}'] = nw_t(L[:, i])
        out.append(row)
    return pl.DataFrame(out)


def main(v):
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
    os.makedirs(OUT, exist_ok=True)
    x, d1 = panel(v)
    x.write_parquet(os.path.join(OUT, f'panel_{v}.parquet'))
    print(f'\n===== {v}：{x.height} 品种-月，{x["m"].min()} ~ {x["m"].max()}，{x["m"].n_unique()} 个月，每月品种数中位数 {x.group_by("m").len()["len"].median():.0f}')
    # Table I：偏度分布
    t1 = x.group_by('product').agg(pl.col('SK').mean().alias('均值'), pl.col('SK').quantile(0.25).alias('25%'), pl.col('SK').median().alias('中位数'),
                                   pl.col('SK').quantile(0.75).alias('75%'), pl.col('SK').std().alias('标准差'), (pl.col('SK') > 0).mean().alias('为正比例'), pl.len().alias('月数')).sort('product')
    t1.write_parquet(os.path.join(OUT, f't1_{v}.parquet'))
    print('偏度为正的月份比例（品种平均）', round(t1['为正比例'].mean(), 3))
    f = benchmark(x)
    f.write_parquet(os.path.join(OUT, f'factors_{v}.parquet'))
    t2 = pl.DataFrame([{'因子': c, **perf(f[c].to_numpy())} for c in ['EW', 'TS', 'MOM', 'LIQ', 'OI', 'VAL', 'VOL']])
    t2.write_parquet(os.path.join(OUT, f't2_{v}.parquet'))
    print(t2.with_columns(pl.col(pl.Float64).round(2)))
    t3, tt = table3(x)
    t3.write_parquet(os.path.join(OUT, f't3_{v}.parquet'))
    pl.DataFrame([tt]).write_parquet(os.path.join(OUT, f't3t_{v}.parquet'))
    print(t3.with_columns(pl.col(pl.Float64).round(4)), tt)
    sk = factor_returns(x, 'SK', high_long=False)
    sk.write_parquet(os.path.join(OUT, f'skew_{v}.parquet'))
    t4 = table4(sk, f)
    t4.write_parquet(os.path.join(OUT, f't4_{v}.parquet'))
    print(t4.with_columns(pl.col(pl.Float64).round(2)))
    # Table V
    rows = []
    for lab, cols in (('基准三因子（EW、TS、Mom）', ['EW', 'TS', 'MOM']), ('+ 流动性', ['EW', 'TS', 'MOM', 'LIQ']), ('+ 持仓变化', ['EW', 'TS', 'MOM', 'OI']),
                      ('+ 价值', ['EW', 'TS', 'MOM', 'VAL']), ('+ 波动率', ['EW', 'TS', 'MOM', 'VOL']),
                      ('全部因子', ['EW', 'TS', 'MOM', 'LIQ', 'OI', 'VAL', 'VOL'])):
        a, t, r2, n = alpha_vs(sk, f, cols)
        rows.append({'检验': f'B 基准模型：{lab}', 'α 年化%': a, 't(α)': t, 'R2': r2, '月数': n})
    fd = daily_factors(x, d1, [('TS', 'RY', True), ('MOM', 'MOM', True)])
    for lab, mode in (('A 过滤后偏度：对三因子回归的残差', 'factor'), ('A 过滤后偏度：对日历月虚拟变量回归的残差', 'season'), ('A 过滤后偏度：两者都控制', 'factor+season')):
        xf = filtered_skew(x, d1, fd, mode)
        skf = factor_returns(xf, 'SKf', high_long=False)
        a, t, r2, n = alpha_vs(skf, f, ['EW', 'TS', 'MOM'])
        rows.append({'检验': lab, 'α 年化%': a, 't(α)': t, 'R2': r2, '月数': n, '多空年化%': skf['r'].mean() * 1200})
    lq = x.with_columns(pl.col('ILLIQ').rank().over('m').alias('rk'), pl.len().over('m').alias('nn')).filter(pl.col('rk') <= 0.8 * pl.col('nn'))
    skl = factor_returns(lq, 'SK', high_long=False)
    a, t, r2, n = alpha_vs(skl, f, ['EW', 'TS', 'MOM'])
    rows.append({'检验': 'C 每月剔除流动性最差的 20%', 'α 年化%': a, 't(α)': t, 'R2': r2, '月数': n, '多空年化%': skl['r'].mean() * 1200})
    be, to = turnover_breakeven(x, 'SK')
    rows.append({'检验': f'C 盈亏平衡交易成本（单边，%）；月均换手 {to:.2f}', 'α 年化%': be, 't(α)': None, 'R2': None, '月数': None})
    for lab, sub in (('D 2007–2015', sk.filter(pl.col('m') < SPLIT)), ('D 2016–2026', sk.filter(pl.col('m') >= SPLIT))):
        a, t, r2, n = alpha_vs(sub, f, ['EW', 'TS', 'MOM'])
        rows.append({'检验': lab, 'α 年化%': a, 't(α)': t, 'R2': r2, '月数': n, '多空年化%': sub['r'].mean() * 1200})
    t5 = pl.DataFrame(rows)
    t5.write_parquet(os.path.join(OUT, f't5_{v}.parquet'))
    print(t5.with_columns(pl.col(pl.Float64).round(2)))
    # Table VI：截面
    fd2 = fd.join(daily_factors(x, d1, [('SKEW', 'SK', False)]).select('trading_day', 'SKEW'), on='trading_day', how='left')
    t6 = fama_macbeth(x, d1, fd2, {'基准三因子': ['EW', 'TS', 'MOM'], '三因子 + 偏度': ['EW', 'TS', 'MOM', 'SKEW']})
    t6.write_parquet(os.path.join(OUT, f't6_{v}.parquet'))
    print(t6.with_columns(pl.col(pl.Float64).round(2)))


if __name__ == '__main__':
    for v in (sys.argv[1:] or ['paper', 'liquid']):
        main(v)
