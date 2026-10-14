"""《A Tale of Two Premiums: The Role of Hedgers and Speculators in Commodity Futures Markets》复现 —— Kang, Rouwenhorst & Tang (2020, Journal of Finance)

论文：CFTC 周度持仓（商业 = 套保者，非商业 = 投机者），每周二；
  Q  = 某类交易者净多头的周变化 / 上周总持仓
  HP = 商业交易者净空头 / 总持仓；平滑 HP̄ = 过去 52 周净空头均值 / 当周总持仓
  Table II   Fama-MacBeth：Q_t 对当周、上周收益与上周 Q
  Table III  Fama-MacBeth：R_{t+1}、R_{t+2} 对 Q_t（控制对数基差、S·特质波动、R_t）
  Table V    按 Q 五分组，看形成后第 1–4、5–10、11–20、21–40 个交易日的收益
  Table VI   R_{t+1} 对 HP、HP̄、Q + HP̄
  Table VIII HP̄ × Q 双重排序
中国对应（用户选定方案 B）：中国没有按交易者类型的持仓，用交易所公布的“前 20 名会员”整体代表“大户”：
  每天把同一品种所有合约的前 20 名会员（剔除“期货公司会员”等汇总行）多头、空头持仓加总；周末（每周最后一个交易日）取值。
  Q_t = (净多_t − 净多_{t−1}) / 总持仓_{t−1}（要求总持仓 ≥ 1000 手，避免新上市、冷门品种比值失真）；HP_t = (空 − 多)_t / 总持仓_t；HP̄ = 过去 52 周 (空 − 多) 均值 / 当周总持仓。
  这检验的是“大户 vs 其余交易者”，不是论文的“套保 vs 投机”；前 20 名成员每周会变，加总值带有成员更替的噪声。
  收益：主力合约后复权日收益（论文取曲线上最活跃的部分，中国即主力，而非常常不活跃的近月）。
  对数基差：周末主力与更远月份中持仓最大的合约，(ln F2 − ln F1)/(T2 − T1)，T 以年计。
  特质波动：过去 52 周周收益对商品等权指数回归的残差年化标准差（论文对标普 500）；S = 大户净多为 1、净空为 −1。
"""
import glob
import os
import sys

import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
sys.path.insert(0, os.path.join(HERE, '..', 'carry'))
import carry as CA  # noqa: E402
from pricing import nw_cov  # noqa: E402

RAW = RAW_DATA
SKIP = {'EC', 'LF', 'PPF', 'VF'}
MIN_N = 10


def nw_t(x, L=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / np.sqrt(nw_cov(x, L)[0, 0] / len(x))


def week_ends():
    cal = pl.read_parquet(f'{RAW}/trading_calendar.parquet').with_columns(pl.col('trading_day').dt.strftime('%G-%V').alias('wk'))
    return cal.group_by('wk').agg(pl.col('trading_day').max().alias('we')).sort('we')['we']


def product_panel(ex, p, we):
    h = pl.read_parquet(f'{RAW}/{ex}/{p}/holding_rank.parquet').filter((pl.col('level') == '合约') & (pl.col('row_type') == '会员'))
    if h.height == 0:
        return None
    g = h.group_by('trading_day').agg(pl.col('long_hld').fill_null(0).sum().alias('L'), pl.col('short_hld').fill_null(0).sum().alias('S'))
    dc = pl.read_parquet(f'{RAW}/{ex}/{p}/daily_contracts.parquet', columns=['symbol', 'trading_day', 'close', 'settle', 'open_interest'])
    oi = dc.group_by('trading_day').agg(pl.col('open_interest').sum().alias('OI'))
    dm = pl.read_parquet(f'{RAW}/{ex}/{p}/daily_main.parquet', columns=['trading_day', 'symbol', 'close_adj']).sort('trading_day')
    dm = dm.with_columns((pl.col('close_adj') / pl.col('close_adj').shift(1) - 1).alias('r'))
    # 周末对数基差
    dcw = dc.filter(pl.col('trading_day').is_in(we)).with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'),
                                                                  pl.col('symbol').map_elements(CA.month_index, return_dtype=pl.Int64).alias('cm'))
    mainw = dm.filter(pl.col('trading_day').is_in(we)).select('trading_day', pl.col('symbol').alias('main'))
    j = dcw.join(mainw, on='trading_day')
    near = j.filter(pl.col('symbol') == pl.col('main')).select('trading_day', pl.col('px').alias('f1'), pl.col('cm').alias('c1'))
    far = j.join(near, on='trading_day').filter((pl.col('cm') > pl.col('c1')) & (pl.col('open_interest') > 0)).sort(['trading_day', 'open_interest'], descending=[False, True]) \
        .group_by('trading_day', maintain_order=True).first().select('trading_day', pl.col('px').alias('f2'), pl.col('cm').alias('c2'))
    basis = near.join(far, on='trading_day').select('trading_day', ((pl.col('f2').log() - pl.col('f1').log()) / ((pl.col('c2') - pl.col('c1')) / 12)).alias('B'))
    d = dm.select('trading_day', 'r').join(g, on='trading_day', how='left').join(oi, on='trading_day', how='left')
    d = d.with_columns(pl.lit(p).alias('product'))
    return d, basis


def build():
    we = week_ends()
    daily, basis = [], []
    for f in glob.glob(f'{RAW}/*/*/holding_rank.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in SKIP:
            continue
        res = product_panel(ex, p, we)
        if res is None:
            continue
        daily.append(res[0])
        basis.append(res[1].with_columns(pl.lit(p).alias('product')))
    d = pl.concat(daily, how='diagonal_relaxed').sort(['product', 'trading_day'])
    # 交易日 → 所属周末
    wk = pl.DataFrame({'we': we}).with_columns(pl.col('we').alias('trading_day'))
    d = d.join_asof(wk.sort('trading_day'), on='trading_day', strategy='forward', by=None) if False else d
    weeks = pl.DataFrame({'we': we}).sort('we')
    d = d.sort('trading_day').join_asof(weeks.with_columns(pl.col('we').alias('trading_day')).sort('trading_day'), on='trading_day', strategy='forward').sort(['product', 'trading_day'])
    return d, pl.concat(basis), we


def weekly(d, basis):
    w = d.group_by(['product', 'we']).agg(((pl.col('r') + 1).product() - 1).alias('R'), pl.col('L').drop_nulls().last(), pl.col('S').drop_nulls().last(), pl.col('OI').drop_nulls().last(), pl.len().alias('nd'))
    w = w.sort(['product', 'we']).with_columns((pl.col('L') - pl.col('S')).alias('NET'))
    w = w.with_columns(((pl.col('NET') - pl.col('NET').shift(1).over('product')) / pl.col('OI').shift(1).over('product')).alias('Q'),
                       ((pl.col('S') - pl.col('L')) / pl.col('OI')).alias('HP'),
                       ((pl.col('S') - pl.col('L')).rolling_mean(52, min_periods=40).over('product') / pl.col('OI')).alias('HPbar'),
                       pl.col('R').shift(-1).over('product').alias('R1'), pl.col('R').shift(-2).over('product').alias('R2'),
                       pl.col('R').shift(1).over('product').alias('Rlag'), pl.col('Q').shift(1).over('product').alias('Qlag') if False else pl.lit(None).alias('_'))
    w = w.with_columns(pl.when(pl.col('Q').is_finite() & (pl.col('OI').shift(1).over('product') >= 1000)).then(pl.col('Q')).alias('Q'),
                       pl.when(pl.col('HP').is_finite()).then(pl.col('HP')).alias('HP'), pl.when(pl.col('HPbar').is_finite()).then(pl.col('HPbar')).alias('HPbar'))
    w = w.with_columns(pl.col('Q').shift(1).over('product').alias('Qlag')).drop('_')
    w = w.join(basis.rename({'trading_day': 'we'}), on=['product', 'we'], how='left')
    # 特质波动：对商品等权指数（当周所有品种平均）回归
    ew = w.group_by('we').agg(pl.col('R').mean().alias('EW'))
    w = w.join(ew, on='we').sort(['product', 'we'])
    iv = []
    for (p,), g in w.partition_by(['product'], as_dict=True).items():
        r, m = g['R'].to_numpy(), g['EW'].to_numpy()
        out = np.full(len(r), np.nan)
        for i in range(51, len(r)):
            y, x = r[i - 51:i + 1], m[i - 51:i + 1]
            ok = ~np.isnan(y) & ~np.isnan(x)
            if ok.sum() >= 40:
                X = np.column_stack([np.ones(ok.sum()), x[ok]])
                e = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
                out[i] = e.std(ddof=1) * np.sqrt(52)
        iv.append(pl.DataFrame({'product': p, 'we': g['we'], 'IV': out}))
    w = w.join(pl.concat(iv), on=['product', 'we'], how='left')
    w = w.with_columns((pl.when(pl.col('NET') > 0).then(1.0).otherwise(-1.0) * pl.col('IV')).alias('SIV'))
    # 品种流动性：当周有持仓排名且总持仓 > 0、当周至少 3 个交易日
    w = w.filter((pl.col('OI') >= 1000) & pl.col('L').is_not_null()).filter(pl.col('nd') >= 3)
    return w


def fmb(w, y, xs):
    rows = []
    ww = w.drop_nulls([y] + xs).filter(pl.all_horizontal([pl.col(c).is_finite() for c in [y] + xs]))
    for (we,), g in ww.group_by(['we'], maintain_order=True):
        if g.height < MIN_N:
            continue
        X = np.column_stack([np.ones(g.height)] + [g[c].to_numpy() for c in xs])
        b = np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0]
        e = g[y].to_numpy() - X @ b
        rows.append(np.r_[b, 1 - e.var() / g[y].to_numpy().var()])
    B = np.array(rows)
    out = {'周数': len(B), 'R2%': B[:, -1].mean() * 100}
    for i, c in enumerate(['常数'] + xs):
        out[f'b_{c}'] = B[:, i].mean()
        out[f't_{c}'] = nw_t(B[:, i])
    return out


def sorts(w, d, col, groups=5):
    """按 col 五分组（每周），形成后第 1–4、5–10、11–20、21–40 个交易日的等权收益（%），以及形成前两周收益。"""
    d = d.sort(['product', 'trading_day']).with_columns(pl.col('trading_day').rank('dense').over('product').alias('k'))
    lr = d.select('product', 'trading_day', 'k', (pl.col('r') + 1).log().alias('lr'))
    idx = lr.select('product', pl.col('trading_day').alias('we'), pl.col('k').alias('k0'))
    rows = []
    mem = []
    for (we,), g in w.drop_nulls([col]).group_by(['we'], maintain_order=True):
        if g.height < MIN_N:
            continue
        o = g.sort([col, 'product'])
        parts = np.array_split(np.arange(g.height), groups)
        for qi, ix in enumerate(parts):
            for p in o['product'][ix.tolist()]:
                mem.append({'we': we, 'product': p, 'q': qi + 1})
    mem = pl.DataFrame(mem).join(idx, on=['we', 'product'])
    for a, b in ((1, 4), (5, 10), (11, 20), (21, 40), (-9, 0)):
        x = mem.join(lr, on='product').filter((pl.col('k') >= pl.col('k0') + a) & (pl.col('k') <= pl.col('k0') + b))
        s = x.group_by(['we', 'product', 'q']).agg(pl.col('lr').sum().alias('cr')).group_by(['we', 'q']).agg(pl.col('cr').mean())
        pv = s.pivot(values='cr', index='we', columns='q').sort('we')
        for q in range(1, groups + 1):
            rows.append({'窗口': f'第 {a}~{b} 日' if a > 0 else '形成前两周', '组': str(q), '平均%': pv[str(q)].mean() * 100})
        diff = (pv[str(groups)] - pv['1']).to_numpy()
        rows.append({'窗口': f'第 {a}~{b} 日' if a > 0 else '形成前两周', '组': f'{groups} − 1', '平均%': np.nanmean(diff) * 100,
                     't': nw_t(diff, max(1, int(np.ceil((b - a + 1) / 5))))})
    return pl.DataFrame(rows)


def double_sort(w, d):
    """先按 HP̄ 分两组，再在组内按 Q 分两组；形成后 1–4、5–20、21–40 日收益（%）。"""
    d = d.sort(['product', 'trading_day']).with_columns(pl.col('trading_day').rank('dense').over('product').alias('k'))
    lr = d.select('product', 'k', (pl.col('r') + 1).log().alias('lr'))
    idx = d.select('product', pl.col('trading_day').alias('we'), pl.col('k').alias('k0'))
    mem = []
    for (we,), g in w.drop_nulls(['HPbar', 'Q']).group_by(['we'], maintain_order=True):
        if g.height < MIN_N:
            continue
        hm = g['HPbar'].median()
        for hl, gg in (('HP̄ 高', g.filter(pl.col('HPbar') > hm)), ('HP̄ 低', g.filter(pl.col('HPbar') <= hm))):
            qm = gg['Q'].median()
            for p, q in zip(gg['product'], gg['Q']):
                mem.append({'we': we, 'product': p, 'hp': hl, 'qq': 'Q 高' if q > qm else 'Q 低'})
    mem = pl.DataFrame(mem).join(idx, on=['we', 'product'])
    rows = []
    for a, b in ((1, 4), (5, 20), (21, 40)):
        x = mem.join(lr, on='product').filter((pl.col('k') >= pl.col('k0') + a) & (pl.col('k') <= pl.col('k0') + b))
        s = x.group_by(['we', 'product', 'hp', 'qq']).agg(pl.col('lr').sum().alias('cr')).group_by(['we', 'hp', 'qq']).agg(pl.col('cr').mean())
        for hp in ('HP̄ 高', 'HP̄ 低'):
            pv = s.filter(pl.col('hp') == hp).pivot(values='cr', index='we', columns='qq').sort('we')
            dd = (pv['Q 高'] - pv['Q 低']).to_numpy()
            rows.append({'窗口': f'第 {a}~{b} 日', '组': f'{hp}：Q 高 − Q 低', '平均%': np.nanmean(dd) * 100, 't': nw_t(dd, max(1, int(np.ceil((b - a + 1) / 5))))})
        pv = s.group_by(['we', 'hp']).agg(pl.col('cr').mean()).pivot(values='cr', index='we', columns='hp').sort('we')
        dd = (pv['HP̄ 高'] - pv['HP̄ 低']).to_numpy()
        rows.append({'窗口': f'第 {a}~{b} 日', '组': 'HP̄ 高 − HP̄ 低', '平均%': np.nanmean(dd) * 100, 't': nw_t(dd, max(1, int(np.ceil((b - a + 1) / 5))))})
    return pl.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
    d, basis, we = build()
    w = weekly(d, basis)
    w.write_parquet(os.path.join(OUT, 'weekly.parquet'))
    print(f"周度面板：{w.height} 品种-周，{w['we'].min()} ~ {w['we'].max()}，品种 {w['product'].n_unique()}，每周品种数中位数 {w.group_by('we').len()['len'].median():.0f}")
    print(w.select(pl.col('Q', 'HP', 'HPbar').mean().name.suffix('_均值'), pl.col('Q', 'HP').std().name.suffix('_标准差')))
    # Table II
    t2 = pl.DataFrame([{'设定': lab, **fmb(w, 'Q', xs)} for lab, xs in (('Q_t ~ R_t', ['R']), ('Q_t ~ R_{t−1}', ['Rlag']), ('Q_t ~ R_t + R_{t−1} + Q_{t−1}', ['R', 'Rlag', 'Qlag']))])
    t2.write_parquet(os.path.join(OUT, 't2.parquet')); print(t2.with_columns(pl.col(pl.Float64).round(3)))
    # Table III（收益以 % 计）
    w2 = w.with_columns((pl.col('R1') * 100).alias('R1p'), (pl.col('R2') * 100).alias('R2p'))
    rows = []
    for y, lab in (('R1p', '第 t+1 周'), ('R2p', '第 t+2 周')):
        rows.append({'因变量': lab, '控制变量': '无', **fmb(w2, y, ['Q'])})
        rows.append({'因变量': lab, '控制变量': '基差、S·特质波动、R_t', **fmb(w2, y, ['Q', 'B', 'SIV', 'R'])})
    t3 = pl.DataFrame(rows)
    t3.write_parquet(os.path.join(OUT, 't3.parquet')); print(t3.with_columns(pl.col(pl.Float64).round(3)))
    # Table V
    t5 = sorts(w, d, 'Q')
    t5.write_parquet(os.path.join(OUT, 't5.parquet')); print(t5.with_columns(pl.col(pl.Float64).round(3)))
    # Table VI
    rows = []
    for lab, xs in (('HP', ['HP', 'B', 'SIV', 'R']), ('平滑 HP̄', ['HPbar', 'B', 'SIV', 'R']), ('Q + 平滑 HP̄', ['Q', 'HPbar', 'B', 'SIV', 'R'])):
        for y, yl in (('R1p', '第 t+1 周'), ('R2p', '第 t+2 周')):
            rows.append({'设定': lab, '因变量': yl, **fmb(w2, y, xs)})
    t6 = pl.DataFrame(rows, infer_schema_length=None)
    t6.write_parquet(os.path.join(OUT, 't6.parquet')); print(t6.with_columns(pl.col(pl.Float64).round(3)))
    # Table VIII
    t8 = double_sort(w, d)
    t8.write_parquet(os.path.join(OUT, 't8.parquet')); print(t8.with_columns(pl.col(pl.Float64).round(3)))
    # 分时期（Table III 含控制）
    rows = []
    for lab, cond in (('2006–2015', pl.col('we') < pl.date(2016, 1, 1)), ('2016–2026', pl.col('we') >= pl.date(2016, 1, 1))):
        rows.append({'时期': lab, **fmb(w2.filter(cond), 'R1p', ['Q', 'B', 'SIV', 'R'])})
        rows.append({'时期': lab + '（平滑 HP̄）', **fmb(w2.filter(cond), 'R1p', ['Q', 'HPbar', 'B', 'SIV', 'R'])})
    ts = pl.DataFrame(rows, infer_schema_length=None)
    ts.write_parquet(os.path.join(OUT, 'subperiods.parquet')); print(ts.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == '__main__':
    main()
