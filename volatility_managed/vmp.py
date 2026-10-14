"""《Volatility-Managed Portfolios》复现 —— Moreira & Muir (2017, Journal of Finance)

论文方法：
  f^σ_{t+1} = c / RV²_t(f) · f_{t+1}
    f_{t+1}：因子下个月的超额收益；RV²_t：本月日收益（去月内均值）的平方和；c 使管理后组合与原组合的无条件波动相同（不影响夏普）
  回归 f^σ_{t+1} = α + β f_{t+1} + ε：α > 0 即波动率择时扩展了均值-方差前沿；评估比率 α/σ_ε，新夏普 = √(旧夏普² + 评估比率²)
  另：按上月波动分组看下月收益与波动（Figure 1）；多因子均值-方差组合（Table II）；分段；
      降低换手的变体（用波动而非方差、AR(1) 预期方差、杠杆上限 1 与 1.5）及盈亏平衡交易成本（Table IV、V）
  补充（解释商品上为何无效）：把上月方差拆成长期水平（过去 12 个月平均）与短期偏离（上月 ÷ 长期水平），分别择时；年度波动与收益的关系
中国因子（期货收益即超额收益）：沪深 300（IF）、中证 500（IC）、10 年国债（T）、商品平均（等权做多商品主力）、
  TSMOM（time_series_momentum 论文口径）、商品 carry（carry 论文口径，截面 carry1-12）
"""
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'time_series_momentum'))
import tsmom as T  # noqa: E402

CARRY_DAILY = os.path.join(HERE, '..', 'carry', 'output', 'paper_carry1_12_xs.parquet')
MIN_DAYS = 15
FACTORS = ['沪深300', '中证500', '10年国债', '商品平均', 'TSMOM', '商品carry']


def nw_cov(X, L=None):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    L = int(np.floor(4 * (n / 100) ** (2 / 9))) if L is None else L
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    for l in range(1, L + 1):
        G = Xc[l:].T @ Xc[:-l] / n
        S += (1 - l / (L + 1)) * (G + G.T)
    return S


def daily_factors():
    df = T.load_cache()
    d = {}
    for p, name in (('IF', '沪深300'), ('IC', '中证500'), ('T', '10年国债')):
        d[name] = df.filter(pl.col('product') == p).select('trading_day', 'ret')
    com = df.filter(pl.col('cls') == 'commodity').sort(['product', 'trading_day']).with_columns(
        pl.col('vol60').shift(1).over('product').alias('v_lag'), pl.col('n_obs').shift(1).over('product').alias('n_lag'))
    d['商品平均'] = (com.filter((pl.col('v_lag') >= 1000) & (pl.col('n_lag') >= 60) & pl.col('ret').is_not_null())
                       .group_by('trading_day').agg(pl.col('ret').mean()))
    d['TSMOM'] = T.portfolio(T.backtest(df, k=12, lag=0)).select('trading_day', pl.col('gross').alias('ret'))
    d['商品carry'] = pl.read_parquet(CARRY_DAILY).select('trading_day', 'ret')
    return {k: v.drop_nulls('ret').sort('trading_day') for k, v in d.items()}


def monthly(dly):
    """月度收益 f 与当月已实现方差 RV²；返回 m, f, rv2（rv2 属于月 m，用于下月仓位）。"""
    return (dly.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
               .group_by('m').agg(((1 + pl.col('ret')).product() - 1).alias('f'),
                                  ((pl.col('ret') - pl.col('ret').mean()) ** 2).sum().alias('rv2'), pl.len().alias('days'))
               .filter(pl.col('days') >= MIN_DAYS).sort('m')
               .with_columns(pl.col('rv2').shift(1).alias('rv2_lag'),
                             (pl.col('m').shift(1) == pl.col('m').dt.offset_by('-1mo')).alias('cont'))
               .filter(pl.col('cont') & pl.col('rv2_lag').is_not_null() & (pl.col('rv2_lag') > 0)))


def managed(f, w):
    """按权重 w（未归一）构造管理组合，c 使波动与原组合相同。返回 (管理收益, 归一后权重)。"""
    raw = w * f
    c = f.std() / raw.std()
    return c * raw, c * w


def regress(fs, f):
    X = np.column_stack([np.ones(len(f)), f])
    b = np.linalg.lstsq(X, fs, rcond=None)[0]
    e = fs - X @ b
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (nw_cov(X * e[:, None]) * len(f)) @ XtX
    sr_old = f.mean() / f.std() * np.sqrt(12)
    ar = b[0] / e.std() * np.sqrt(12)
    return dict(α年化=b[0] * 1200, t_α=b[0] / np.sqrt(V[0, 0]), β=b[1], R2=1 - e.var() / fs.var(), RMSE年化=e.std() * np.sqrt(12) * 100,
                评估比率=ar, 原夏普=sr_old, 管理后夏普=fs.mean() / fs.std() * np.sqrt(12), 新夏普=np.sqrt(sr_old ** 2 + ar ** 2), 月数=len(f))


def table1(mon):
    rows = []
    for name in FACTORS:
        x = mon[name]
        f = x['f'].to_numpy()
        fs, _ = managed(f, 1 / x['rv2_lag'].to_numpy())
        rows.append({'因子': name, '起始': str(x['m'].min()), **regress(fs, f)})
    return pl.DataFrame(rows)


def figure1(mon):
    rows = []
    for name in FACTORS:
        x = mon[name].with_columns(pl.col('rv2_lag').qcut(5, labels=['最低', '2', '3', '4', '最高']).alias('q'))
        nxt = x.with_columns((pl.col('rv2') * 12).sqrt().alias('vol_next'))
        g = nxt.group_by('q').agg((pl.col('f').mean() * 12).alias('平均收益'), (pl.col('f').std() * np.sqrt(12)).alias('收益波动'),
                                   pl.col('vol_next').mean().alias('当月已实现波动')).sort('q')
        g = g.with_columns((pl.col('平均收益') / pl.col('收益波动') ** 2).alias('收益÷方差'))
        rows.append(g.with_columns(pl.lit(name).alias('因子')))
    return pl.concat(rows).select('因子', pl.col('q').cast(pl.Utf8).alias('上月波动分组'), '平均收益', '收益波动', '当月已实现波动', '收益÷方差')


def mve(dly, names, start=None):
    """样本内最大夏普的静态组合（论文 Table II），再做波动率管理。"""
    j = None
    for n in names:
        s = dly[n].rename({'ret': n})
        j = s if j is None else j.join(s, on='trading_day', how='inner')
    if start:
        j = j.filter(pl.col('trading_day') >= start)
    m = (j.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
          .group_by('m').agg(*[((1 + pl.col(n)).product() - 1).alias(n) for n in names], pl.len().alias('days')).filter(pl.col('days') >= MIN_DAYS).sort('m'))
    F = m.select(names).to_numpy()
    b = np.linalg.solve(np.cov(F.T), F.mean(0))
    b = b / np.abs(b).sum()
    dm = j.with_columns(pl.sum_horizontal([pl.col(n) * b[i] for i, n in enumerate(names)]).alias('ret')).select('trading_day', 'ret')
    x = monthly(dm)
    f = x['f'].to_numpy()
    fs, _ = managed(f, 1 / x['rv2_lag'].to_numpy())
    return dict(组合=' + '.join(names), 权重='，'.join(f'{n} {w:+.2f}' for n, w in zip(names, b)), 起始=str(x['m'].min()), **regress(fs, f))


def variants(x):
    """论文 Table IV、V：降低换手的变体；换手 = 月均 |Δw|，盈亏平衡成本 = α ÷ (12 × 换手)，单位 bp。"""
    f = x['f'].to_numpy()
    rv = x['rv2_lag'].to_numpy()
    lrv = np.log(rv)
    b, a = np.polyfit(lrv[:-1], lrv[1:], 1)                              # AR(1)：log RV²（全样本，同论文）
    ev = np.exp(b * lrv + a + 0.5 * np.var(lrv[1:] - (b * lrv[:-1] + a)))
    out = []
    base_w = None
    for name, w, cap in (('1/方差（基准）', 1 / rv, None), ('1/波动', 1 / np.sqrt(rv), None), ('1/AR(1) 预期方差', 1 / ev, None),
                         ('1/方差，权重上限 1', 1 / rv, 1.0), ('1/方差，权重上限 1.5', 1 / rv, 1.5)):
        fs, wn = managed(f, w)
        if cap is not None:
            wn = np.minimum(base_w, cap)
            fs = wn * f
        if base_w is None:
            base_w = wn
        r = regress(fs, f)
        turn = np.abs(np.diff(wn)).mean()
        out.append({'做法': name, '权重中位数': float(np.median(wn)), '权重 90%分位': float(np.percentile(wn, 90)), '权重 99%分位': float(np.percentile(wn, 99)),
                    '月均换手 |Δw|': float(turn), 'α年化': r['α年化'], 't_α': r['t_α'], '管理后夏普': r['管理后夏普'], '原夏普': r['原夏普'],
                    '盈亏平衡成本bp': r['α年化'] / 100 / (12 * turn) * 1e4 if turn > 0 else np.nan})
    return pl.DataFrame(out)


def subperiods(mon):
    rows = []
    for name in FACTORS:
        x = mon[name]
        mid = x['m'][len(x) // 2]
        for lab, q in (('前半', x.filter(pl.col('m') < mid)), ('后半', x.filter(pl.col('m') >= mid))):
            f = q['f'].to_numpy()
            fs, _ = managed(f, 1 / q['rv2_lag'].to_numpy())
            r = regress(fs, f)
            rows.append({'因子': name, '时段': f"{lab} {q['m'].min()}~{q['m'].max()}", 'α年化': r['α年化'], 't_α': r['t_α'],
                         '原夏普': r['原夏普'], '管理后夏普': r['管理后夏普']})
    return pl.DataFrame(rows)


def regime_split(mon):
    """上月方差 = 长期水平（过去 12 个月平均方差）× 短期偏离（上月 ÷ 长期水平）。分别只用一部分调整仓位。"""
    rows, years = [], []
    for name in FACTORS:
        x = mon[name].sort('m').with_columns(pl.col('rv2_lag').rolling_mean(12, min_periods=12).alias('base')).drop_nulls('base')
        f = x['f'].to_numpy()
        for lab, w in (('论文做法（1/上月方差）', 1 / x['rv2_lag'].to_numpy()), ('只用长期水平', 1 / x['base'].to_numpy()),
                       ('只用短期偏离', x['base'].to_numpy() / x['rv2_lag'].to_numpy())):
            fs, _ = managed(f, w)
            r = regress(fs, f)
            rows.append({'因子': name, '做法': lab, 'α年化': r['α年化'], 't_α': r['t_α'], '原夏普': r['原夏普'], '管理后夏普': r['管理后夏普'], '月数': r['月数']})
        y = (mon[name].group_by(pl.col('m').dt.year().alias('年')).agg(pl.col('rv2').mean().sqrt().alias('v'), ((1 + pl.col('f')).product() - 1).alias('r'),
                                                                      pl.len().alias('k')).filter(pl.col('k') >= 10))
        years.append({'因子': name, '年度波动与当年收益相关': float(np.corrcoef(y['v'], y['r'])[0, 1]), '年数': y.height})
    return pl.DataFrame(rows), pl.DataFrame(years)


def low_vol_years(mon, name='商品平均'):
    x = mon[name].with_columns(pl.col('rv2_lag').qcut(5, labels=['最低', '2', '3', '4', '最高']).alias('q'))
    return (x.group_by([pl.col('m').dt.year().alias('年'), 'q']).agg(pl.len()).pivot(index='年', columns='q', values='len')
             .fill_null(0).sort('年').select('年', '最低', '2', '3', '4', '最高'))


def main():
    os.makedirs(OUT, exist_ok=True)
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(240)
    dly = daily_factors()
    mon = {k: monthly(v) for k, v in dly.items()}
    for k, v in mon.items():
        v.with_columns(pl.lit(k).alias('因子')).write_parquet(os.path.join(OUT, f'monthly_{k}.parquet'))
    t1 = table1(mon)
    t1.write_parquet(os.path.join(OUT, 'table1.parquet'))
    print('== Table I：管理后组合对原因子回归 ==')
    print(t1.with_columns(pl.exclude('因子', '起始', '月数').round(2)))
    f1 = figure1(mon)
    f1.write_parquet(os.path.join(OUT, 'figure1.parquet'))
    print('\n== Figure 1：按上月波动分组，下月的收益、波动（年化）==')
    print(f1.with_columns(pl.exclude('因子', '上月波动分组').round(3)))
    t2 = pl.DataFrame([mve(dly, ['沪深300', '商品平均', 'TSMOM', '商品carry']),
                       mve(dly, ['沪深300', '中证500', '10年国债', '商品平均', 'TSMOM', '商品carry']),
                       mve(dly, ['商品平均', 'TSMOM', '商品carry'])])
    t2.write_parquet(os.path.join(OUT, 'table2.parquet'))
    print('\n== Table II：多因子均值-方差组合 ==')
    print(t2.with_columns(pl.exclude('组合', '权重', '起始', '月数').round(2)))
    sp = subperiods(mon)
    sp.write_parquet(os.path.join(OUT, 'subperiods.parquet'))
    print('\n== 分段 ==')
    print(sp.with_columns(pl.exclude('因子', '时段').round(2)))
    vs = []
    for name in FACTORS:
        v = variants(mon[name]).with_columns(pl.lit(name).alias('因子'))
        vs.append(v)
    vs = pl.concat(vs)
    vs.write_parquet(os.path.join(OUT, 'variants.parquet'))
    rs, ry = regime_split(mon)
    rs.write_parquet(os.path.join(OUT, 'regime_split.parquet'))
    ry.write_parquet(os.path.join(OUT, 'annual_vol_return.parquet'))
    low_vol_years(mon).write_parquet(os.path.join(OUT, 'commodity_vol_years.parquet'))
    print('\n== 补充：长期水平 vs 短期偏离 ==')
    print(rs.with_columns(pl.exclude('因子', '做法', '月数').round(2)))
    print(ry.with_columns(pl.col('年度波动与当年收益相关').round(2)))
    print('\n== Table IV/V：降低换手与杠杆的变体 ==')
    print(vs.select('因子', *[c for c in vs.columns if c != '因子']).with_columns(pl.exclude('因子', '做法').round(2)))


if __name__ == '__main__':
    main()
