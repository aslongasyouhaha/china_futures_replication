"""Basis-Momentum 复现的检验 —— Boons & Porras Prado (2019, Journal of Finance)

特征（月末 t）：
  基差 B = F2/F1 − 1；动量 M1 = Π_{s=t−11..t}(1 + R1_s) − 1；第二近月动量 M2 同理；
  basis-momentum BM = Π(1 + R1_s) − Π(1 + R2_s)；分解：曲率 Curv = Σ_{s=t−11..t−1}(B2_s − B_s)，斜率变化 ΔSlope = B_{t−12} − B_t（B2 = F3/F2 − 1）
收益（t+1）：近月收益 R1；价差收益 R1 − R2
检验（对应论文）：
  Table I   单变量排序：High4 / Mid / Low4（各排前、后 4 个品种），等权
  Table II  混合回归（时间、品种固定效应，按时间聚类标准误），Panel C 分解
  Table III 因子描述统计；对 Szymanowska et al. (2014) 与 Bakshi-Gao-Rossi 模型的 spanning 回归（NW 滞后 1）与 GRS
  Table IV  截面定价：A 组合层面（全样本 beta，Shanken t，MAPE）；B 品种层面（过去一年日收益滚动 beta，Fama-MacBeth）
  Table V   独立双重排序（中位数分组）：基差、动量、事后波动、过去 12 个月平均基差
  Table VIII 商品市场波动能否预测 basis-momentum 收益
  Table IA.XI 期限结构更远端
中国的必要调整：每月 F1、F2 持仓 ≥ 100 手且当天有成交才进入排序；品种池同三因子复现（主力 60 日平均成交 ≥ 1000 手），2006 年起，每月 ≥ 10 个品种。
  （持仓 ≥ 500 手时，非主力月份在 2018 年以前大多不合格，样本只剩 125 个月，作为稳健性检验 paper500）
用法: python bm.py [paper|paper500|liquid]（默认全跑，paper 为主结果）
"""
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'open_interest'))
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
from oi import sector  # noqa: E402
from pricing import nw_cov  # noqa: E402

MIN_OI = 100
VARIANTS = {'paper': ('paper', 100), 'paper500': ('paper', 500), 'liquid': ('liquid', 100)}
MIN_N = 10
K4 = 4
SPLIT = pl.date(2016, 1, 1)
SECTORS = ['农产品', '有色贵金属', '黑色', '能源化工']


# ---------------------------------------------------------------- 数据
def panel(v):
    c = pl.read_parquet(os.path.join(OUT, f'curve_{v}.parquet'))
    w = None
    for n in range(1, 5):
        x = c.filter(pl.col('n') == n).select('product', 'm', *[pl.col(k).alias(f'{k}{n}') for k in ('symbol', 'px', 'ret', 'open_interest', 'volume')])
        w = x if w is None else w.join(x, on=['product', 'm'], how='left')
    w = w.sort(['product', 'm'])
    # 过去 12 个月实现的滚动收益：形成月 s 的 ret 属于 s+1
    for n in (1, 2):
        h = w.select('product', pl.col('m').dt.offset_by('1mo').alias('m'), pl.col(f'ret{n}').alias(f'rin{n}'))
        w = w.join(h, on=['product', 'm'], how='left')
    w = w.sort(['product', 'm']).with_columns(
        (pl.col('px2') / pl.col('px1') - 1).alias('B'), (pl.col('px3') / pl.col('px2') - 1).alias('B2'))
    w = w.with_columns(
        (pl.col('rin1') + 1).log().rolling_sum(12).over('product').exp().alias('G1'),
        (pl.col('rin2') + 1).log().rolling_sum(12).over('product').exp().alias('G2'),
        (pl.col('m').shift(11).over('product') == pl.col('m').dt.offset_by('-11mo')).alias('h_ok'),
        (pl.col('B2') - pl.col('B')).shift(1).rolling_sum(11).over('product').alias('Curv'),
        (pl.col('B').shift(12).over('product') - pl.col('B')).alias('dSlope'),
        pl.col('B').rolling_mean(12).over('product').alias('B12'))
    w = w.with_columns((pl.col('G1') - 1).alias('M1'), (pl.col('G2') - 1).alias('M2'), (pl.col('G1') - pl.col('G2')).alias('BM'),
                       (pl.col('m').shift(12).over('product') == pl.col('m').dt.offset_by('-12mo')).alias('s_ok'))
    uni = pl.read_parquet(os.path.join(OUT, 'universe.parquet'))
    w = w.join(uni, on=['product', 'm'])
    w = w.filter(pl.col('h_ok') & pl.col('BM').is_not_null() & pl.col('ret1').is_not_null() & pl.col('ret2').is_not_null()
                 & (pl.col('open_interest1') >= MIN_OI) & (pl.col('open_interest2') >= MIN_OI) & (pl.col('volume1') > 0) & (pl.col('volume2') > 0))
    w = w.with_columns(pl.when(pl.col('s_ok')).then(pl.col('Curv')).alias('Curv'), pl.when(pl.col('s_ok')).then(pl.col('dSlope')).alias('dSlope'),
                       (pl.col('ret1') - pl.col('ret2')).alias('spr'),
                       pl.col('product').map_elements(sector, return_dtype=pl.Utf8).alias('sector'))
    n = w.group_by('m').agg(pl.len().alias('N'))
    return w.join(n, on='m').filter(pl.col('N') >= MIN_N).sort(['m', 'product'])


def daily(v, x):
    """持有月内逐日的近月、价差收益（只取当月进入排序的品种），以及形成月。"""
    d = pl.read_parquet(os.path.join(OUT, f'daily_{v}.parquet')).filter(pl.col('r').is_not_null())
    d = d.pivot(values='r', index=['product', 'trading_day', 'm_form'], columns='n', aggregate_function='first')
    d = d.rename({'1': 'd1', '2': 'd2'}).with_columns((pl.col('d1') - pl.col('d2')).alias('dspr'))
    return d


# ---------------------------------------------------------------- 工具
def nw_t(x, L=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / np.sqrt(nw_cov(x, L)[0, 0] / len(x))


def ann(x):
    return np.nanmean(x) * 1200


def sort_groups(g, col, k=K4):
    o = g.sort([col, 'product'], descending=[True, False])
    return o.head(k), o.slice(k, o.height - 2 * k), o.tail(k)


def sorts(x, col, k=K4, prop=None):
    """返回每月 High / Mid / Low 的近月与价差收益。prop: 按比例分组（如 0.2）。"""
    rows = []
    for (m,), g in x.group_by(['m'], maintain_order=True):
        kk = max(1, int(round(g.height * prop))) if prop else k
        if g.height < 2 * kk + 1:
            continue
        h, mid, l = sort_groups(g, col, kk)
        rows.append({'m': m, 'H_nb': h['ret1'].mean(), 'M_nb': mid['ret1'].mean(), 'L_nb': l['ret1'].mean(),
                     'H_spr': h['spr'].mean(), 'M_spr': mid['spr'].mean(), 'L_spr': l['spr'].mean()})
    s = pl.DataFrame(rows).sort('m')
    return s.with_columns((pl.col('H_nb') - pl.col('L_nb')).alias('HL_nb'), (pl.col('H_spr') - pl.col('L_spr')).alias('HL_spr'))


def stat(x):
    x = np.asarray(x, float)
    return {'年化%': ann(x), 't': nw_t(x), '夏普': np.nanmean(x) / np.nanstd(x, ddof=1) * np.sqrt(12)}


# ---------------------------------------------------------------- Table I
def table1(x):
    rows = []
    for col, lab in (('BM', 'Basis-momentum'), ('B', '基差'), ('M1', '动量')):
        s = sorts(x, col)
        for leg in ('H', 'M', 'L', 'HL'):
            for kind, kl in (('nb', '近月'), ('spr', '价差')):
                rows.append({'排序变量': lab, '组合': {'H': 'High4', 'M': 'Mid', 'L': 'Low4', 'HL': 'High4−Low4'}[leg], '收益': kl,
                             '月数': s.height, **stat(s[f'{leg}_{kind}'].to_numpy())})
    return pl.DataFrame(rows)


def table1_robust(x):
    rows = []
    for lab, kw, sub in (('High4−Low4，全样本', {}, None), ('前后 20%，全样本', {'prop': 0.2}, None),
                         ('High4−Low4，2007–2015', {}, 'a'), ('High4−Low4，2016–2026', {}, 'b')):
        xx = x if sub is None else x.filter(pl.col('m') < SPLIT) if sub == 'a' else x.filter(pl.col('m') >= SPLIT)
        for col, cl in (('BM', 'Basis-momentum'), ('B', '基差'), ('M1', '动量')):
            s = sorts(xx, col, **kw)
            for kind, kl in (('nb', '近月'), ('spr', '价差')):
                rows.append({'设定': lab, '排序变量': cl, '收益': kl, '月数': s.height, **stat(s[f'HL_{kind}'].to_numpy())})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Table II
def pooled(x, y, xs, tfe, cfe):
    d = x.drop_nulls([y] + xs)
    Y = d[y].to_numpy() * 100
    cols = [d[c].to_numpy() for c in xs]
    X = [np.ones(d.height)] + cols
    if tfe:
        X += [(d['m'] == m).to_numpy().astype(float) for m in d['m'].unique().sort()[1:]]
    if cfe:
        X += [(d['product'] == p).to_numpy().astype(float) for p in d['product'].unique().sort()[1:]]
    X = np.column_stack(X)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ Y
    u = Y - X @ b
    k = 1 + len(xs)
    g = d['m'].to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    Xu = X * u[:, None]
    for m in np.unique(g):
        s = Xu[g == m].sum(0)
        meat += np.outer(s, s)
    G = len(np.unique(g))
    V = XtXi @ meat @ XtXi * G / (G - 1)
    se = np.sqrt(np.diag(V)[1:k])
    r2 = 1 - u.var() / Y.var()
    return b[1:k], b[1:k] / se, r2, d.height


def table2(x):
    rows = []
    specs = [('(1)', ['BM'], False, False), ('(2)', ['BM'], True, False), ('(3)', ['BM'], False, True), ('(4)', ['BM'], True, True),
             ('(5)', ['B'], True, True), ('(6)', ['M1'], True, True), ('(7)', ['BM', 'B', 'M1'], True, True)]
    for y, yl in (('ret1', '近月'), ('spr', '价差')):
        for lab, xs, tf, cf in specs:
            b, t, r2, n = pooled(x, y, xs, tf, cf)
            rows.append({'收益': yl, '模型': lab, **{f'λ_{c}': b[i] for i, c in enumerate(xs)}, **{f't_{c}': t[i] for i, c in enumerate(xs)},
                         '时间FE': tf, '品种FE': cf, 'R2': r2, '观测数': n})
        for lab, sub in (('(7a) 2007–2015', x.filter(pl.col('m') < SPLIT)), ('(7b) 2016–2026', x.filter(pl.col('m') >= SPLIT))):
            b, t, r2, n = pooled(sub, y, ['BM', 'B', 'M1'], True, True)
            rows.append({'收益': yl, '模型': lab, **{f'λ_{c}': b[i] for i, c in enumerate(['BM', 'B', 'M1'])},
                         **{f't_{c}': t[i] for i, c in enumerate(['BM', 'B', 'M1'])}, '时间FE': True, '品种FE': True, 'R2': r2, '观测数': n})
        for lab, xs in (('C1 近月动量 vs 第二近月动量', ['M1', 'M2']), ('C2 曲率 vs 斜率变化', ['Curv', 'dSlope']),
                        ('C3 BM + 过去 12 个月平均基差', ['BM', 'B12']), ('C4 只用过去 12 个月平均基差', ['B12'])):
            b, t, r2, n = pooled(x, y, xs, True, True)
            rows.append({'收益': yl, '模型': lab, **{f'λ_{c}': b[i] for i, c in enumerate(xs)}, **{f't_{c}': t[i] for i, c in enumerate(xs)},
                         '时间FE': True, '品种FE': True, 'R2': r2, '观测数': n})
    sd = {c: x[c].std() for c in ('BM', 'B', 'M1', 'Curv', 'dSlope', 'B12')}
    return pl.DataFrame(rows), sd


# ---------------------------------------------------------------- 因子
def factors(x):
    bm, b, mo = sorts(x, 'BM'), sorts(x, 'B'), sorts(x, 'M1')
    b12 = sorts(x, 'B12')
    avg = x.group_by('m').agg(pl.col('ret1').mean().alias('AVG'))
    f = (bm.select('m', pl.col('HL_nb').alias('BM_nb'), pl.col('HL_spr').alias('BM_spr'))
         .join(b.select('m', pl.col('HL_nb').alias('B_nb'), pl.col('H_spr').alias('B_H4_spr'), pl.col('L_spr').alias('B_L4_spr')), on='m')
         .join(mo.select('m', pl.col('HL_nb').alias('M_nb')), on='m').join(avg, on='m')
         .join(b12.select('m', pl.col('HL_nb').alias('B12_nb')), on='m', how='left'))
    return f.sort('m')


def spanning(f):
    rows = []
    models = {'Szymanowska et al. (2014)': ['B_nb', 'B_H4_spr', 'B_L4_spr'], 'Bakshi-Gao-Rossi': ['AVG', 'B_nb', 'M_nb'],
              'BGR + 12 个月平均基差因子': ['AVG', 'B_nb', 'M_nb', 'B12_nb']}
    for name, fs in models.items():
        d = f.drop_nulls(fs + ['BM_nb', 'BM_spr'])
        F = d.select(fs).to_numpy()
        T = len(F)
        X = np.column_stack([np.ones(T), F])
        XtXi = np.linalg.inv(X.T @ X)
        al, E = [], []
        for y in ('BM_nb', 'BM_spr'):
            Y = d[y].to_numpy()
            b = XtXi @ X.T @ Y
            e = Y - X @ b
            V = XtXi @ (nw_cov(X * e[:, None], 1) * T) @ XtXi
            tb = b / np.sqrt(np.diag(V))
            rows.append({'基准模型': name, '因子': y, 'α 年化%': b[0] * 1200, 't(α)': tb[0],
                         **{f'β_{c}': b[i + 1] for i, c in enumerate(fs)}, **{f't_{c}': tb[i + 1] for i, c in enumerate(fs)},
                         'R2': 1 - e.var() / Y.var(), '月数': T})
            al.append(b[0])
            E.append(e)
        E = np.column_stack(E)
        a = np.array(al)
        N, K = 2, len(fs)
        Se = E.T @ E / (T - K - 1)
        mu = F.mean(0)
        Om = np.cov(F.T).reshape(K, K)
        grs = (T / N) * ((T - N - K) / (T - K - 1)) * (a @ np.linalg.solve(Se, a)) / (1 + mu @ np.linalg.solve(Om, mu))
        for r in rows[-2:]:
            r['GRS'] = grs
            r['GRS p'] = st.f.sf(grs, N, T - N - K)
    return pl.DataFrame(rows)


def factor_stats(f):
    cols = ['BM_nb', 'B_nb', 'AVG', 'M_nb', 'BM_spr', 'B_H4_spr', 'B_L4_spr', 'B12_nb']
    rows = []
    for c in cols:
        x = f[c].drop_nulls().to_numpy()
        rows.append({'因子': c, '年化%': ann(x), 't': nw_t(x), '年化波动%': x.std(ddof=1) * np.sqrt(12) * 100,
                     '偏度': st.skew(x), '峰度': st.kurtosis(x, fisher=False), 'AR(1)': np.corrcoef(x[1:], x[:-1])[0, 1]})
    corr = f.select(cols).drop_nulls().corr().with_columns(pl.Series('因子', cols)).select(['因子'] + cols)
    return pl.DataFrame(rows), corr


# ---------------------------------------------------------------- Table IV
MODELS = {'Szymanowska et al.': ['B_nb', 'B_H4_spr', 'B_L4_spr'], 'Bakshi-Gao-Rossi': ['AVG', 'B_nb', 'M_nb'],
          'Szymanowska + BM': ['B_nb', 'B_H4_spr', 'B_L4_spr', 'BM_nb'], 'BGR + BM': ['AVG', 'B_nb', 'M_nb', 'BM_nb'],
          'AVG + BM（两因子）': ['AVG', 'BM_nb'], 'AVG + BM + M + BM价差（四因子）': ['AVG', 'BM_nb', 'M_nb', 'BM_spr']}


def test_assets(x):
    parts = []
    for col, lab in (('BM', 'BM'), ('B', '基差'), ('M1', '动量')):
        s = sorts(x, col)
        parts.append(s.select('m', *[pl.col(f'{leg}_{k}').alias(f'{lab}{nm}_{k}') for leg, nm in (('H', 'High4'), ('M', 'Mid'), ('L', 'Low4'))
                                     for k in ('nb', 'spr')]))
    sec = x.group_by(['m', 'sector']).agg(pl.col('ret1').mean().alias('nb'), pl.col('spr').mean().alias('spr'))
    for s in SECTORS:
        parts.append(sec.filter(pl.col('sector') == s).select('m', pl.col('nb').alias(f'{s}_nb'), pl.col('spr').alias(f'{s}_spr')))
    t = parts[0]
    for p in parts[1:]:
        t = t.join(p, on='m', how='left')
    return t.sort('m')


def cs_portfolio(f, ta):
    d = ta.join(f, on='m').drop_nulls()
    assets = [c for c in ta.columns if c != 'm']
    R = d.select(assets).to_numpy()
    T, N = R.shape
    rows = []
    for name, fs in MODELS.items():
        F = d.select(fs).to_numpy()
        K = len(fs)
        X = np.column_stack([np.ones(T), F])
        B = np.linalg.lstsq(X, R, rcond=None)[0][1:].T                 # N × K
        Z = np.column_stack([np.ones(N), B])
        g_t = np.linalg.lstsq(Z, R.T, rcond=None)[0].T                 # T × (K+1)
        g = g_t.mean(0)
        Sf = np.cov(F.T).reshape(K, K)
        c = g[1:] @ np.linalg.solve(Sf, g[1:])
        Vg = np.cov(g_t.T) / T
        Vs = (1 + c) * Vg
        Vs[1:, 1:] += Sf / T
        ts = g / np.sqrt(np.diag(Vs))
        rbar = R.mean(0)
        a = rbar - Z @ g
        nb = np.array([k.endswith('_nb') for k in assets])
        rows.append({'模型': name, **{f'γ_{k}': g[i] * 1200 for i, k in enumerate(['常数'] + fs)}, **{f't_{k}': ts[i] for i, k in enumerate(['常数'] + fs)},
                     'R2': 1 - a.var() / rbar.var(), 'MAPE%': np.abs(a).mean() * 1200, 'MAPE近月%': np.abs(a[nb]).mean() * 1200,
                     'MAPE价差%': np.abs(a[~nb]).mean() * 1200, '资产数': N, '月数': T})
    return pl.DataFrame(rows)


def daily_factors(x, dd):
    """逐日因子收益：成员按形成月固定。"""
    mem = []
    for (m,), g in x.group_by(['m'], maintain_order=True):
        for col, nm in (('BM', 'BM'), ('B', 'B'), ('M1', 'M')):
            h, mid, l = sort_groups(g, col)
            mem += [{'m_form': m, 'product': p, 'f': nm, 'leg': leg} for leg, grp in (('H', h), ('L', l)) for p in grp['product']]
        mem += [{'m_form': m, 'product': p, 'f': 'AVG', 'leg': 'H'} for p in g['product']]
    mem = pl.DataFrame(mem)
    j = dd.join(mem, on=['product', 'm_form'])
    a = j.group_by(['trading_day', 'f', 'leg']).agg(pl.col('d1').mean(), pl.col('dspr').mean())
    out = None
    for f, leg, val, nm in (('BM', None, 'd1', 'BM_nb'), ('BM', None, 'dspr', 'BM_spr'), ('B', None, 'd1', 'B_nb'), ('B', 'H', 'dspr', 'B_H4_spr'),
                            ('B', 'L', 'dspr', 'B_L4_spr'), ('M', None, 'd1', 'M_nb'), ('AVG', 'H', 'd1', 'AVG')):
        q = a.filter(pl.col('f') == f)
        if leg is None:
            q = q.pivot(values=val, index='trading_day', columns='leg').select('trading_day', (pl.col('H') - pl.col('L')).alias(nm))
        else:
            q = q.filter(pl.col('leg') == leg).select('trading_day', pl.col(val).alias(nm))
        out = q if out is None else out.join(q, on='trading_day', how='outer_coalesce')
    return out.sort('trading_day')


def cs_commodity(x, dd, fd):
    """品种层面条件定价（论文 Table IV Panel B）：过去 12 个月日收益估计 beta（≥ 125 个观测），逐月 Fama-MacBeth。"""
    dd = dd.join(fd, on='trading_day').with_columns(pl.col('trading_day').dt.truncate('1mo').alias('mm'))
    months = x['m'].unique().sort().to_list()
    res = {k: [] for k in MODELS}
    per_asset = {k: {} for k in MODELS}
    by_prod = dd.partition_by(['product'], as_dict=True)
    xg = x.partition_by(['m'], as_dict=True)
    for m in months:
        g = xg[(m,)]
        lo = (m.replace(day=1) if hasattr(m, 'replace') else m)
        start = pl.Series([lo]).dt.offset_by('-11mo')[0]
        for name, fs in MODELS.items():
            Y, Bs, ids = [], [], []
            for r in g.iter_rows(named=True):
                q = by_prod.get((r['product'],))
                if q is None:
                    continue
                q = q.filter((pl.col('mm') >= start) & (pl.col('mm') <= m)).drop_nulls(fs + ['d1', 'dspr'])
                if q.height < 125:
                    continue
                F = np.column_stack([np.ones(q.height), q.select(fs).to_numpy()])
                b = np.linalg.lstsq(F, q.select('d1', 'dspr').to_numpy(), rcond=None)[0][1:]   # K × 2
                Y += [r['ret1'], r['spr']]
                Bs += [b[:, 0], b[:, 1]]
                ids += [(r['product'], 'nb'), (r['product'], 'spr')]
            if len(Y) < len(fs) + 3:
                continue
            Z = np.column_stack([np.ones(len(Y)), np.array(Bs)])
            gam = np.linalg.lstsq(Z, np.array(Y), rcond=None)[0]
            res[name].append(gam)
            for i, k in enumerate(ids):
                per_asset[name].setdefault(k, []).append((Y[i], Bs[i]))
    rows = []
    for name, fs in MODELS.items():
        G = np.array(res[name])
        T = len(G)
        g = G.mean(0)
        t = g / (G.std(0, ddof=1) / np.sqrt(T))
        ybar = np.array([np.mean([a for a, _ in v]) for v in per_asset[name].values()])
        bbar = np.array([np.mean([b for _, b in v], 0) for v in per_asset[name].values()])
        Z = np.column_stack([np.ones(len(ybar)), bbar])
        gg = np.linalg.lstsq(Z, ybar, rcond=None)[0]
        a = ybar - Z @ gg
        rows.append({'模型': name, **{f'γ_{k}': g[i] * 1200 for i, k in enumerate(['常数'] + fs)}, **{f't_{k}': t[i] for i, k in enumerate(['常数'] + fs)},
                     'R2': 1 - a.var() / ybar.var(), 'MAPE%': np.abs(a).mean() * 1200, '月数': T, '资产数': len(ybar)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Table V
def table5(x, dd):
    vol = dd.group_by(['product', 'm_form']).agg((pl.col('d1').std() * np.sqrt(21)).alias('vol')).rename({'m_form': 'm'})
    x = x.join(vol, on=['product', 'm'], how='left').with_columns((pl.col('vol') - pl.col('vol').mean().over('product')).alias('vol_dm'))
    rows = []
    ctrls = (('基差', 'B', 'zero'), ('动量', 'M1', 'median'), ('事后波动（去品种均值）', 'vol_dm', 'median'), ('过去 12 个月平均基差', 'B12', 'zero'))
    for lab, col, how in [('Basis-momentum', 'BM', 'median')] + list(ctrls):
        rec = {k: [] for k in ('H_nb', 'L_nb', 'H_spr', 'L_spr', 'HH_nb', 'HL_nb_', 'LH_nb', 'LL_nb', 'HH_spr', 'HL_spr_', 'LH_spr', 'LL_spr')}
        for (m,), g in x.drop_nulls([col]).group_by(['m'], maintain_order=True):
            cut = 0.0 if how == 'zero' else g[col].median()
            hi = g.filter(pl.col(col) > cut)
            lo = g.filter(pl.col(col) <= cut)
            if hi.height == 0 or lo.height == 0:
                continue
            rec['H_nb'].append(hi['ret1'].mean()); rec['L_nb'].append(lo['ret1'].mean())
            rec['H_spr'].append(hi['spr'].mean()); rec['L_spr'].append(lo['spr'].mean())
            if col == 'BM':
                continue
            bmc = g['BM'].median()
            for a, ga in (('H', hi), ('L', lo)):
                for b, gb in (('H', ga.filter(pl.col('BM') > bmc)), ('L', ga.filter(pl.col('BM') <= bmc))):
                    for k in ('nb', 'spr'):
                        v = gb['ret1' if k == 'nb' else 'spr'].mean() if gb.height else np.nan
                        rec[f'{a}{b}_{k}' if not (a == 'H' and b == 'L') else f'HL_{k}_'].append(v)
        for k, kl in (('nb', '近月'), ('spr', '价差')):
            H, L = np.array(rec[f'H_{k}']), np.array(rec[f'L_{k}'])
            r = {'控制变量': lab, '收益': kl, '单排序 High': ann(H), '单排序 Low': ann(L), '单排序 H−L': ann(H - L), 't(H−L)': nw_t(H - L)}
            if col != 'BM':
                for a in ('H', 'L'):
                    hh = np.array(rec[f'{a}H_{k}'])
                    ll = np.array(rec[f'HL_{k}_'] if a == 'H' else rec[f'LL_{k}'])
                    dlt = hh - ll
                    r[f'控制={a}：BM H−L'] = ann(dlt)
                    r[f'控制={a}：t'] = nw_t(dlt)
            rows.append(r)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Table VIII
def table8(x, dd, f):
    j = dd.join(x.select('product', pl.col('m').alias('m_form')), on=['product', 'm_form'])
    idx = j.group_by(['m_form', 'trading_day']).agg(pl.col('d1').mean().alias('r'))
    vm = idx.group_by('m_form').agg((pl.col('r') ** 2).sum().alias('var_mkt'))
    va = j.group_by(['m_form', 'product']).agg((pl.col('d1') ** 2).sum().alias('v')).group_by('m_form').agg(pl.col('v').mean().alias('var_avg'))
    # 持有月 = m_form 的下个月；作为“t 月方差”时对应的形成月就是 t（信号月）
    v = vm.join(va, on='m_form').with_columns(pl.col('m_form').dt.offset_by('1mo').alias('m')).select('m', 'var_mkt', 'var_avg')
    d = f.select('m', 'BM_nb', 'BM_spr').join(v, on='m', how='left').sort('m')
    rows = []
    for k in (1, 6, 12):
        for y in ('BM_nb', 'BM_spr'):
            r = d[y].to_numpy()
            cum = np.array([np.prod(1 + r[i:i + k]) - 1 if i + k <= len(r) else np.nan for i in range(len(r))])
            for vv in ('var_mkt', 'var_avg'):
                z = d[vv].to_numpy()
                ok = ~np.isnan(cum) & ~np.isnan(z)
                zz = (z[ok] - z[ok].mean()) / z[ok].std()
                Y = cum[ok] * 100
                X = np.column_stack([np.ones(ok.sum()), zz])
                XtXi = np.linalg.inv(X.T @ X)
                b = XtXi @ X.T @ Y
                e = Y - X @ b
                V = XtXi @ (nw_cov(X * e[:, None], k) * len(Y)) @ XtXi
                rows.append({'k（月）': k, '因子': y, '方差': vv, 'v1（%，每 1 个标准差）': b[1], 't': b[1] / np.sqrt(V[1, 1]), 'R2': 1 - e.var() / Y.var()})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Table IA.XI
def across_curve(v, x):
    c = pl.read_parquet(os.path.join(OUT, f'curve_{v}.parquet'))
    ok = c.filter((pl.col('open_interest') >= MIN_OI) & (pl.col('volume') > 0)).select('product', 'm', 'n', 'ret')
    w = x.select('product', 'm', 'BM')
    for n in (2, 3, 4):
        w = w.join(ok.filter(pl.col('n') == n).select('product', 'm', pl.col('ret').alias(f'r{n}')), on=['product', 'm'], how='left')
    rows = []
    for lab, expr in (('第二近月', pl.col('r2')), ('第三近月', pl.col('r3')), ('第二−第三近月', pl.col('r2') - pl.col('r3')), ('第三−第四近月', pl.col('r3') - pl.col('r4'))):
        q = w.with_columns(expr.alias('y')).drop_nulls('y')
        s = []
        for (m,), g in q.group_by(['m'], maintain_order=True):
            if g.height < MIN_N:
                continue
            h, _, l = sort_groups(g, 'BM')
            s.append(h['y'].mean() - l['y'].mean())
        rows.append({'收益': lab, '月数': len(s), **stat(np.array(s))})
    return pl.DataFrame(rows)


def main(tag):
    global MIN_OI
    v, MIN_OI = VARIANTS[tag]
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
    x = panel(v)
    print(f'\n======== {v}: {x.height} 品种-月，{x["m"].min()} ~ {x["m"].max()}，{x["m"].n_unique()} 个月，'
          f'每月品种数中位数 {x.group_by("m").len()["len"].median():.0f}，品种 {x["product"].n_unique()} 个')
    x.write_parquet(os.path.join(OUT, f'panel_{tag}.parquet'))
    t1 = table1(x); t1.write_parquet(os.path.join(OUT, f't1_{tag}.parquet'))
    print(t1.with_columns(pl.col(pl.Float64).round(2)))
    t1r = table1_robust(x); t1r.write_parquet(os.path.join(OUT, f't1r_{tag}.parquet'))
    print(t1r.with_columns(pl.col(pl.Float64).round(2)))
    t2, sd = table2(x); t2.write_parquet(os.path.join(OUT, f't2_{tag}.parquet'))
    print(t2.with_columns(pl.col(pl.Float64).round(2)))
    print('特征标准差', {k: round(s, 4) for k, s in sd.items()})
    pl.DataFrame([sd]).write_parquet(os.path.join(OUT, f'sd_{tag}.parquet'))
    f = factors(x); f.write_parquet(os.path.join(OUT, f'factors_{tag}.parquet'))
    fs, corr = factor_stats(f)
    fs.write_parquet(os.path.join(OUT, f'fstats_{tag}.parquet')); corr.write_parquet(os.path.join(OUT, f'fcorr_{tag}.parquet'))
    print(fs.with_columns(pl.col(pl.Float64).round(2))); print(corr.with_columns(pl.col(pl.Float64).round(2)))
    sp = spanning(f); sp.write_parquet(os.path.join(OUT, f'span_{tag}.parquet'))
    print(sp.with_columns(pl.col(pl.Float64).round(3)))
    if tag != 'paper':
        return
    ta = test_assets(x)
    c4a = cs_portfolio(f, ta); c4a.write_parquet(os.path.join(OUT, f't4a_{tag}.parquet'))
    print(c4a.with_columns(pl.col(pl.Float64).round(2)))
    dd = daily(v, x)
    fd = daily_factors(x, dd)
    c4b = cs_commodity(x, dd, fd); c4b.write_parquet(os.path.join(OUT, f't4b_{tag}.parquet'))
    print(c4b.with_columns(pl.col(pl.Float64).round(2)))
    t5 = table5(x, dd); t5.write_parquet(os.path.join(OUT, f't5_{tag}.parquet'))
    print(t5.with_columns(pl.col(pl.Float64).round(2)))
    t8 = table8(x, dd, f); t8.write_parquet(os.path.join(OUT, f't8_{tag}.parquet'))
    print(t8.with_columns(pl.col(pl.Float64).round(3)))
    ac = across_curve(v, x); ac.write_parquet(os.path.join(OUT, f'curve_ext_{tag}.parquet'))
    print(ac.with_columns(pl.col(pl.Float64).round(2)))


if __name__ == '__main__':
    for t in (sys.argv[1:] or list(VARIANTS)):
        main(t)
