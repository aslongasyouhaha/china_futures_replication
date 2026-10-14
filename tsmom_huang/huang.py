"""《Time Series Momentum: Is It There?》复现 —— Huang, Li, Wang & Zhou (2020, Journal of Financial Economics)

资产与收益（与本库 TSMOM 复现相同）：全部期货品种的主力连续后复权价，日收益连乘成月收益；
  事前年化波动 σ = 日收益 EWMA（中心质量 60 天）；月末主力 60 日平均成交 ≥ 1000 手、已有 60 个交易日才纳入下个月。
  资产类别：商品、股指（IF/IH/IC/IM）、国债（T/TF/TS/TL）；中国没有外汇期货。
检验（对应论文）：
  Table 2  逐个资产回归 r_{t+1} = α + β r_{t−12,t}：NW t、R²；样本外 R²_OS（相对历史均值，Clark-West 检验）
  Table 3  各资产均值 / 夏普是否相同：ANOVA、Welch ANOVA、Kruskal-Wallis、自助法
  Table 4  混合回归（按月聚类）r_{t+1}/σ_t 对 滞后 h 月收益 / 过去 h 月累计收益（及其符号）；wild 与 pairs 自助法 97.5% 分位数
  Table 5–8 分资产类别、不做波动率缩放、子样本、控制固定效应
  Fig. 4   混合回归的样本外 R²_OS
  Table 9–10 TSM 与 TSH（按历史均值符号）在单个资产和组合层面的比较；四种权重
  Table 11 Lewellen 预测斜率
  Fig. 5   模拟：不同真实可预测性下 TSM 能否胜过 TSH
关于自助法：论文按“估计出的模型”生成样本，取模拟 t 值的 97.5% 分位数作为临界值（论文口径，下文 'paper'）；
  这样的分布以真实 t 值为中心，临界值天然大于真实 t。另报告标准做法：以 (β* − β̂)/se* 的分布检验 β = 0（下文 'centered'）。
"""
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'time_series_momentum'))
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
import tsmom as T  # noqa: E402
from pricing import nw_cov  # noqa: E402

N_BOOT = 1000
N_SIM = 1000
SPLIT = 2016
CLS_CN = {'commodity': '商品', 'equity': '股指', 'bond': '国债'}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- 数据
def monthly():
    df = T.load_cache()
    m = (df.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'), pl.col('ret').fill_null(0.0))
           .group_by(['product', 'm']).agg(((pl.col('ret') + 1).product() - 1).alias('r'), pl.col('cls').first(),
                                           pl.col('sigma').sort_by('trading_day').last(), pl.col('vol60').sort_by('trading_day').last(),
                                           pl.col('n_obs').sort_by('trading_day').last(), pl.len().alias('nd'))
           .sort(['product', 'm']))
    m = m.with_columns(((pl.col('vol60') >= T.MIN_VOL_LOTS) & (pl.col('n_obs') >= T.MIN_OBS) & (pl.col('sigma') > 0)).alias('ok'))
    months = pl.date_range(m['m'].min(), m['m'].max(), '1mo', eager=True)
    prods = m['product'].unique().sort().to_list()
    grid = pl.DataFrame({'m': months}).join(pl.DataFrame({'product': prods}), how='cross')
    g = grid.join(m, on=['product', 'm'], how='left').sort(['m', 'product'])
    shape = (len(months), len(prods))
    R = g['r'].to_numpy().reshape(shape).astype(float)
    S = g['sigma'].to_numpy().reshape(shape).astype(float)
    OK = g['ok'].fill_null(False).to_numpy().reshape(shape)
    cls = m.group_by('product').agg(pl.col('cls').first()).sort('product')['cls'].to_list()
    # 第一个月（品种刚上市）不完整，去掉
    first = np.argmax(~np.isnan(R), axis=0)
    for j, f in enumerate(first):
        R[f, j] = np.nan
    return months.to_list(), prods, np.array(cls), R, S, OK


def cumret(R, h):
    """C[t] = 月 t−h+1..t 的累计收益（含 t）。"""
    L = np.log1p(R)
    c = np.full_like(R, np.nan)
    cs = np.nancumsum(np.nan_to_num(L), axis=0)
    cnt = np.cumsum(~np.isnan(L), axis=0)
    for t in range(h - 1, len(R)):
        s = cs[t] - (cs[t - h] if t >= h else 0)
        n = cnt[t] - (cnt[t - h] if t >= h else 0)
        c[t] = np.where(n == h, np.expm1(s), np.nan)
    return c


def obs(R, S, OK, kind, h, scale=True, sign=False, cols=None, t_from=None, t_to=None):
    """返回 y、x、月份索引、资产索引（观测为 (t+1, i)，资产在 t 月末可交易）。"""
    Tn, N = R.shape
    if kind == 'lag':                       # r_{t−h+1} / σ_{t−h}
        X = np.full_like(R, np.nan)
        X[h - 1:] = R[:Tn - h + 1] if h > 1 else R
        Sx = np.full_like(R, np.nan)
        Sx[h:] = S[:Tn - h]
    else:                                   # r_{t−h,t} / σ_{t−1}
        X = cumret(R, h)
        Sx = np.full_like(R, np.nan)
        Sx[1:] = S[:-1]
    Y = np.full_like(R, np.nan)
    Y[:-1] = R[1:]
    if scale:
        Y = Y / S
        X = X / Sx
    if sign:
        X = np.where(np.isnan(X), np.nan, np.where(X >= 0, 1.0, -1.0))
    msk = OK & ~np.isnan(Y) & ~np.isnan(X)
    if cols is not None:
        cm = np.zeros(N, bool)
        cm[cols] = True
        msk &= cm[None, :]
    if t_from is not None:
        msk[:t_from] = False
    if t_to is not None:
        msk[t_to:] = False
    tt, ii = np.nonzero(msk)
    return Y[tt, ii], X[tt, ii], tt, ii


# ---------------------------------------------------------------- 回归与自助法
def ols_cl(y, x, g, fe=None):
    """y = a + b x（fe 给定时先按资产去均值、无常数）；按月聚类的 t。"""
    if fe is not None:
        y = y - (np.bincount(fe, y) / np.bincount(fe))[fe]
        x = x - (np.bincount(fe, x) / np.bincount(fe))[fe]
        b = (x @ y) / (x @ x)
        u = y - b * x
        s = np.bincount(g, x * u)
        V = (s @ s) / (x @ x) ** 2
        return 0.0, b, b / np.sqrt(V), u
    xm, ym = x.mean(), y.mean()
    b = ((x - xm) @ (y - ym)) / ((x - xm) @ (x - xm))
    a = ym - b * xm
    u = y - a - b * x
    X = np.column_stack([np.ones_like(x), x])
    A = np.linalg.inv(X.T @ X)
    s0 = np.bincount(g, u)
    s1 = np.bincount(g, x * u)
    M = np.array([[s0 @ s0, s0 @ s1], [s0 @ s1, s1 @ s1]])
    V = A @ M @ A
    G = len(np.unique(g))
    V *= G / (G - 1)
    return a, b, b / np.sqrt(V[1, 1]), u


def boot(y, x, g, fe=None, n=N_BOOT):
    """论文口径：按估计出的模型（wild）或成对重抽（pairs）生成样本，取模拟 t 的 97.5% 分位数；
    中心化：同一批样本的 (β* − β̂)/se*，其 97.5% 分位数是检验 β = 0 的临界值，p 为 P(中心化 t* ≥ 真实 t)。
    控制固定效应时（Gonçalves-Kaffo 做法），先按资产去均值，再对去均值后的数据做自助法。"""
    if fe is not None:
        y = y - (np.bincount(fe, y) / np.bincount(fe))[fe]
        x = x - (np.bincount(fe, x) / np.bincount(fe))[fe]
    a, b, t, u = ols_cl(y, x, g, fe)
    N = len(y)
    res = {}
    for kind in ('wild', 'pairs'):
        tp, tc = np.empty(n), np.empty(n)
        for k in range(n):
            if kind == 'wild':
                v = RNG.choice([-1.0, 1.0], N)
                ys = a + b * x + u * v
                _, bs, ts, _ = ols_cl(ys, x, g, fe)
            else:
                idx = RNG.integers(0, N, N)
                _, gi = np.unique(g[idx], return_inverse=True)
                _, bs, ts, _ = ols_cl(y[idx], x[idx], gi, None if fe is None else np.unique(fe[idx], return_inverse=True)[1])
            se = bs / ts
            tp[k], tc[k] = ts, (bs - b) / se
        res[kind] = (np.quantile(tp, 0.975), np.quantile(tc, 0.975), np.mean(tc >= t))
    return b, t, res


def boot_cluster(y, x, g, fe=None, n=N_BOOT):
    """按月整块自助（保留同一个月内各资产之间的相关性）：
    wild cluster —— 每个月一个 Rademacher 符号，并施加原假设 β = 0（Cameron-Gelbach-Miller 2008 的 WCR 做法）；
    pairs cluster —— 有放回地重抽整个月，用 (β* − β̂)/se* 的分布。均返回 97.5% 临界值与单侧 p 值 P(t* ≥ 真实 t)。"""
    if fe is not None:
        y = y - (np.bincount(fe, y) / np.bincount(fe))[fe]
        x = x - (np.bincount(fe, x) / np.bincount(fe))[fe]
    a, b, t, u = ols_cl(y, x, g, fe)
    G = g.max() + 1
    # 原假设下的残差
    if fe is None:
        a0 = y.mean()
        u0 = y - a0
    else:
        a0, u0 = 0.0, y
    order = np.argsort(g, kind='stable')
    starts = np.searchsorted(g[order], np.arange(G))
    ends = np.searchsorted(g[order], np.arange(G), side='right')
    res = {}
    tw, tpz = np.empty(n), np.empty(n)
    for k in range(n):
        v = RNG.choice([-1.0, 1.0], G)[g]
        _, bs, ts, _ = ols_cl(a0 + u0 * v, x, g, fe)
        tw[k] = ts
        pick = RNG.integers(0, G, G)
        idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
        gi = np.repeat(np.arange(G), ends[pick] - starts[pick])
        _, bs, ts, _ = ols_cl(y[idx], x[idx], gi, None if fe is None else np.unique(fe[idx], return_inverse=True)[1])
        tpz[k] = (bs - b) / (bs / ts)
    res['wild'] = (np.quantile(tw, 0.975), np.mean(tw >= t))
    res['pairs'] = (np.quantile(tpz, 0.975), np.mean(tpz >= t))
    return b, t, res


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


def clark_west(y, f, fb):
    adj = (y - fb) ** 2 - ((y - f) ** 2 - (fb - f) ** 2)
    t = nw_t(adj)
    return 1 - ((y - f) ** 2).sum() / ((y - fb) ** 2).sum(), st.norm.sf(t)


# ---------------------------------------------------------------- Table 2 与 Fig. 1
def table2(months, prods, cls, R, OK):
    C12 = cumret(R, 12)
    rows = []
    for j, p in enumerate(prods):
        y_all = np.full(len(R), np.nan)
        y_all[:-1] = R[1:, j]
        m = OK[:, j] & ~np.isnan(y_all) & ~np.isnan(C12[:, j])
        idx = np.nonzero(m)[0]
        if len(idx) < 60:
            continue
        y, x = y_all[idx], C12[idx, j]
        b, t, r2 = nw_reg(y, x)
        K = max(60, len(idx) // 2)
        f, fb, yy = [], [], []
        for k in range(K, len(idx)):
            bb = np.polyfit(x[:k], y[:k], 1)
            f.append(np.polyval(bb, x[k]))
            fb.append(y[:k].mean())
            yy.append(y[k])
        r2os, p_cw = clark_west(np.array(yy), np.array(f), np.array(fb)) if len(yy) >= 24 else (np.nan, np.nan)
        rows.append({'品种': p, '类别': CLS_CN[cls[j]], '月数': len(idx), '起点': str(months[idx[0] + 1]), 'β': b[1], 't': t[1], 'R2%': r2 * 100,
                     'R2_OS%': r2os * 100, 'CW p': p_cw, '样本外月数': len(yy)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Table 3
def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups])
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = (w * m).sum() / w.sum()
    A = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = A / B
    return st.f.sf(F, k - 1, (k ** 2 - 1) / (3 * lam))


def table3(R, OK):
    out = []
    for lab, scale in (('均值', False), ('夏普', True)):
        groups = []
        for j in range(R.shape[1]):
            okn = np.zeros(len(R), bool)
            okn[1:] = OK[:-1, j]
            r = R[okn & ~np.isnan(R[:, j]), j]
            if len(r) >= 60:
                groups.append(r / r.std() if scale else r)
        F = st.f_oneway(*groups).pvalue
        W = welch_anova(groups)
        K = st.kruskal(*groups).pvalue
        allr = np.concatenate([g - g.mean() for g in groups])
        stat0 = st.f_oneway(*groups).statistic
        cnt = 0
        for _ in range(N_BOOT):
            gs = [RNG.choice(allr, len(g)) for g in groups]
            cnt += st.f_oneway(*gs).statistic >= stat0
        out.append({'检验对象': lab, '资产数': len(groups), 'ANOVA p': F, 'Welch ANOVA p': W, 'Kruskal-Wallis p': K, '自助法 p': cnt / N_BOOT})
    return pl.DataFrame(out)


# ---------------------------------------------------------------- Table 4–8
def pooled_tables(months, cls, R, S, OK, cluster=False):
    rows = []
    yrs = np.array([m.year for m in months])
    t_split = int(np.argmax(yrs >= SPLIT))
    specs = []
    for kind, hs in (('lag', (1, 3, 6, 12)), ('cum', (1, 3, 6, 12))):
        for h in hs:
            for sg in (False, True):
                specs.append(('全部资产', kind, h, sg, True, None, None, None, False))
    for c in ('commodity', 'equity', 'bond'):
        cols = np.nonzero(cls == c)[0]
        for h in (1, 3, 6, 12):
            for sg in (False, True):
                specs.append((CLS_CN[c], 'cum', h, sg, True, cols, None, None, False))
    for h in (1, 3, 6, 12):
        for sg in (False, True):
            specs.append(('不做波动率缩放', 'cum', h, sg, False, None, None, None, False))
            specs.append((f'{months[0].year}–{SPLIT - 1}', 'cum', h, sg, True, None, None, t_split, False))
            specs.append((f'{SPLIT}–{months[-1].year}', 'cum', h, sg, True, None, t_split, None, False))
            specs.append(('控制固定效应', 'cum', h, sg, True, None, None, None, True))
    for lab, kind, h, sg, sc, cols, t0, t1, fe in specs:
        y, x, tt, ii = obs(R, S, OK, kind, h, sc, sg, cols, t0, t1)
        _, g = np.unique(tt, return_inverse=True)
        if cluster:
            b, t, res = boot_cluster(y, x, g, np.unique(ii, return_inverse=True)[1] if fe else None)
            rows.append({'样本': lab, '预测变量': ('滞后 h 月收益' if kind == 'lag' else '过去 h 月累计收益') + ('（符号）' if sg else ''), 'h': h,
                         '观测数': len(y), 'β': b, 't': t, 'wild cluster 临界值': res['wild'][0], 'wild cluster p': res['wild'][1],
                         'pairs cluster 临界值': res['pairs'][0], 'pairs cluster p': res['pairs'][1]})
            print('cluster', lab, kind, h, sg, round(t, 2), flush=True)
            continue
        b, t, res = boot(y, x, g, ii if fe else None)
        rows.append({'样本': lab, '预测变量': ('滞后 h 月收益' if kind == 'lag' else '过去 h 月累计收益') + ('（符号）' if sg else ''), 'h': h,
                     '观测数': len(y), 'β': b, 't': t, 'wild 论文口径': res['wild'][0], 'pairs 论文口径': res['pairs'][0],
                     'wild 中心化': res['wild'][1], 'pairs 中心化': res['pairs'][1], 'wild p': res['wild'][2], 'pairs p': res['pairs'][2]})
        print(lab, kind, h, sg, round(t, 2), flush=True)
    return pl.DataFrame(rows)


def fig3(R, S, OK):
    rows = []
    for h in range(1, 61):
        for sg in (False, True):
            y, x, tt, ii = obs(R, S, OK, 'lag', h, True, sg)
            _, g = np.unique(tt, return_inverse=True)
            _, b, t, _ = ols_cl(y, x, g)
            rows.append({'滞后月数': h, '符号': sg, 't': t})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Fig. 4：混合回归样本外
def pooled_oos(months, prods, cls, R, S, OK):
    C12 = cumret(R, 12)
    yrs = np.array([m.year for m in months])
    t0 = int(np.argmax(yrs >= SPLIT))
    preds = {sc: np.full_like(R, np.nan) for sc in (True, False)}
    for t in range(t0, len(R) - 1):
        for sc in (True, False):
            y, x, tt, ii = obs(R, S, OK, 'cum', 12, sc, False, t_to=t)     # 只用到 t 月为止已实现的观测
            X = np.column_stack([np.ones_like(x), x])
            a, b = np.linalg.lstsq(X, y, rcond=None)[0]
            xs = C12[t] / (S[t - 1] if sc else 1)
            preds[sc][t + 1] = (a + b * xs) * (S[t] if sc else 1)
    rows = []
    for j, p in enumerate(prods):
        okn = np.zeros(len(R), bool)
        okn[1:] = OK[:-1, j]
        rec = {}
        for sc in (True, False):
            f, fb, yy = [], [], []
            for t in range(t0 + 1, len(R)):
                if not okn[t] or np.isnan(R[t, j]) or np.isnan(preds[sc][t, j]):
                    continue
                hist = R[:t, j][okn[:t] & ~np.isnan(R[:t, j])]
                if len(hist) < 24:
                    continue
                f.append(preds[sc][t, j]); fb.append(hist.mean()); yy.append(R[t, j])
            if len(yy) >= 24:
                r2, pv = clark_west(np.array(yy), np.array(f), np.array(fb))
                rec[sc] = (r2 * 100, pv, len(yy))
        if rec:
            rows.append({'品种': p, '类别': CLS_CN[cls[j]], 'R2_OS% 缩放': rec.get(True, (np.nan,) * 3)[0], 'CW p 缩放': rec.get(True, (np.nan,) * 3)[1],
                         'R2_OS% 不缩放': rec.get(False, (np.nan,) * 3)[0], 'CW p 不缩放': rec.get(False, (np.nan,) * 3)[1], '月数': rec.get(True, (0, 0, 0))[2]})
    return pl.DataFrame(rows), preds


# ---------------------------------------------------------------- Table 9–10：TSM vs TSH
def signals(R, OK):
    C12 = cumret(R, 12)
    Ln = np.log1p(np.nan_to_num(R))
    hist = np.cumsum(Ln, axis=0)                  # 从第一个月到 t 的累计（对数），符号同历史均值
    cnt = np.cumsum(~np.isnan(R), axis=0)
    H = np.where(cnt >= 12, hist, np.nan)
    Y = np.full_like(R, np.nan)
    Y[:-1] = R[1:]
    valid = OK & ~np.isnan(Y) & ~np.isnan(C12) & ~np.isnan(H)
    return C12, H, Y, valid


def table9(months, prods, cls, R, OK):
    C12, H, Y, valid = signals(R, OK)
    rows = []
    for j, p in enumerate(prods):
        m = valid[:, j]
        if m.sum() < 60:
            continue
        y = Y[m, j]
        a = np.sign(np.where(C12[m, j] >= 0, 1, -1)) * y
        b = np.where(H[m, j] >= 0, 1, -1) * y
        d = a - b
        sa, sb = a.mean() / a.std(), b.mean() / b.std()
        rho = np.corrcoef(a, b)[0, 1]
        n = len(y)
        v = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / n      # Jobson-Korkie（Memmel 修正）
        rows.append({'品种': p, '类别': CLS_CN[cls[j]], '月数': n, 'TSM 月均%': a.mean() * 100, 'TSH 月均%': b.mean() * 100,
                     'TSM 夏普': sa, 'TSH 夏普': sb, '收益差%': d.mean() * 100, '收益差 p': 2 * st.norm.sf(abs(nw_t(d))) if d.std() > 0 else 1.0,
                     '夏普差': sa - sb, '夏普差 p': 2 * st.norm.sf(abs((sa - sb) / np.sqrt(v))) if v > 0 else 1.0})
    return pl.DataFrame(rows)


def bench(months):
    vm = pl.read_parquet(os.path.join(HERE, '..', 'value_momentum', 'output', 'portfolios.parquet'))
    vm = vm.select(pl.col('m').dt.offset_by('1mo').alias('m'), pl.col('V_F').alias('VAL'), pl.col('M_F').alias('MOM'))
    return vm


def table10(months, prods, cls, R, S, OK):
    C12, H, Y, valid = signals(R, OK)
    mkt = np.array([np.nanmean(np.where(valid[t], Y[t], np.nan)) if valid[t].any() else np.nan for t in range(len(R))])
    rec = []
    for wname in ('等权', '波动率加权（40%/σ）', '按过去收益加权', '多空各 1 元（Goyal-Jegadeesh）'):
        for sname, sig in (('TSM', C12), ('TSH', H)):
            legs = {'多头': [], '空头': [], '多 − 空': []}
            for t in range(len(R)):
                m = valid[t]
                if m.sum() < 5:
                    for k in legs:
                        legs[k].append(np.nan)
                    continue
                s = np.where(sig[t, m] >= 0, 1.0, -1.0)
                y = Y[t, m]
                n = m.sum()
                if wname == '等权':
                    w = s / n
                elif wname.startswith('波动率'):
                    w = s * 0.4 / S[t, m] / n
                elif wname.startswith('按过去'):
                    z = sig[t, m] if sname == 'TSM' else sig[t, m] / np.maximum(1, np.cumsum(~np.isnan(R), axis=0)[t, m])
                    w = z / np.abs(z).sum()
                else:
                    nl, ns = (s > 0).sum(), (s < 0).sum()
                    w = np.where(s > 0, 1 / max(nl, 1), -1 / max(ns, 1))
                legs['多头'].append((w[w > 0] * y[w > 0]).sum())
                legs['空头'].append((w[w < 0] * y[w < 0]).sum())
                legs['多 − 空'].append((w * y).sum())
            for k, v in legs.items():
                rec.append({'权重': wname, '策略': sname, '腿': k, 'r': np.array(v)})
    # 基准：市场（全部资产等权做多）；AMP 中国版（市场 + 价值 + 动量，商品）
    bm = pl.DataFrame({'m': [months[t + 1] if t + 1 < len(months) else None for t in range(len(R))], 'MKT': mkt})
    bm = bm.join(bench(months), on='m', how='left')
    rows = []
    for r in rec:
        x = r['r']
        okm = ~np.isnan(x)
        row = {'权重': r['权重'], '策略': r['策略'], '腿': r['腿'], '月均%': np.nanmean(x) * 100, 't': nw_t(x[okm])}
        for bname, fs in (('市场', ['MKT']), ('AMP 中国版', ['MKT', 'VAL', 'MOM'])):
            F = bm.select(fs).to_numpy()
            mm = okm & ~np.isnan(F).any(1)
            b, t, r2 = nw_reg(x[mm] * 100, F[mm])
            row[f'α% {bname}'] = b[0]
            row[f't(α) {bname}'] = t[0]
        rows.append(row)
    out = pl.DataFrame(rows)
    diffs = []
    for wname in out['权重'].unique(maintain_order=True):
        a = [r['r'] for r in rec if r['权重'] == wname and r['策略'] == 'TSM' and r['腿'] == '多 − 空'][0]
        b = [r['r'] for r in rec if r['权重'] == wname and r['策略'] == 'TSH' and r['腿'] == '多 − 空'][0]
        d = a - b
        okm = ~np.isnan(d)
        row = {'权重': wname, '收益差%': np.nanmean(d) * 100, '收益差 p': 2 * st.norm.sf(abs(nw_t(d[okm])))}
        for bname, fs in (('市场', ['MKT']), ('AMP 中国版', ['MKT', 'VAL', 'MOM'])):
            F = bm.select(fs).to_numpy()
            mm = okm & ~np.isnan(F).any(1)
            bb, tt, _ = nw_reg(d[mm] * 100, F[mm])
            row[f'α 差% {bname}'] = bb[0]
            row[f'α 差 p {bname}'] = 2 * st.norm.sf(abs(tt[0]))
        diffs.append(row)
    return out, pl.DataFrame(diffs)


# ---------------------------------------------------------------- Table 11：Lewellen 斜率
def table11(cls, R, OK, preds):
    C12, H, Y, valid = signals(R, OK)
    cnt = np.cumsum(~np.isnan(R), axis=0)
    hm = H / np.maximum(cnt, 1)                     # 历史平均对数收益 ≈ TSH 预测
    rows = []
    for sc in (True, False):
        P = preds[sc]
        for lab, cs in (('全部', None), ('商品', 'commodity'), ('股指', 'equity'), ('国债', 'bond')):
            cm = np.ones(R.shape[1], bool) if cs is None else (cls == cs)
            slopes, xs, ys, gs = [], [], [], []
            for t in range(1, len(R)):
                m = cm & valid[t - 1] & ~np.isnan(P[t]) & ~np.isnan(R[t])
                if m.sum() < 3:
                    continue
                f, y = P[t, m], R[t, m]
                if f.std() == 0:
                    continue
                slopes.append(np.polyfit(f, y, 1)[0])
                xs.append(hm[t - 1, m]); ys.append(f); gs.append(np.full(m.sum(), t))
            sl = np.array(slopes)
            x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
            d = (x @ y) / (x @ x)
            u = y - d * x
            _, gi = np.unique(g, return_inverse=True)
            s = np.bincount(gi, x * u)
            td = d / np.sqrt((s @ s) / (x @ x) ** 2)
            rows.append({'预测': '波动率缩放' if sc else '不缩放', '资产类别': lab, 'Lewellen 斜率': sl.mean(), 't': sl.mean() / (sl.std(ddof=1) / np.sqrt(len(sl))),
                         '月数': len(sl), 'TSM 预测对 TSH 预测的斜率 d': d, 't(d)': td, 'R2%': (1 - u.var() / y.var()) * 100})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- Fig. 5：模拟
def simulate(R, OK):
    C12, H, Y, valid = signals(R, OK)
    Tn, N = R.shape
    present = ~np.isnan(R)
    mu = np.array([np.nanmean(R[:, j]) if present[:, j].sum() > 24 else np.nan for j in range(N)])
    # 用含固定效应的混合回归估计真实斜率（r_{t−12,t}/12）
    y, x, tt, ii = obs(R, np.ones_like(R), OK, 'cum', 12, False, False)
    _, g = np.unique(tt, return_inverse=True)
    _, b_real, _, u = ols_cl(y, x / 12, g, ii)
    sd = np.array([np.nanstd(R[:, j]) for j in range(N)])
    rows = []
    for beta in (0.0, 0.1, 0.2, 0.4):
        diffs, tst = [], []
        for _ in range(N_SIM):
            Rs = np.full_like(R, np.nan)
            for t in range(Tn):
                p = present[t] & ~np.isnan(mu)
                past = cumret(Rs[max(0, t - 12):t], 12)[-1] if t >= 12 else np.full(N, np.nan)
                drift = np.where(np.isnan(past), 0.0, beta * past / 12)
                e = RNG.standard_normal(N) * sd
                Rs[t] = np.where(p, mu * (1 - beta) + drift + e, np.nan)
            Cs, Hs, Ys, vs = signals(Rs, OK)
            a = np.nanmean(np.where(vs, np.where(Cs >= 0, 1, -1) * Ys, np.nan), axis=1)
            b = np.nanmean(np.where(vs, np.where(Hs >= 0, 1, -1) * Ys, np.nan), axis=1)
            d = a - b
            d = d[~np.isnan(d)]
            diffs.append(d.mean())
            tst.append(nw_t(d))
        diffs, tst = np.array(diffs), np.array(tst)
        rows.append({'真实斜率 β': beta, 'TSM − TSH 月均差% 平均': diffs.mean() * 100, '差值 t 中位数': np.median(tst),
                     'TSM 显著胜出比例（t > 1.96）': (tst > 1.96).mean(), 'TSM 胜出比例': (diffs > 0).mean()})
        print('sim', beta, rows[-1], flush=True)
    return pl.DataFrame(rows), b_real


def main():
    os.makedirs(OUT, exist_ok=True)
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    months, prods, cls, R, S, OK = monthly()
    okn = OK.sum()
    print(f'{len(prods)} 个品种，{months[0]} ~ {months[-1]}，可交易品种-月 {okn}')
    t2 = table2(months, prods, cls, R, OK); t2.write_parquet(os.path.join(OUT, 'table2.parquet'))
    print(t2.with_columns(pl.col(pl.Float64).round(2)))
    t3 = table3(R, OK); t3.write_parquet(os.path.join(OUT, 'table3.parquet')); print(t3)
    f3 = fig3(R, S, OK); f3.write_parquet(os.path.join(OUT, 'fig3.parquet'))
    pt = pooled_tables(months, cls, R, S, OK); pt.write_parquet(os.path.join(OUT, 'pooled.parquet'))
    print(pt.with_columns(pl.col(pl.Float64).round(2)))
    oos, preds = pooled_oos(months, prods, cls, R, S, OK); oos.write_parquet(os.path.join(OUT, 'pooled_oos.parquet'))
    print(oos.with_columns(pl.col(pl.Float64).round(2)))
    t9 = table9(months, prods, cls, R, OK); t9.write_parquet(os.path.join(OUT, 'table9.parquet'))
    print(t9.with_columns(pl.col(pl.Float64).round(3)))
    t10, t10d = table10(months, prods, cls, R, S, OK)
    t10.write_parquet(os.path.join(OUT, 'table10.parquet')); t10d.write_parquet(os.path.join(OUT, 'table10_diff.parquet'))
    print(t10.with_columns(pl.col(pl.Float64).round(3))); print(t10d.with_columns(pl.col(pl.Float64).round(3)))
    t11 = table11(cls, R, OK, preds); t11.write_parquet(os.path.join(OUT, 'table11.parquet'))
    print(t11.with_columns(pl.col(pl.Float64).round(3)))
    sim, b_real = simulate(R, OK); sim.write_parquet(os.path.join(OUT, 'sim.parquet'))
    pl.DataFrame({'β_real': [b_real]}).write_parquet(os.path.join(OUT, 'beta_real.parquet'))
    print('真实数据斜率（含固定效应，r_{t−12,t}/12）', round(b_real, 3))
    print(sim)


def main_cluster():
    months, prods, cls, R, S, OK = monthly()
    pc = pooled_tables(months, cls, R, S, OK, cluster=True)
    pc.write_parquet(os.path.join(OUT, 'pooled_cluster.parquet'))
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=250):
        print(pc.with_columns(pl.col(pl.Float64).round(2)))


if __name__ == '__main__':
    main_cluster() if sys.argv[1:] == ['cluster'] else main()
