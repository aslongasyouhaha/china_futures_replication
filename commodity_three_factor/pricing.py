"""商品三因子：描述统计（论文 Table 1、2）、主成分、Fama-MacBeth 截面定价与 HJ 距离（Table 3）、时间序列 α 检验（Table 6、7）。
用法: python pricing.py [paper|main]（默认两个都跑）；输出文件带版本后缀。"""
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
TEST = ['P1', 'P2', 'P3', 'P4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', '农产品', '金属']
MODELS = {'三因子': ['AVG', 'CARRY', 'CMOM'], '去掉动量': ['AVG', 'CARRY'], '去掉 carry': ['AVG', 'CMOM'], '只有 AVG': ['AVG']}
N_BOOT, BLOCK = 10000, 6


def nw_lag(T):
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def nw_cov(X, L=None):
    """X: T × k（已去均值前的序列），返回长期协方差（Bartlett 核）。"""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    T = len(X)
    L = nw_lag(T) if L is None else L
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / T
    for l in range(1, L + 1):
        G = Xc[l:].T @ Xc[:-l] / T
        S += (1 - l / (L + 1)) * (G + G.T)
    return S


def boot_ci(M, rng):
    """平稳自助法（平均块长 6 个月）给出均值的 95% 置信区间。M: T × k。"""
    T = len(M)
    idx = np.empty((N_BOOT, T), dtype=int)
    idx[:, 0] = rng.integers(0, T, N_BOOT)
    newb = rng.random((N_BOOT, T)) < 1 / BLOCK
    rnd = rng.integers(0, T, (N_BOOT, T))
    for t in range(1, T):
        idx[:, t] = np.where(newb[:, t], rnd[:, t], (idx[:, t - 1] + 1) % T)
    means = M[idx].mean(1)                                  # N_BOOT × k
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def describe(df, cols, rng):
    M = df.select(cols).to_numpy()
    lo, hi = boot_ci(M, rng)
    rows = []
    for j, c in enumerate(cols):
        x = M[:, j]
        t_nw = x.mean() / np.sqrt(nw_cov(x)[0, 0] / len(x))
        rows.append({'组合': c, '年化均值%': x.mean() * 1200, '95%下限': lo[j] * 1200, '95%上限': hi[j] * 1200,
                     '年化波动%': x.std(ddof=1) * np.sqrt(12) * 100, '夏普': x.mean() / x.std(ddof=1) * np.sqrt(12),
                     '偏度': st.skew(x), 't(NW)': t_nw, '正收益月%': (x > 0).mean() * 100})
    return pl.DataFrame(rows)


def fama_macbeth(R, F):
    T, N = R.shape
    K = F.shape[1]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0][1:].T          # N × K
    lam_t = np.linalg.lstsq(B, R.T, rcond=None)[0].T          # T × K（截面回归不含常数，同论文）
    lam = lam_t.mean(0)
    a_t = R - lam_t @ B.T
    alpha = a_t.mean(0)
    Sf = np.cov(F.T).reshape(K, K)
    c = lam @ np.linalg.solve(Sf, lam)
    V_nw = nw_cov(lam_t) / T
    V_sh = (1 + c) * V_nw + Sf / T
    A_nw = nw_cov(a_t) / T
    chi_nw = alpha @ np.linalg.pinv(A_nw) @ alpha
    chi_sh = alpha @ np.linalg.pinv((1 + c) * A_nw) @ alpha
    rbar = R.mean(0)
    V = np.cov(R.T)
    Vi = np.linalg.pinv(V)
    lam_g = np.linalg.solve(B.T @ Vi @ B, B.T @ Vi @ rbar)
    ag = rbar - B @ lam_g
    df = N - K
    return dict(lam=lam, p_nw=2 * st.norm.sf(np.abs(lam / np.sqrt(np.diag(V_nw)))),
                p_sh=2 * st.norm.sf(np.abs(lam / np.sqrt(np.diag(V_sh)))),
                r2_ols=1 - alpha @ alpha / (rbar @ rbar), r2_gls=1 - ag @ Vi @ ag / (rbar @ Vi @ rbar),
                chi_nw=chi_nw, pchi_nw=st.chi2.sf(chi_nw, df), chi_sh=chi_sh, pchi_sh=st.chi2.sf(chi_sh, df),
                alpha=alpha, rbar=rbar, pred=B @ lam, beta=B)


def hj_distance(R, F, n_sim=100000, seed=0):
    """Hansen-Jagannathan 距离（SDF m = 1 − b'(f − μ)，超额收益 E[mR] = 0），p 值按 Jagannathan & Wang (1996) 的加权 χ² 分布模拟。"""
    T, N = R.shape
    Fc = F - F.mean(0)
    G = R.T @ R / T                                     # 二阶矩矩阵
    Gi = np.linalg.inv(G)
    Cm = R.T @ Fc / T                                   # E[R (f−μ)']，N × K
    rbar = R.mean(0)
    b = np.linalg.solve(Cm.T @ Gi @ Cm, Cm.T @ Gi @ rbar)
    e = rbar - Cm @ b
    d2 = e @ Gi @ e
    u = R * (1 - Fc @ b)[:, None]                       # 矩条件 m_t R_t
    S = nw_cov(u)
    w, V = np.linalg.eigh(G)
    Gih = V @ np.diag(w ** -0.5) @ V.T                  # G^{-1/2}
    ws, Vs = np.linalg.eigh(S)
    Sh = Vs @ np.diag(np.sqrt(np.maximum(ws, 0))) @ Vs.T
    D = Gih @ Cm
    P = np.eye(N) - D @ np.linalg.solve(D.T @ D, D.T)
    A = Sh @ Gih @ P @ Gih @ Sh
    lam = np.sort(np.linalg.eigvalsh(A))[::-1][:N - F.shape[1]]
    rng = np.random.default_rng(seed)
    sim = (rng.chisquare(1, (n_sim, len(lam))) * lam).sum(1)
    return float(np.sqrt(d2)), float((sim >= T * d2).mean())


def ts_regression(R, F):
    T, N = R.shape
    K = F.shape[1]
    X = np.column_stack([np.ones(T), F])
    coef = np.linalg.lstsq(X, R, rcond=None)[0]
    E = R - X @ coef
    a = coef[0]
    Se = E.T @ E / (T - K - 1)
    mu = F.mean(0)
    Om = np.cov(F.T).reshape(K, K)
    grs = (T / N) * ((T - N - K) / (T - K - 1)) * (a @ np.linalg.solve(Se, a)) / (1 + mu @ np.linalg.solve(Om, mu))
    # 每个组合 α 的 NW t 值
    XtX = np.linalg.inv(X.T @ X)
    t_a, r2 = [], []
    for i in range(N):
        S = nw_cov(X * E[:, [i]], ) * T
        V = XtX @ S @ XtX
        t_a.append(a[i] / np.sqrt(V[0, 0]))
        r2.append(1 - E[:, i].var() / R[:, i].var())
    return dict(alpha=a, t_alpha=np.array(t_a), betas=coef[1:].T, r2=np.array(r2), grs=grs, p_grs=st.f.sf(grs, N, T - N - K))


def main(v):
    rng = np.random.default_rng(0)
    fac = pl.read_parquet(f'{OUT}/factors_{v}.parquet')
    port = pl.read_parquet(f'{OUT}/test_portfolios_{v}.parquet')
    d = fac.join(port, on='m').drop_nulls(TEST).sort('m')
    print(f'\n######## {v}：定价样本 {d["m"].min()} ~ {d["m"].max()}，T = {d.height} 个月 ########')
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(220)
    cols = ['AVG', 'CARRY', 'CMOM', 'C1', 'C2', 'C3', 'C4', 'M1', 'M2', 'M3', 'M4', 'CARRY_long', 'CARRY_short', 'CMOM_long',
            'CMOM_short'] + TEST
    desc = describe(d, cols, rng)
    desc.write_parquet(f'{OUT}/describe_{v}.parquet')
    print('== 因子与组合（年化，%）==')
    print(desc.with_columns(pl.exclude('组合').round(2)))
    corr = np.corrcoef(d.select('AVG', 'CARRY', 'CMOM').to_numpy().T)
    sign_diff = (np.sign(d['CARRY']) != np.sign(d['CMOM'])).mean()
    print(f"相关：AVG-CARRY {corr[0,1]:.2f}  AVG-CMOM {corr[0,2]:.2f}  CARRY-CMOM {corr[1,2]:.2f}（论文 0.27）；符号相反月份 {sign_diff:.0%}（论文 41%）")
    R = d.select(TEST).to_numpy()
    Rc = (R - R.mean(0)) / R.std(0)
    w, vec = np.linalg.eigh(np.cov(Rc.T))
    order = np.argsort(w)[::-1]
    pcs = Rc @ vec[:, order[:3]]
    share = w[order[:3]] / w.sum()
    pc = pl.DataFrame([{'因子': f, **{f'PC{k+1}': abs(np.corrcoef(d[f], pcs[:, k])[0, 1]) for k in range(3)},
                        **{f'PC{k+1}解释': share[k] for k in range(3)}} for f in ('AVG', 'CARRY', 'CMOM')])
    pc.write_parquet(f'{OUT}/pca_{v}.parquet')
    print(f"主成分：前三个解释 {share.sum():.0%}（{', '.join(f'{x:.0%}' for x in share)}）")
    print(pc.select('因子', 'PC1', 'PC2', 'PC3').with_columns(pl.exclude('因子').round(2)))
    fm_rows, pe_rows, ts_rows = [], [], []
    print('== Fama-MacBeth（截面无常数，λ 月度 %）+ HJ 距离 ==')
    for name, fs in MODELS.items():
        Fm = d.select(fs).to_numpy()
        r = fama_macbeth(R, Fm)
        hjd, hjp = hj_distance(R, Fm)
        row = {'模型': name}
        for j, f in enumerate(fs):
            row[f'λ_{f}'] = r['lam'][j] * 100
            row[f'p_{f}(NW)'] = r['p_nw'][j]
            row[f'p_{f}(Shanken)'] = r['p_sh'][j]
        row.update({'R2_GLS': r['r2_gls'], 'R2_OLS': r['r2_ols'], 'χ2_NW': r['chi_nw'], 'p_χ2_NW': r['pchi_nw'],
                    'χ2_Shanken': r['chi_sh'], 'p_χ2_Shanken': r['pchi_sh'], 'HJ': hjd, 'p_HJ': hjp, 'T': d.height})
        fm_rows.append(row)
        for i, tname in enumerate(TEST):
            pe_rows.append({'模型': name, '组合': tname, '实际%': r['rbar'][i] * 1200, '模型预测%': r['pred'][i] * 1200})
        t = ts_regression(R, Fm)
        for i, tname in enumerate(TEST):
            ts_rows.append({'模型': name, '组合': tname, 'α%': t['alpha'][i] * 1200, 't(α)': t['t_alpha'][i], 'R2': t['r2'][i],
                            **{f'β_{f}': t['betas'][i, j] for j, f in enumerate(fs)}, 'GRS': t['grs'], 'p_GRS': t['p_grs']})
        lam = '  '.join(f"{f} {r['lam'][j]*100:.2f}%（NW p={r['p_nw'][j]:.2f}，Sh p={r['p_sh'][j]:.2f}）" for j, f in enumerate(fs))
        print(f"{name:8s} {lam}\n         R²(GLS) {r['r2_gls']:.1%}  R²(OLS) {r['r2_ols']:.1%}  χ² NW p={r['pchi_nw']:.2f}  Shanken p={r['pchi_sh']:.2f}"
              f"  HJ {hjd:.3f}（p={hjp:.2f}）  GRS p={t['p_grs']:.2f}  显著 α {int((np.abs(t['t_alpha']) > 1.96).sum())}/{len(TEST)}")
    pl.DataFrame(fm_rows).write_parquet(f'{OUT}/fama_macbeth_{v}.parquet')
    pl.DataFrame(pe_rows).write_parquet(f'{OUT}/pricing_errors_{v}.parquet')
    pl.DataFrame(ts_rows).write_parquet(f'{OUT}/time_series_{v}.parquet')
    sub = []
    for lo, hi in ((2006, 2015), (2016, 2026)):
        q = d.filter(pl.col('m').dt.year().is_between(lo, hi))
        r = fama_macbeth(q.select(TEST).to_numpy(), q.select('AVG', 'CARRY', 'CMOM').to_numpy())
        sub.append({'时段': f'{lo}–{hi}', 'T': q.height, **{f'λ_{f}': r['lam'][j] * 100 for j, f in enumerate(('AVG', 'CARRY', 'CMOM'))},
                    **{f'p_{f}': r['p_nw'][j] for j, f in enumerate(('AVG', 'CARRY', 'CMOM'))}, 'R2_GLS': r['r2_gls'], 'p_χ2': r['pchi_nw']})
        print(f"分段 {lo}–{hi}（T={q.height}）λ = {', '.join(f'{x*100:.2f}%' for x in r['lam'])}  NW p = {', '.join(f'{x:.2f}' for x in r['p_nw'])}  χ² p={r['pchi_nw']:.2f}")
    pl.DataFrame(sub).write_parquet(f'{OUT}/subperiod_{v}.parquet')


if __name__ == '__main__':
    for v in (sys.argv[1:] or ['paper', 'main']):
        main(v)
