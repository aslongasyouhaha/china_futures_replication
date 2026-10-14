"""《Volatility is Rough》其余检验：
1. 增量分布：Δ = 1, 5, 25 日对数波动率增量的峰度；标准差是否按 Δ^H 缩放
2. 协方差结构：Cov(log σ_t, log σ_{t+Δ}) 对 Δ^{2H} 线性（RFSV）还是对 log Δ 呈幂律（长记忆）
3. 预测：RFSV 预测对数方差 vs AR(5)、AR(10)、HAR（log 方差对当日、5 日、20 日均值回归），500 日滚动；
   指标 P = Σ(y − ŷ)² / Σ(y − ȳ)²，ȳ 为全样本均值（同论文），越小越好；Δ = 1, 5, 20
4. 伪长记忆：用估计的 H、ν 模拟 RFSV（分数布朗运动），对模拟的波动率水平做 Local Whittle 估计 d
预测与伪长记忆用全天 RV（与 har_rv 研究一致）；H 用全样本估计（同论文）。
"""
import os
from math import gamma, cos, pi

import numpy as np
import polars as pl

import rough_vol as RV

OUT = RV.OUT
WIN = 500
HORIZONS = (1, 5, 20)


def lw_d(x, alpha=0.65):
    n = len(x)
    m = int(n ** alpha)
    xc = x - x.mean()
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    I = np.abs(np.fft.fft(xc)[1:m + 1]) ** 2 / (2 * np.pi * n)
    g = np.arange(-0.3, 1.3, 0.005)
    R = [np.log(np.mean(lam ** (2 * d) * I)) - 2 * d * np.mean(np.log(lam)) for d in g]
    return float(g[int(np.argmin(R))])


def fbm(n, H, rng):
    """Davies-Harte 循环嵌入法模拟分数高斯噪声，累加得到分数布朗运动（步长 1）。"""
    k = np.arange(n + 1)
    gam = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([gam, gam[-2:0:-1]])
    lam = np.fft.fft(c).real
    lam[lam < 0] = 0
    m = len(c)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    fgn = np.fft.fft(np.sqrt(lam / m) * z).real[:n]
    return np.concatenate([[0], np.cumsum(fgn)])


def rfsv_weights(H, h, L=WIN):
    """离散化 E[log σ²_{t+Δ}|F_t] = (cos Hπ/π) Δ^{H+1/2} ∫ log σ²_{t−s} / ((s+Δ) s^{H+1/2}) ds：
    第 j 天（s ∈ [j, j+1)）的权重为该区间上核的积分（数值积分），截断到 L 天后归一化。"""
    w = np.empty(L)
    for j in range(L):
        s = np.linspace(j, j + 1, 201)[1:] if j == 0 else np.linspace(j, j + 1, 51)
        f = 1.0 / ((s + h) * s ** (H + 0.5))
        w[j] = np.trapz(f, s) if j > 0 else np.trapz(f, s) + (2 / (1 - 2 * H)) * s[0] ** (0.5 - H) / h
    return w / w.sum()


def forecast_compare(y, H):
    n = len(y)
    res = {}
    for h in HORIZONS:
        idx = np.arange(WIN + 20, n - h)
        tgt = y[idx + h]
        preds = {}
        w = rfsv_weights(H, h)
        preds['RFSV'] = np.array([w @ y[i - np.arange(WIN)] for i in idx])
        for name, lags in (('AR(5)', 5), ('AR(10)', 10)):
            p = np.empty(len(idx))
            for k, i in enumerate(idx):
                rows = np.arange(i - WIN + 1, i - h + 1)          # 训练样本的预测起点（目标已知）
                X = np.column_stack([np.ones(len(rows))] + [y[rows - l] for l in range(lags)])
                b = np.linalg.lstsq(X, y[rows + h], rcond=None)[0]
                p[k] = np.r_[1, [y[i - l] for l in range(lags)]] @ b
            preds[name] = p
        m5 = np.convolve(y, np.ones(5) / 5)[:n]
        m20 = np.convolve(y, np.ones(20) / 20)[:n]
        p = np.empty(len(idx))
        for k, i in enumerate(idx):
            rows = np.arange(i - WIN + 1, i - h + 1)
            X = np.column_stack([np.ones(len(rows)), y[rows], m5[rows], m20[rows]])
            b = np.linalg.lstsq(X, y[rows + h], rcond=None)[0]
            p[k] = np.r_[1, y[i], m5[i], m20[i]] @ b
        preds['HAR'] = p
        denom = ((tgt - y.mean()) ** 2).sum()
        for name, pr in preds.items():
            res[(h, name)] = ((tgt - pr) ** 2).sum() / denom
    return res


def main():
    rng = np.random.default_rng(42)
    hest = pl.read_parquet(f'{OUT}/h_estimates.parquet')
    dist_rows, cov_rows, fc_rows, lm_rows = [], [], [], []
    for ex, p, name in RV.PRODUCTS:
        for label, v in (('1 小时窗口', pl.read_parquet(f'{OUT}/vol1h_{p}.parquet')), ('全天 RV', RV.full_day_vol(p))):
            x = np.log(v['vol'].to_numpy())
            H = hest.filter((pl.col('品种') == name) & (pl.col('代理') == label))['H(=ζ2/2)'][0]
            # 1. 增量分布
            sd1 = np.std(x[1:] - x[:-1])
            r = {'品种': name, '代理': label, 'H': H}
            for D in (1, 5, 25):
                d = x[D:] - x[:-D]
                r[f'峰度 Δ={D}'] = round(float(((d - d.mean()) ** 4).mean() / d.var() ** 2), 2)
                if D > 1:
                    r[f'标准差比 Δ={D}（实际/按Δ^H）'] = round(float(np.std(d) / sd1 / D ** H), 3)
            dist_rows.append(r)
            # 2. 协方差：对 Δ^{2H} 线性 vs 对 log Δ 线性（幂律）
            lags = np.arange(1, 101)
            xc = x - x.mean()
            cov = np.array([np.mean(xc[l:] * xc[:-l]) for l in lags])
            def r2(X, yv):
                X = np.column_stack([np.ones(len(yv)), X])
                b = np.linalg.lstsq(X, yv, rcond=None)[0]
                e = yv - X @ b
                return 1 - e @ e / ((yv - yv.mean()) @ (yv - yv.mean()))
            cov_rows.append({'品种': name, '代理': label, 'R²: Cov 对 Δ^{2H}': round(r2(lags ** (2 * H), cov), 4),
                             'R²: log Cov 对 log Δ（幂律）': round(r2(np.log(lags), np.log(np.maximum(cov, 1e-12))), 4)})
            if label == '全天 RV':
                # 3. 预测（对数方差）
                y = 2 * x
                fc = forecast_compare(y, H)
                for (h, m), v_ in fc.items():
                    fc_rows.append({'品种': name, 'Δ': h, '模型': m, 'P': v_})
                # 4. 伪长记忆：模拟 RFSV，与实际估计的 d 对比
                nu = hest.filter((pl.col('品种') == name) & (pl.col('代理') == label))['ν'][0]
                sims = [lw_d(np.exp(nu * fbm(len(x) - 1, H, rng))) for _ in range(20)]
                lm_rows.append({'品种': name, 'H': H, '实际 RV 的 d': round(lw_d(np.exp(x)), 3),
                                '模拟 RFSV 的 d（20 次均值）': round(float(np.mean(sims)), 3),
                                '模拟 d 的 5%~95%': f'{np.percentile(sims, 5):.2f}~{np.percentile(sims, 95):.2f}'})
        print(name, 'done', flush=True)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(240)
    for nm, rows in (('dist', dist_rows), ('cov', cov_rows), ('forecast', fc_rows), ('longmem', lm_rows)):
        pl.DataFrame(rows).write_parquet(f'{OUT}/{nm}.parquet')
    print('\n== 增量分布 =='); print(pl.DataFrame(dist_rows))
    print('\n== 协方差结构 =='); print(pl.DataFrame(cov_rows))
    f = pl.DataFrame(fc_rows)
    print('\n== 预测 P 值（对数方差，越小越好）==')
    print(f.pivot(index=['品种', 'Δ'], columns='模型', values='P').with_columns(pl.exclude('品种', 'Δ').round(3)))
    rk = f.with_columns(pl.col('P').rank().over(['品种', 'Δ']).alias('rank'))
    print(rk.group_by(['Δ', '模型']).agg(pl.col('rank').mean().round(2), pl.col('P').mean().round(3)).sort(['Δ', 'rank']))
    print('\n== 伪长记忆 =='); print(pl.DataFrame(lm_rows))


if __name__ == '__main__':
    main()
