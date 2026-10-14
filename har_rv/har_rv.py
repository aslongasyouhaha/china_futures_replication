"""HAR-RV 复现 —— Corsi (2009, Journal of Financial Econometrics)

日已实现波动率 RV_d = sqrt(Σ r²)，r 为 5 分钟对数收益（只取连续交易时段内，跨午休/小节休息/夜盘→日盘的跳空不计），
年化（× sqrt(243)，单位 %），不取对数（同论文）。
HAR: RV_{t+1} = c + βd·RV_t + βw·RV_t^(w) + βm·RV_t^(m)，周 = 过去 5 日均值，月 = 过去 22 日均值。
样本内: OLS + Newey-West(5)；样本外: 1000 日滚动窗口逐日重估，预测 h = 1/5/10 日的平均 RV，
对比 AR(1)、AR(3)、AR(22)（ARFIMA 的替代）和 EWMA（TSMOM 现用的波动率），指标 RMSE、MAE、MZ R²。
"""
import numpy as np
import polars as pl
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
OUT = f'{RESEARCH}/har_rv/output'
ANN = 243
PRODUCTS = [('CFFEX', 'IF', '沪深300股指'), ('CFFEX', 'T', '10年期国债'), ('SHFE', 'RB', '螺纹钢'), ('SHFE', 'CU', '铜'),
            ('DCE', 'M', '豆粕'), ('INE', 'SC', '原油'), ('SHFE', 'AU', '黄金'), ('CZCE', 'TA', 'PTA')]
WINDOW = 1000
HORIZONS = (1, 5, 10, 22)   # 22 日 ≈ 一个月，对应 TSMOM 的月度调仓


def daily_rv(ex, p, freq='5m', max_gap_min=10):
    m = pl.read_parquet(f'{DATA}/{ex}/{p}/minute_main.parquet', columns=['trading_day', 'datetime', 'symbol', 'close'])
    # 5 分钟重采样：每个 5 分钟区间的最后一个收盘价
    b = (m.with_columns(pl.col('datetime').dt.truncate(freq).alias('bucket'))
           .group_by(['trading_day', 'symbol', 'bucket']).agg(pl.col('close').last(), pl.col('datetime').max().alias('t'))
           .sort(['trading_day', 'symbol', 't']))
    b = b.with_columns((pl.col('close').log() - pl.col('close').log().shift(1).over(['trading_day', 'symbol'])).alias('r'),
                       ((pl.col('t') - pl.col('t').shift(1).over(['trading_day', 'symbol'])).dt.total_minutes()).alias('gap'))
    # 只保留连续时段内的收益（相邻 5 分钟 K 线间隔不超过 10 分钟）
    b = b.filter(pl.col('r').is_not_null() & (pl.col('gap') <= max_gap_min))
    rv = b.group_by('trading_day').agg((pl.col('r') ** 2).sum().sqrt().alias('rv'), pl.len().alias('n')).sort('trading_day')
    # 剔除日内收益太少的日子（如节前半天、数据缺失）；参照用滚动 60 日中位数，
    # 因为夜盘推出前后每日 K 线数相差约一倍，不能用全样本中位数
    rv = rv.with_columns(pl.col('n').rolling_median(60, min_periods=10, center=True).alias('n_med'))
    rv = rv.filter(pl.col('n') >= 0.5 * pl.col('n_med')).drop('n_med')
    return rv.with_columns((pl.col('rv') * np.sqrt(ANN) * 100).alias('rv'))   # 年化 %


def features(rv):
    x = rv.with_columns(pl.col('rv').rolling_mean(5).alias('rv_w'), pl.col('rv').rolling_mean(22).alias('rv_m'))
    for h in HORIZONS:   # 目标: 未来 h 日的平均 RV
        x = x.with_columns(pl.col('rv').shift(-1).rolling_mean(h).shift(-(h - 1)).alias(f'y{h}'))
    return x


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def nw_se(X, e, lags=5):
    n = len(e)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))


def mz_r2(y, f):
    X = np.column_stack([np.ones(len(f)), f])
    b = ols(X, y)
    e = y - X @ b
    return 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))


def design(rvs, i, model):
    """第 i 天（含）为止的信息构造回归变量。rvs = 日 RV 数组。"""
    if model == 'HAR':
        return [1.0, rvs[i], rvs[i - 4:i + 1].mean(), rvs[i - 21:i + 1].mean()]
    lags = {'AR1': 1, 'AR3': 3, 'AR22': 22}[model]
    return [1.0] + list(rvs[i - lags + 1:i + 1][::-1])


# ---------------- ARFIMA(5, d, 0) ----------------
AR_P = 5
D_GRID = np.round(np.arange(0.0, 0.50, 0.01), 2)


def frac_weights(d, K):
    """(1-L)^d 的展开系数 w_0..w_{K-1}。"""
    w = np.empty(K)
    w[0] = 1.0
    for k in range(1, K):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def frac_diff(x, d):
    """u_t = Σ_k w_k x_{t-k}（截断到序列起点）。"""
    return np.convolve(x, frac_weights(d, len(x)))[:len(x)]


def ar_fit(u, p=AR_P, burn=50):
    Y = u[burn + p:]
    X = np.column_stack([u[burn + p - j:len(u) - j] for j in range(1, p + 1)])
    b = ols(X, Y)
    e = Y - X @ b
    return b, e @ e


def arfima_d(x, alpha=0.65):
    """Local Whittle 估计 d（Robinson 1995）：只用最低的 m = n^alpha 个频率，
    不受短期自回归结构干扰；d 截断在 [0, 0.49]。"""
    n = len(x)
    m = int(n ** alpha)
    xc = x - x.mean()
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    I = np.abs(np.fft.fft(xc)[1:m + 1]) ** 2 / (2 * np.pi * n)
    def R(d):
        return np.log(np.mean(lam ** (2 * d) * I)) - 2 * d * np.mean(np.log(lam))
    grid = np.arange(-0.2, 0.9, 0.005)
    d = float(grid[int(np.argmin([R(g) for g in grid]))])
    return float(min(max(d, 0.0), 0.49))


def arfima_forecast(x, d, hmax):
    """用到 x 末尾为止的数据，递推预测未来 1..hmax 日的 x。"""
    mu = x.mean()
    xc = x - mu
    u = frac_diff(xc, d)
    phi, _ = ar_fit(u)
    K = len(xc) + hmax
    w = frac_weights(d, K)
    xs = list(xc)
    us = list(u)
    out = []
    for j in range(hmax):
        u_hat = sum(phi[k] * us[-1 - k] for k in range(AR_P))
        # x_t = u_t - Σ_{k>=1} w_k x_{t-k}
        hist = np.array(xs[::-1])                 # x_{t-1}, x_{t-2}, ...
        x_hat = u_hat - (w[1:len(hist) + 1] @ hist)
        us.append(u_hat)
        xs.append(x_hat)
        out.append(x_hat + mu)
    return np.array(out)


def arfima_oos(v, hmax, refit_every=21):
    """滚动 1000 日窗口：d 每 refit_every 天重估，AR 系数每天重估。返回 {h: 预测数组}。"""
    n = len(v)
    start = WINDOW + 22
    preds = {h: np.full(n, np.nan) for h in HORIZONS}
    d = None
    ds = []
    for i in range(start, n - 1):
        x = v[i - WINDOW + 1:i + 1]
        if d is None or (i - start) % refit_every == 0:
            d = arfima_d(x)
            ds.append(d)
        f = arfima_forecast(x, d, hmax)
        for h in HORIZONS:
            preds[h][i] = f[:h].mean()
    return preds, ds


def oos(rv, h, extra=None):
    """1000 日滚动窗口，逐日重估，预测未来 h 日平均 RV。"""
    v = rv['rv'].to_numpy()
    n = len(v)
    tgt = np.full(n, np.nan)
    for i in range(n - h):
        tgt[i] = v[i + 1:i + 1 + h].mean()
    # EWMA（同 TSMOM：中心质量 60 天，对日收益平方做指数加权；这里对 RV² 做同样的加权以便可比）
    def ewma(com):
        lam = com / (com + 1)
        out = np.zeros(n)
        s = v[0] ** 2
        for i in range(n):
            s = lam * s + (1 - lam) * v[i] ** 2
            out[i] = np.sqrt(s)
        return out
    models = ['HAR', 'AR1', 'AR3', 'AR22']
    preds = {m: np.full(n, np.nan) for m in models}
    start = WINDOW + 22
    rows = {m: np.array([design(v, j, m) for j in range(21, n)]) for m in models}   # 行 j-21 对应第 j 天
    for i in range(start, n - h):
        lo = i - WINDOW
        for m in models:
            # 训练样本: 第 lo ~ i-h 天（这些天的目标值在第 i 天已知），rows 的下标 = 天数 - 21
            Xw = rows[m][lo - 21:i - h - 21 + 1]
            yw = tgt[lo:i - h + 1]
            ok = np.isfinite(yw)
            b = ols(Xw[ok], yw[ok])
            preds[m][i] = rows[m][i - 21] @ b
    if extra is not None:
        preds['ARFIMA'] = extra[h]
    preds['EWMA'] = ewma(60)       # TSMOM 现用：中心质量 60 天
    preds['EWMA20'] = ewma(20)     # 对照：中心质量 20 天，反应更快
    idx = np.arange(start, n - h)
    y = tgt[idx]
    res = {}
    for m, f in preds.items():
        fi = f[idx]
        ok = np.isfinite(y) & np.isfinite(fi)
        e = y[ok] - fi[ok]
        res[m] = dict(RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(), MZ_R2=mz_r2(y[ok], fi[ok]))
    return res, rv['trading_day'][int(idx[0])], rv['trading_day'][int(idx[-1])]


def main():
    pl.Config.set_tbl_rows(50)
    insample, oos_rows = [], []
    for ex, p, name in PRODUCTS:
        rv = daily_rv(ex, p)
        rv.write_parquet(f'{OUT}/rv_{p}.parquet')
        x = features(rv).drop_nulls(['rv_w', 'rv_m', 'y1'])
        X = np.column_stack([np.ones(x.height), x['rv'], x['rv_w'], x['rv_m']])
        y = x['y1'].to_numpy()
        b = ols(X, y)
        e = y - X @ b
        se = nw_se(X, e, 5)
        r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
        insample.append(dict(品种=name, 样本=f"{x['trading_day'].min()}~{x['trading_day'].max()}", 天数=x.height,
                             RV均值=round(float(rv['rv'].mean()), 1),
                             βd=f'{b[1]:.3f} ({b[1]/se[1]:.1f})', βw=f'{b[2]:.3f} ({b[2]/se[2]:.1f})',
                             βm=f'{b[3]:.3f} ({b[3]/se[3]:.1f})', 系数和=round(b[1] + b[2] + b[3], 3), R2=round(r2, 3)))
        afp, ds = arfima_oos(rv['rv'].to_numpy(), max(HORIZONS))
        d_full = arfima_d(rv['rv'].to_numpy())
        insample[-1]['d全样本'] = d_full
        insample[-1]['d滚动均值'] = round(float(np.mean(ds)), 3)
        for h in HORIZONS:
            res, d0, d1 = oos(rv, h, afp)
            for m, s in res.items():
                oos_rows.append(dict(品种=name, h=h, 模型=m, RMSE=s['RMSE'], MAE=s['MAE'], MZ_R2=s['MZ_R2']))
        print(f'{name} done', flush=True)
    print('\n== 样本内 HAR 系数（括号内为 Newey-West t 值）==')
    print(pl.DataFrame(insample))
    o = pl.DataFrame(oos_rows)
    o.write_parquet(f'{OUT}/har_oos.parquet')
    print('\n== 样本外 Mincer-Zarnowitz R²（越高越好）==')
    print(o.pivot(index=['品种', 'h'], columns='模型', values='MZ_R2').with_columns(pl.exclude('品种', 'h').round(3)))
    print('\n== 样本外 RMSE 相对 HAR（>1 表示比 HAR 差）==')
    har = o.filter(pl.col('模型') == 'HAR').select('品种', 'h', pl.col('RMSE').alias('har'))
    rel = o.join(har, on=['品种', 'h']).with_columns((pl.col('RMSE') / pl.col('har')).round(3).alias('rel'))
    print(rel.pivot(index=['品种', 'h'], columns='模型', values='rel'))
    print('\n== 各模型平均名次（按 RMSE，1 = 最好）==')
    rk = o.with_columns(pl.col('RMSE').rank().over(['品种', 'h']).alias('rank'))
    print(rk.group_by(['h', '模型']).agg(pl.col('rank').mean().round(2)).pivot(index='模型', columns='h', values='rank').sort('1'))


if __name__ == '__main__':
    main()
