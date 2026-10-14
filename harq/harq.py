"""《Exploiting the Errors: A Simple Approach for Improved Volatility Forecasting》复现 —— Bollerslev, Patton & Quaedvlieg (2016, Journal of Econometrics)

模型（目标：未来 h 天平均 RV，直接回归；h = 1、5、22）：
  AR:     RV_d                         HAR:    RV_d, RV_w, RV_m
  HAR-J:  HAR + J_d（J = max(RV−BPV, 0)） CHAR:   BPV_d, BPV_w, BPV_m
  SHAR:   RV+_d, RV−_d, RV_w, RV_m      ARQ:    RV_d, RV_d·√RQ_d
  HARQ:   HAR + RV_d·√RQ_d               HARQ-F: HARQ + RV_w·√RQ_w + RV_m·√RQ_m
  HARQ-h: HAR + 与预测期对应的那一项（h=5：RV_w·√RQ_w；h=22：RV_m·√RQ_m）
  其中 RV_w、RV_m 为过去 5、22 天平均；√RQ_w、√RQ_m 为过去 5、22 天平均 RQ 的平方根
样本外：每天重估；滚动窗口（RW，1000 天）与扩展窗口（IW，从 1000 天起）；
  insanity filter：预测超出估计窗口内目标的取值范围时，以窗口内目标均值替代（同论文，对所有模型）
损失：MSE、QLIKE，报告相对 HAR 的比值；另按前一日 RQ 是否处于样本外最高 5% 分层；
修正的 Reality Check（HARQ 是否优于所有基准中最好者），平稳自助法 999 次、平均块长 5。
单位：RV 为日方差 × 10⁴（即 %²），RQ × 10⁸。
"""
import os

import numpy as np
import polars as pl

import measures as MS

HERE = MS.HERE
OUT = MS.OUT
WIN = 1000
HS = (1, 5, 22)
MODELS = ['AR', 'HAR', 'HAR-J', 'CHAR', 'SHAR', 'ARQ', 'HARQ', 'HARQ-F', 'HARQ-h']
BENCH = ['AR', 'HAR', 'HAR-J', 'CHAR', 'SHAR']


def load(p):
    d = pl.read_parquet(os.path.join(OUT, f'measures_{p}.parquet')).sort('trading_day')
    d = d.with_columns(pl.col('RV', 'BPV', 'RVp', 'RVn') * 1e4, pl.col('RQ') * 1e8)
    d = d.with_columns(
        pl.col('RV').rolling_mean(5).alias('RV_w'), pl.col('RV').rolling_mean(22).alias('RV_m'),
        pl.col('BPV').rolling_mean(5).alias('BPV_w'), pl.col('BPV').rolling_mean(22).alias('BPV_m'),
        (pl.col('RV') - pl.col('BPV')).clip(lower_bound=0).alias('J'),
        pl.col('RQ').sqrt().alias('q_d'), pl.col('RQ').rolling_mean(5).sqrt().alias('q_w'), pl.col('RQ').rolling_mean(22).sqrt().alias('q_m'))
    for h in HS:
        d = d.with_columns(pl.col('RV').shift(-1).rolling_mean(h).shift(-(h - 1)).alias(f'y{h}'))
    return d


def design(d, model, h):
    c = {k: d[k].to_numpy() for k in ('RV', 'RV_w', 'RV_m', 'BPV', 'BPV_w', 'BPV_m', 'J', 'RVp', 'RVn', 'q_d', 'q_w', 'q_m')}
    one = np.ones(d.height)
    har = [one, c['RV'], c['RV_w'], c['RV_m']]
    if model == 'AR':
        cols = [one, c['RV']]
    elif model == 'HAR':
        cols = har
    elif model == 'HAR-J':
        cols = har + [c['J']]
    elif model == 'CHAR':
        cols = [one, c['BPV'], c['BPV_w'], c['BPV_m']]
    elif model == 'SHAR':
        cols = [one, c['RVp'], c['RVn'], c['RV_w'], c['RV_m']]
    elif model == 'ARQ':
        cols = [one, c['RV'], c['RV'] * c['q_d']]
    elif model == 'HARQ':
        cols = har + [c['RV'] * c['q_d']]
    elif model == 'HARQ-F':
        cols = har + [c['RV'] * c['q_d'], c['RV_w'] * c['q_w'], c['RV_m'] * c['q_m']]
    elif model == 'HARQ-h':
        extra = {1: c['RV'] * c['q_d'], 5: c['RV_w'] * c['q_w'], 22: c['RV_m'] * c['q_m']}[h]
        cols = har + [extra]
    return np.column_stack(cols)


def rolling_forecasts(X, y, start, h, window):
    """每个预测起点 t（t ≥ start）只用目标已知的样本 s ≤ t−h；RW 取最近 1000 个，IW 取全部（≥ 第 22 行）。"""
    n, k = X.shape
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xz = np.where(ok[:, None], X, 0.0)
    yz = np.where(ok, y, 0.0)
    Sxx = np.concatenate([np.zeros((1, k, k)), np.cumsum(Xz[:, :, None] * Xz[:, None, :], axis=0)])
    Sxy = np.concatenate([np.zeros((1, k)), np.cumsum(Xz * yz[:, None], axis=0)])
    ts = np.arange(start, n)
    hi = ts - h + 1                                            # 训练样本 s < hi
    lo = np.maximum(hi - window, 0) if window else np.zeros_like(hi)
    A = Sxx[hi] - Sxx[lo]
    bvec = Sxy[hi] - Sxy[lo]
    A = A + np.eye(k)[None] * 1e-10
    coef = np.linalg.solve(A, bvec[:, :, None])[:, :, 0]
    f = np.einsum('ij,ij->i', X[ts], coef)
    # insanity filter：目标在训练窗口内的取值范围
    yy = np.where(ok, y, np.nan)
    ymin = np.array([np.nanmin(yy[l:u]) for l, u in zip(lo, hi)])
    ymax = np.array([np.nanmax(yy[l:u]) for l, u in zip(lo, hi)])
    ymean = (Sxy[hi, 0] - Sxy[lo, 0]) / np.maximum((Sxx[hi, 0, 0] - Sxx[lo, 0, 0]), 1)
    bad = ~np.isfinite(f) | (f < ymin) | (f > ymax)
    f = np.where(bad, ymean, f)
    return ts, f, bad.mean()


def stationary_bootstrap_idx(n, B, block, rng):
    idx = np.empty((B, n), dtype=int)
    idx[:, 0] = rng.integers(0, n, B)
    new = rng.random((B, n)) < 1 / block
    rnd = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], rnd[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def reality_check(loss, rng, B=999, block=5):
    """H0：min_k E[L_k − L_HARQ] ≤ 0；拒绝 = HARQ 显著优于所有基准中最好者。"""
    D = np.column_stack([loss[m] - loss['HARQ'] for m in BENCH])
    n = len(D)
    dbar = D.mean(0)
    stat = np.sqrt(n) * dbar.min()
    idx = stationary_bootstrap_idx(n, B, block, rng)
    boot = np.sqrt(n) * (D[idx].mean(1) - dbar).min(1)
    return float((boot >= stat).mean())


def run_product(p, rng):
    d = load(p)
    rows, strat, rc = [], [], []
    for h in HS:
        y = d[f'y{h}'].to_numpy()
        start = WIN + 22
        if len(y) - h <= start + 100:
            continue
        for wname, window in (('RW', WIN), ('IW', None)):
            preds = {}
            filt = {}
            for m in MODELS:
                if m == 'HARQ-h' and h == 1:
                    continue
                ts, f, fb = rolling_forecasts(design(d, m, h), y, start, h, window)
                preds[m] = f
                filt[m] = fb
            valid = np.isfinite(y[ts])
            yt = y[ts][valid]
            q_prev = d['RQ'].to_numpy()[ts][valid]
            top = q_prev >= np.quantile(q_prev, 0.95)
            loss = {}
            for m, f in preds.items():
                f = f[valid]
                loss[m] = {'MSE': (yt - f) ** 2, 'QLIKE': yt / f - np.log(yt / f) - 1}
            for lf in ('MSE', 'QLIKE'):
                base = loss['HAR'][lf].mean()
                for m in preds:
                    rows.append({'品种': p, 'h': h, '窗口': wname, '损失': lf, '模型': m, '比值': loss[m][lf].mean() / base, '过滤比例': filt[m]})
                    for lab, sel in (('最高 5% RQ', top), ('其余 95%', ~top)):
                        strat.append({'品种': p, 'h': h, '窗口': wname, '损失': lf, '模型': m, '分层': lab,
                                      '比值': loss[m][lf][sel].mean() / loss['HAR'][lf][sel].mean()})
                if h == 1:
                    rc.append({'品种': p, '窗口': wname, '损失': lf, 'RC p 值': reality_check({m: loss[m][lf] for m in BENCH + ['HARQ']}, rng)})
    return rows, strat, rc


def nw_se(X, e, L):
    n = len(e)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtX @ S @ XtX))


def insample(p, h=1):
    """全样本估计（论文 Table 3 / 6）；交互项中的 √RQ 去均值，使 β1 为平均测量误差下的系数。"""
    d = load(p).drop_nulls(['RV_m', 'q_m', f'y{h}'])
    y = d[f'y{h}'].to_numpy()
    rv, rw, rm = d['RV'].to_numpy(), d['RV_w'].to_numpy(), d['RV_m'].to_numpy()
    qd, qw, qm = (d[c].to_numpy() - d[c].mean() for c in ('q_d', 'q_w', 'q_m'))
    one = np.ones(len(y))
    specs = {'AR': ([one, rv], ['β0', 'β1']), 'HAR': ([one, rv, rw, rm], ['β0', 'β1', 'β2', 'β3']),
             'ARQ': ([one, rv, rv * qd], ['β0', 'β1', 'β1Q']), 'HARQ': ([one, rv, rw, rm, rv * qd], ['β0', 'β1', 'β2', 'β3', 'β1Q']),
             'HARQ-F': ([one, rv, rw, rm, rv * qd, rw * qw, rm * qm], ['β0', 'β1', 'β2', 'β3', 'β1Q', 'β2Q', 'β3Q'])}
    out = []
    for m, (cols, names) in specs.items():
        X = np.column_stack(cols)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        se = nw_se(X, e, {1: 5, 5: 10, 22: 44}[h])
        f = X @ b
        ok = f > 0
        row = {'品种': p, 'h': h, '模型': m, 'R2': 1 - e.var() / y.var(), 'MSE': float((e ** 2).mean()),
               'QLIKE': float((y[ok] / f[ok] - np.log(y[ok] / f[ok]) - 1).mean())}
        for nm, v, s in zip(names, b, se):
            row[nm] = float(v)
            row[f'se_{nm}'] = float(s)
        if 'β2' in row:
            b1, b2, b3 = row['β1'], row['β2'], row['β3']
            w = np.array([b1 + b2 / 5 + b3 / 22] + [b2 / 5 + b3 / 22] * 4 + [b3 / 22] * 17)
            row['持续性'] = b1 + b2 + b3
            row['平均滞后'] = float((np.arange(1, 23) * w).sum() / w.sum())
        else:
            row['持续性'] = row['β1']
        out.append(row)
    ar1 = float(np.corrcoef(rv[1:], rv[:-1])[0, 1])
    return out, {'品种': p, '天数': len(y), 'RV 均值': float(rv.mean()), 'RV 中位数': float(np.median(rv)), 'RV 最大': float(rv.max()),
                 'AR(1)': ar1, 'ARQ β1': [r for r in out if r['模型'] == 'ARQ'][0]['β1'],
                 'RQ^0.5 标准差（相对均值）': float(d['q_d'].std() / d['q_d'].mean())}


def main():
    rng = np.random.default_rng(0)
    rows, strat, rc, ins, summ = [], [], [], [], []
    for _, p in MS.PRODUCTS:
        r, s, c = run_product(p, rng)
        rows += r
        strat += s
        rc += c
        for h in HS:
            o, sm = insample(p, h)
            ins += o
            if h == 1:
                summ.append(sm)
        print(p, 'done', flush=True)
    pl.DataFrame(rows).write_parquet(os.path.join(OUT, 'oos.parquet'))
    pl.DataFrame(strat).write_parquet(os.path.join(OUT, 'oos_strat.parquet'))
    pl.DataFrame(rc).write_parquet(os.path.join(OUT, 'reality_check.parquet'))
    pl.DataFrame(ins).write_parquet(os.path.join(OUT, 'insample.parquet'))
    pl.DataFrame(summ).write_parquet(os.path.join(OUT, 'summary.parquet'))
    o = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(220)
    for h in HS:
        print(f'\n== h = {h}：沪深 300（IF）损失比值（相对 HAR）==')
        q = o.filter((pl.col('品种') == 'IF') & (pl.col('h') == h)).pivot(index=['窗口', '损失'], columns='模型', values='比值')
        print(q.with_columns(pl.exclude('窗口', '损失').round(4)))
        print(f'== h = {h}：商品（22 个）损失比值的平均 / 中位数 ==')
        c = o.filter((pl.col('品种') != 'IF') & (pl.col('h') == h)).group_by(['窗口', '损失', '模型']).agg(
            pl.col('比值').mean().alias('avg'), pl.col('比值').median().alias('med'))
        print(c.pivot(index=['窗口', '损失'], columns='模型', values='avg').with_columns(pl.exclude('窗口', '损失').round(4)))
    print(pl.DataFrame(rc).group_by(['窗口', '损失']).agg((pl.col('RC p 值') < 0.05).mean().alias('5% 拒绝比例'),
                                                       (pl.col('RC p 值') < 0.10).mean().alias('10% 拒绝比例')))
    print(pl.DataFrame(rc).filter(pl.col('品种') == 'IF'))


if __name__ == '__main__':
    main()
