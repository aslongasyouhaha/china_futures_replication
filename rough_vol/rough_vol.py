"""《Volatility is Rough》复现 —— Gatheral, Jaisson & Rosenbaum (2018, Quantitative Finance)

波动率代理（单位无关，取对数后差分会抵消常数）：
  ① 1 小时窗口: 每天 10:30–11:29 的 1 分钟对数收益平方和（论文用每天固定 1 小时窗口，减小积分平滑带来的 H 偏高）
  ② 全天 RV: research/har_rv/output/rv_*.parquet（5 分钟收益、连续时段、含夜盘）
H 的估计: m(q,Δ) = mean |log σ_{t+Δ} − log σ_t|^q（重叠样本），q ∈ {0.5,1,1.5,2,3}，Δ = 1..30 交易日；
  log m(q,Δ) 对 log Δ 做 OLS，斜率 ζ_q；单分形时 ζ_q = qH；H 取 ζ_2/2，ν² = exp(截距)（q=2）。
"""
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
HAR_OUT = os.path.join(HERE, '..', 'har_rv', 'output')
PRODUCTS = [('CFFEX', 'IF', '沪深300股指'), ('CFFEX', 'T', '10年期国债'), ('SHFE', 'RB', '螺纹钢'), ('SHFE', 'CU', '铜'),
            ('DCE', 'M', '豆粕'), ('INE', 'SC', '原油'), ('SHFE', 'AU', '黄金'), ('CZCE', 'TA', 'PTA')]
QS = (0.5, 1.0, 1.5, 2.0, 3.0)
LAGS = np.arange(1, 31)


def hour_window_vol(ex, p, h0=(10, 30), h1=(11, 30)):
    """每天 10:30–11:29 的 1 分钟已实现波动率（同一合约内相邻分钟收益）。"""
    m = pl.read_parquet(f'{DATA}/{ex}/{p}/minute_main.parquet', columns=['trading_day', 'datetime', 'symbol', 'close'])
    mins = pl.col('datetime').dt.hour().cast(pl.Int32) * 60 + pl.col('datetime').dt.minute().cast(pl.Int32)   # dt.hour 是 int8，先转型防溢出
    m = m.filter((mins >= h0[0] * 60 + h0[1]) & (mins < h1[0] * 60 + h1[1])).sort(['trading_day', 'datetime'])
    m = m.with_columns((pl.col('close').log() - pl.col('close').log().shift(1).over(['trading_day', 'symbol'])).alias('r'),
                       (pl.col('datetime') - pl.col('datetime').shift(1).over(['trading_day', 'symbol'])).dt.total_minutes().alias('gap'))
    m = m.filter(pl.col('r').is_not_null() & (pl.col('gap') <= 2))
    v = m.group_by('trading_day').agg((pl.col('r') ** 2).sum().sqrt().alias('vol'), pl.len().alias('n')).sort('trading_day')
    # 至少 40 个分钟收益、且波动不为 0（剔除半天、停牌、价格不动的日子）
    return v.filter((pl.col('n') >= 40) & (pl.col('vol') > 0))


def full_day_vol(p):
    v = pl.read_parquet(f'{HAR_OUT}/rv_{p}.parquet').rename({'rv': 'vol'})
    return v.filter(pl.col('vol') > 0)


def m_q(x, q, lag):
    d = np.abs(x[lag:] - x[:-lag])
    return np.mean(d ** q)


def estimate_h(logvol):
    """返回 {q: (ζ_q, 截距)}，以及 H = ζ_2/2、ν。"""
    out = {}
    for q in QS:
        y = np.log([m_q(logvol, q, l) for l in LAGS])
        X = np.column_stack([np.ones(len(LAGS)), np.log(LAGS)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        out[q] = (b[1], b[0])
    H = out[2.0][0] / 2
    nu = np.sqrt(np.exp(out[2.0][1]))
    return out, H, nu


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ex, p, name in PRODUCTS:
        hv = hour_window_vol(ex, p)
        hv.write_parquet(f'{OUT}/vol1h_{p}.parquet')
        fv = full_day_vol(p)
        for label, v in (('1 小时窗口', hv), ('全天 RV', fv)):
            x = np.log(v['vol'].to_numpy())
            z, H, nu = estimate_h(x)
            half = len(x) // 2
            H1 = estimate_h(x[:half])[1]
            H2 = estimate_h(x[half:])[1]
            rows.append({'品种': name, '代理': label, '天数': len(x), 'H(=ζ2/2)': round(H, 3), 'ν': round(nu, 3),
                         **{f'ζ{q}/{q}': round(z[q][0] / q, 3) for q in QS},
                         '前半H': round(H1, 3), '后半H': round(H2, 3)})
        print(name, 'done', flush=True)
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/h_estimates.parquet')
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(14)
    pl.Config.set_tbl_width_chars(240)
    print(t)


if __name__ == '__main__':
    main()
