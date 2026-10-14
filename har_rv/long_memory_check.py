"""检验中国期货 RV 的高 d 是否是结构变化造成的“伪长记忆”。

① 原始: 全天 RV，全样本
② 只用日盘: 9:00–15:15 的 5 分钟收益算 RV，夜盘推出前后口径一致
③ 去台阶: 结构变化前后各自减去均值再估计
④ 只用变化之后的样本
⑤ 不同带宽 m = n^alpha（alpha 0.5–0.8）：伪长记忆通常在小带宽（只看最低频）下 d 更高
d 用 Local Whittle 估计，不设上限。
"""
import datetime as dt

import numpy as np
import polars as pl

import har_rv as H

# 结构变化日期：夜盘推出首日；沪深 300 为 2015-09-07 交易限制；10 年期国债无已知台阶（对照组）
BREAKS = {'IF': dt.date(2015, 9, 7), 'T': None, 'RB': dt.date(2014, 12, 29), 'CU': dt.date(2013, 12, 23),
          'M': dt.date(2014, 12, 29), 'SC': None, 'AU': dt.date(2013, 7, 8), 'TA': dt.date(2014, 12, 15)}
ALPHAS = (0.5, 0.6, 0.65, 0.7, 0.8)


def lw(x, alpha=0.65):
    n = len(x)
    m = int(n ** alpha)
    xc = x - x.mean()
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    I = np.abs(np.fft.fft(xc)[1:m + 1]) ** 2 / (2 * np.pi * n)
    g = np.arange(-0.3, 1.3, 0.005)
    R = [np.log(np.mean(lam ** (2 * d) * I)) - 2 * d * np.mean(np.log(lam)) for d in g]
    return float(g[int(np.argmin(R))])


def day_session_rv(ex, p):
    """只用日盘（9:00–15:15）的 5 分钟收益算 RV。"""
    m = pl.read_parquet(f'{H.DATA}/{ex}/{p}/minute_main.parquet', columns=['trading_day', 'datetime', 'symbol', 'close'])
    hr = pl.col('datetime').dt.hour()
    m = m.filter((hr >= 9) & (hr < 16))
    b = (m.with_columns(pl.col('datetime').dt.truncate('5m').alias('bucket'))
           .group_by(['trading_day', 'symbol', 'bucket']).agg(pl.col('close').last(), pl.col('datetime').max().alias('t'))
           .sort(['trading_day', 'symbol', 't']))
    b = b.with_columns((pl.col('close').log() - pl.col('close').log().shift(1).over(['trading_day', 'symbol'])).alias('r'),
                       (pl.col('t') - pl.col('t').shift(1).over(['trading_day', 'symbol'])).dt.total_minutes().alias('gap'))
    b = b.filter(pl.col('r').is_not_null() & (pl.col('gap') <= 10))
    rv = b.group_by('trading_day').agg((pl.col('r') ** 2).sum().sqrt().alias('rv'), pl.len().alias('n')).sort('trading_day')
    rv = rv.with_columns(pl.col('n').rolling_median(60, min_periods=10, center=True).alias('nm')) \
           .filter(pl.col('n') >= 0.5 * pl.col('nm')).drop('nm')
    return rv.with_columns(pl.col('rv') * np.sqrt(H.ANN) * 100)


def main():
    rows, bw = [], []
    for ex, p, name in H.PRODUCTS:
        full = pl.read_parquet(f'{H.OUT}/rv_{p}.parquet')
        day = day_session_rv(ex, p)
        brk = BREAKS[p]
        v_full, v_day = full['rv'].to_numpy(), day['rv'].to_numpy()
        r = {'品种': name, '结构变化': str(brk) if brk else '无', '①原始': lw(v_full), '②只用日盘': lw(v_day)}
        if brk:
            pre = pl.col('trading_day') < brk
            dm = full.with_columns((pl.col('rv') - pl.col('rv').mean().over(pre)).alias('x'))
            r['③去台阶'] = lw(dm['x'].to_numpy())
            post = full.filter(~pre)['rv'].to_numpy()
            r['④变化后样本'] = lw(post)
            r['变化前均值'] = round(float(full.filter(pre)['rv'].mean()), 1)
            r['变化后均值'] = round(float(full.filter(~pre)['rv'].mean()), 1)
        rows.append(r)
        bw.append(dict(品种=name, **{f'α={a}': round(lw(v_full, a), 2) for a in ALPHAS}))
        print(name, 'done', flush=True)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(220)
    print('\n== d 的估计（Local Whittle，带宽 n^0.65，不设上限）==')
    print(pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col(pl.Float64).round(3)))
    print('\n== 原始 RV 在不同带宽下的 d ==')
    print(pl.DataFrame(bw))


if __name__ == '__main__':
    main()
