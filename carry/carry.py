"""Carry 复现（论文口径）—— Koijen, Moskowitz, Pedersen & Vrugt (2018, JFE)

商品 carry：持有期货、假设现货价格不变时的收益 ≈ 期限结构斜率。
  每个月末，剔除交割月和交割前一个月的合约；持仓量最大者为近月，比它更远、持仓量最大者为远月（远月持仓 >= 500 手）。
  carry = ln(F_近 / F_远) × 12 / 相差月数（年化；正值 = 贴水 = 正 carry）。价格用官方结算价，缺失时用收盘价。
截面策略：月末按 carry 排名，w_i = z · (rank_i − (N+1)/2)，多头合计 +1、空头合计 −1，持有 1 个月。
  信号: carry1m（当月）与 carry1-12（过去 12 个月平均，去季节性）。
时序策略: carry > 0 做多、< 0 做空，权重 ±2/N（总敞口 2）；另一版与该品种截至当时的历史均值比较。
收益: 主力连续后复权日收益。不扣成本，月末收盘成交。商品品种（不含中金所）。
"""
import glob
import os
import sys

import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'time_series_momentum'))
import tsmom as T  # noqa: E402  复用日收益、统计函数

ANN = 243
MIN_FAR_OI = 500
MIN_VOL_LOTS = 1000          # 品种池：主力 60 日平均成交 >= 1000 手（同 TSMOM 论文口径）
FINANCIAL = {'IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS', 'TL'}


def month_index(symbol):
    yymm = symbol[-5:-1] if symbol.endswith('F') else symbol[-4:]
    return 12 * (2000 + int(yymm[:2])) + int(yymm[2:]) - 1


def carry_panel():
    """每个品种每个月末的 carry。"""
    rows = []
    for f in glob.glob(f'{DATA}/*/*/daily_contracts.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p.endswith('F') and len(p) > 1 and p[:-1] in ('L', 'PP', 'V'):
            continue
        d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'open_interest'])
        d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'),
                           pl.col('trading_day').dt.truncate('1mo').alias('month'))
        me = d.group_by('month').agg(pl.col('trading_day').max().alias('me'))
        d = d.join(me, on='month').filter(pl.col('trading_day') == pl.col('me'))
        d = d.with_columns(pl.col('symbol').map_elements(month_index, return_dtype=pl.Int64).alias('cm'),
                           (pl.col('trading_day').dt.year() * 12 + pl.col('trading_day').dt.month() - 1).alias('now'))
        # 剔除交割月及交割前一个月
        d = d.filter((pl.col('cm') >= pl.col('now') + 2) & (pl.col('px') > 0) & (pl.col('open_interest') > 0))
        for (td,), g in d.group_by(['trading_day']):
            g = g.sort('open_interest', descending=True)
            near = g.row(0, named=True)
            far = g.filter((pl.col('cm') > near['cm']) & (pl.col('open_interest') >= MIN_FAR_OI))
            if far.height == 0:
                continue
            far = far.row(0, named=True)
            dm = far['cm'] - near['cm']
            rows.append(dict(product=p, trading_day=td, near=near['symbol'], far=far['symbol'], months_apart=dm,
                             carry=np.log(near['px'] / far['px']) * 12 / dm))
    return pl.DataFrame(rows).sort(['product', 'trading_day'])


def backtest(cp, df, signal='carry1m', kind='xs', ts_ref='zero'):
    """返回组合日收益。kind='xs' 截面排名；'ts' 时序。"""
    cp = cp.sort(['product', 'trading_day']).with_columns(
        pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('carry1_12'),
        pl.col('carry').cum_sum().over('product').alias('_cs'), pl.col('carry').cum_count().over('product').alias('_cn'))
    cp = cp.with_columns((pl.col('_cs') / pl.col('_cn')).alias('carry_hist_mean'))
    # 品种池：月末主力流动性与历史长度（同 TSMOM）
    elig = (df.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
              .group_by(['product', 'm']).agg(pl.all().sort_by('trading_day').last())
              .select('product', 'trading_day', 'vol60', 'n_obs'))
    x = cp.join(elig, on=['product', 'trading_day'], how='inner').filter(
        (pl.col('vol60') >= MIN_VOL_LOTS) & (pl.col('n_obs') >= 60) & pl.col(signal).is_not_null())
    out = []
    for (td,), g in x.group_by(['trading_day']):
        n = g.height
        if n < 4:
            continue
        if kind == 'xs':
            rk = g[signal].rank('average').to_numpy()
            raw = rk - (n + 1) / 2
            z = 1 / raw[raw > 0].sum() if (raw > 0).any() else 0
            w = raw * z
        else:
            ref = 0 if ts_ref == 'zero' else g['carry_hist_mean'].to_numpy()
            w = np.where(g[signal].to_numpy() > ref, 1.0, -1.0) * 2 / n
        out.append(pl.DataFrame({'product': g['product'], 'trading_day': [td] * n, 'w_set': w}))
    W = pl.concat(out)
    d = (df.select('product', 'trading_day', 'ret').join(W, on=['product', 'trading_day'], how='left')
           .sort(['product', 'trading_day']))
    # 每个月末都重新设定：当月不在池子里的品种仓位清零
    month_ends = W.select('trading_day').unique().with_columns(pl.lit(True).alias('_me'))
    d = d.join(month_ends, on='trading_day', how='left').with_columns(
        pl.when(pl.col('_me')).then(pl.col('w_set').fill_null(0.0)).otherwise(pl.col('w_set')).alias('w_set'))
    d = d.with_columns(pl.col('w_set').forward_fill().over('product').alias('w_hold'))
    d = d.with_columns(pl.col('w_hold').shift(1).over('product').alias('w'))   # 月末设定，次日起生效
    port = (d.filter(pl.col('w').is_not_null()).group_by('trading_day')
              .agg((pl.col('w') * pl.col('ret').fill_null(0)).sum().alias('ret'), (pl.col('w') != 0).sum().alias('n'))
              .sort('trading_day'))
    return port.filter(pl.col('trading_day') >= pl.date(2006, 1, 1))


def stats(r):
    r = np.asarray(r, dtype=float)
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    mu, sd = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
    return dict(ann_ret=mu, ann_vol=sd, sharpe=mu / sd, max_dd=dd, t=r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


def monthly(port):
    return port.group_by(pl.col('trading_day').dt.truncate('1mo').alias('m')).agg(((1 + pl.col('ret')).product() - 1).alias('r')).sort('m')


def fmt(s):
    return (f"年化收益 {s['ann_ret']:7.2%}  年化波动 {s['ann_vol']:6.2%}  夏普 {s['sharpe']:5.2f}  "
            f"最大回撤 {s['max_dd']:7.2%}  t值 {s['t']:5.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    cp = carry_panel()
    cp.write_parquet(f'{OUT}/carry_panel.parquet')
    print(f'carry 面板: {cp["product"].n_unique()} 个品种, {cp.height} 个品种-月；相差月数分布',
          cp['months_apart'].value_counts().sort('months_apart').head(6).rows())
    print('carry 年化分位数（%）:', [round(cp['carry'].quantile(q) * 100, 1) for q in (0.05, 0.25, 0.5, 0.75, 0.95)])
    df = T.load_cache()
    df = df.filter(~pl.col('product').is_in(list(FINANCIAL)))
    res = {}
    for name, kw in [('截面 carry1m', dict(signal='carry', kind='xs')), ('截面 carry1-12', dict(signal='carry1_12', kind='xs')),
                     ('时序 carry1m（阈值 0）', dict(signal='carry', kind='ts')),
                     ('时序 carry1m（阈值 历史均值）', dict(signal='carry', kind='ts', ts_ref='hist'))]:
        port = backtest(cp, df, **kw)
        res[name] = port
        port.write_parquet(f"{OUT}/paper_{kw['signal']}_{kw['kind']}{'_hist' if kw.get('ts_ref') == 'hist' else ''}.parquet")
        m = monthly(port)['r'].to_numpy()
        sk = ((m - m.mean()) ** 3).mean() / m.std() ** 3
        ku = ((m - m.mean()) ** 4).mean() / m.std() ** 4
        print(f'{name:22s} {fmt(stats(port["ret"]))}  月度偏度 {sk:5.2f} 峰度 {ku:5.1f}  平均品种数 {port["n"].mean():.0f}')
    # 分时段
    print()
    for name in ('截面 carry1m', '截面 carry1-12'):
        for lo in (2010, 2016):
            q = res[name].filter(pl.col('trading_day').dt.year() >= lo)
            print(f'{name} {lo} 年起: {fmt(stats(q["ret"]))}')
    # 与 TSMOM（商品，论文口径）的相关性
    tp = T.portfolio(T.backtest(df, k=12, lag=0))
    tm = monthly(tp.select('trading_day', pl.col('gross').alias('ret')))
    print()
    for name in ('截面 carry1m', '截面 carry1-12'):
        j = monthly(res[name]).join(tm, on='m', suffix='_ts')
        pl.DataFrame({'name': [name], 'corr': [float(np.corrcoef(j['r'], j['r_ts'])[0, 1])]}).write_parquet(
            f"{OUT}/corr_tsmom_{'1m' if '1m' in name else '1_12'}.parquet")
        print(f'{name} 与 TSMOM（商品）月度相关性: {np.corrcoef(j["r"], j["r_ts"])[0, 1]:.2f}')
    yr = {n: res[n].group_by(pl.col('trading_day').dt.year().alias('y')).agg(((1 + pl.col('ret')).product() - 1).alias('r')).sort('y')
          for n in ('截面 carry1m', '截面 carry1-12')}
    print('\n逐年（截面 carry1m / carry1-12）: ' + '  '.join(
        f"{a}:{b:+.1%}/{c:+.1%}" for (a, b), (_, c) in zip(yr['截面 carry1m'].rows(), yr['截面 carry1-12'].rows())))


if __name__ == '__main__':
    main()
