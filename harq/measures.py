"""日度已实现度量（5 分钟收益，同一合约、连续时段内，相邻 K 线间隔 ≤ 10 分钟，口径同 har_rv）：
  RV = Σr²（日方差，未年化）；RQ = M/3 · Σr⁴；BPV = (π/2)·Σ|r_i||r_{i+1}|；RV+ / RV− = 正 / 负收益的平方和；M = 当日收益个数
剔除收益个数不足滚动 60 日中位数一半的日子（半天交易、数据缺失）。输出 output/measures_{品种}.parquet。
"""
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
DATA = RAW_DATA
# 指数：沪深 300；“个股”对应：历史长、流动性好的商品期货
PRODUCTS = [('CFFEX', 'IF'), ('SHFE', 'CU'), ('SHFE', 'AL'), ('SHFE', 'ZN'), ('SHFE', 'AU'), ('SHFE', 'AG'), ('SHFE', 'RB'), ('SHFE', 'RU'),
            ('SHFE', 'BU'), ('DCE', 'M'), ('DCE', 'Y'), ('DCE', 'P'), ('DCE', 'C'), ('DCE', 'I'), ('DCE', 'J'), ('DCE', 'L'), ('DCE', 'V'),
            ('CZCE', 'TA'), ('CZCE', 'SR'), ('CZCE', 'CF'), ('CZCE', 'MA'), ('CZCE', 'RM'), ('CZCE', 'FG')]


def measures(ex, p):
    m = pl.read_parquet(f'{DATA}/{ex}/{p}/minute_main.parquet', columns=['trading_day', 'datetime', 'symbol', 'close'])
    b = (m.with_columns(pl.col('datetime').dt.truncate('5m').alias('bucket'))
          .group_by(['trading_day', 'symbol', 'bucket']).agg(pl.col('close').last(), pl.col('datetime').max().alias('t'))
          .sort(['trading_day', 'symbol', 't']))
    b = b.with_columns((pl.col('close').log() - pl.col('close').log().shift(1).over(['trading_day', 'symbol'])).alias('r'),
                       (pl.col('t') - pl.col('t').shift(1).over(['trading_day', 'symbol'])).dt.total_minutes().alias('gap'))
    b = b.filter(pl.col('r').is_not_null() & (pl.col('gap') <= 10)).sort(['trading_day', 't'])
    b = b.with_columns(pl.col('r').abs().shift(1).over(['trading_day', 'symbol']).alias('r_prev_abs'))
    d = b.group_by('trading_day').agg(
        (pl.col('r') ** 2).sum().alias('RV'), pl.len().alias('M'), (pl.col('r') ** 4).sum().alias('r4'),
        (pl.col('r').abs() * pl.col('r_prev_abs')).sum().alias('bp'),
        (pl.when(pl.col('r') > 0).then(pl.col('r') ** 2).otherwise(0)).sum().alias('RVp'),
        (pl.when(pl.col('r') < 0).then(pl.col('r') ** 2).otherwise(0)).sum().alias('RVn')).sort('trading_day')
    d = d.with_columns((pl.col('M') / 3 * pl.col('r4')).alias('RQ'), (np.pi / 2 * pl.col('bp')).alias('BPV'),
                       pl.col('M').rolling_median(60, min_periods=10, center=True).alias('M_med'))
    d = d.filter((pl.col('M') >= 0.5 * pl.col('M_med')) & (pl.col('RV') > 0))
    return d.select('trading_day', 'RV', 'RQ', 'BPV', 'RVp', 'RVn', 'M')


def main():
    os.makedirs(OUT, exist_ok=True)
    for ex, p in PRODUCTS:
        d = measures(ex, p)
        d.write_parquet(os.path.join(OUT, f'measures_{p}.parquet'))
        print(p, d.height, str(d['trading_day'].min()), flush=True)


if __name__ == '__main__':
    main()
