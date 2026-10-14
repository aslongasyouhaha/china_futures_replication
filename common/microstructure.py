"""从分钟数据估计：
1. 每个品种每年的买卖价差（Roll 估计，单位：跳）
2. 每个品种每天主力合约的 VWAP（原始价格，及与 close_adj 一致的复权 VWAP）
"""
import glob
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
OUT = f'{RESEARCH}/common/output'


def roll_spread(dp, dp1):
    c = np.cov(dp, dp1)[0, 1]
    return 2 * np.sqrt(-c) if c < 0 else np.nan


def main():
    ticks = pl.read_parquet(f'{OUT}/ticks.parquet').select('product', pl.col('year').cast(pl.Int32), 'tick')
    spreads, vwaps = [], []
    for f in sorted(glob.glob(f'{DATA}/*/*/minute_main.parquet')):
        p = os.path.basename(os.path.dirname(f))
        m = pl.read_parquet(f, columns=['trading_day', 'symbol', 'close', 'volume', 'adj_factor'])
        # VWAP：成交量加权的分钟收盘价；无成交的日子用当日分钟收盘均价
        v = m.group_by('trading_day').agg(
            pl.when(pl.col('volume').sum() > 0).then((pl.col('close') * pl.col('volume')).sum() / pl.col('volume').sum())
              .otherwise(pl.col('close').mean()).alias('vwap'),
            pl.col('adj_factor').first())
        vwaps.append(v.with_columns(pl.lit(p).alias('product'), (pl.col('vwap') * pl.col('adj_factor')).alias('vwap_adj')).drop('adj_factor'))
        # Roll 价差：同一合约同一交易日内相邻分钟价格变动的自协方差
        m = m.with_columns((pl.col('close') - pl.col('close').shift(1).over(['symbol', 'trading_day'])).alias('dp'))
        m = m.with_columns(pl.col('dp').shift(1).over(['symbol', 'trading_day']).alias('dp1')).drop_nulls(['dp', 'dp1'])
        for (y,), g in m.group_by(pl.col('trading_day').dt.year()):
            if g.height >= 2000:
                spreads.append(dict(product=p, year=int(y), spread_px=float(roll_spread(g['dp'].to_numpy(), g['dp1'].to_numpy()))))
        print(p, flush=True, end=' ')
    s = pl.DataFrame(spreads).with_columns(pl.col('year').cast(pl.Int32)).join(ticks, on=['product', 'year'], how='left')
    s = s.with_columns((pl.col('spread_px') / pl.col('tick')).alias('spread_ticks'))
    s.write_parquet(f'{OUT}/spreads.parquet')
    pl.concat(vwaps).write_parquet(f'{OUT}/vwap.parquet')
    print('\nspreads', s.height, 'vwap rows', sum(x.height for x in vwaps))


if __name__ == '__main__':
    main()
