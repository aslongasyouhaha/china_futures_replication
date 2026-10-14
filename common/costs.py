"""交易成本估计：按品种、按天估计单边成本率（占名义价值的比例）。

成本 = 手续费 + 滑点
  手续费：统一假设 1bp
  滑点  ：n 跳 × 最小变动价位 / 价格；n 随流动性变化（见 SCENARIOS）
最小变动价位（一跳）从分钟数据里估计：每个品种每年，价格相邻变动的最小正值（且占比足够高）。
"""
import glob
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
COMMISSION = 0.0001
# 情形: (日均量 >= 1 万手, 2 千~1 万手, < 2 千手) 时各吃几跳
SCENARIOS = {'乐观': (0.5, 1, 2), '基准': (1, 2, 4), '悲观': (2, 4, 8)}


def estimate_ticks():
    """每个品种每年的一跳大小。"""
    rows = []
    for f in glob.glob(f'{DATA}/*/*/minute_main.parquet'):
        p = os.path.basename(os.path.dirname(f))
        m = pl.read_parquet(f, columns=['trading_day', 'symbol', 'close'])
        m = m.filter(pl.col('trading_day').dt.ordinal_day() <= 60)   # 每年前 60 天足够估计
        d = (m.with_columns((pl.col('close') - pl.col('close').shift(1).over('symbol')).abs().round(6).alias('dp'))
               .filter(pl.col('dp') > 0))
        for (y,), g in d.group_by(pl.col('trading_day').dt.year()):
            vc = g['dp'].value_counts().sort('dp')
            n = vc['count'].sum()
            # 最小的、出现频率至少 1% 的价格变动
            cand = vc.filter(pl.col('count') >= 0.01 * n)
            if cand.height:
                rows.append(dict(product=p, year=y, tick=float(cand['dp'][0])))
    return pl.DataFrame(rows).sort(['product', 'year'])


def cost_table(ticks, scenario='基准'):
    """每个品种每天的单边成本率。"""
    lo, mid, hi = SCENARIOS[scenario]
    parts = []
    for f in glob.glob(f'{DATA}/*/*/daily_main.parquet'):
        p = os.path.basename(os.path.dirname(f))
        d = pl.read_parquet(f, columns=['trading_day', 'symbol', 'close', 'volume'])
        parts.append(d.with_columns(pl.lit(p).alias('product')))
    d = pl.concat(parts).sort(['product', 'trading_day'])
    d = d.with_columns(pl.col('volume').rolling_mean(60, min_periods=20).over('product').alias('v60'),
                       pl.col('trading_day').dt.year().alias('year'),
                       # 换月日: 主力合约代码变化
                       (pl.col('symbol') != pl.col('symbol').shift(1).over('product')).fill_null(False).alias('roll'))
    d = d.join(ticks.with_columns(pl.col('year').cast(pl.Int32)), on=['product', 'year'], how='left').with_columns(pl.col('tick').forward_fill().backward_fill().over('product'))
    n_ticks = pl.when(pl.col('v60') >= 10000).then(lo).when(pl.col('v60') >= 2000).then(mid).otherwise(hi)
    return d.select('product', 'trading_day', 'roll',
                    (COMMISSION + n_ticks * pl.col('tick') / pl.col('close')).alias('cost_rate'))
