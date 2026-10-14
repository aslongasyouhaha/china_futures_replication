"""交易成本（对应论文 Table 7 的盈亏平衡思路）：一个来回按 1 跳买卖价差计（入场、出场各付半跳），不含交易所手续费。
成本(bp) = 最小变动价位 / 主力合约收盘价，按品种取样本期平均；与择时策略每笔平均收益对比。输出: output/costs.parquet"""
import os

import polars as pl

from imom import MAIN, COMMOD, DATA, OUT
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

ticks = pl.read_parquet(f'{RESEARCH}/common/output/ticks.parquet')
rows = []
for tag, products in (('fin', MAIN), ('com', COMMOD)):
    tim = pl.read_parquet(os.path.join(OUT, f'{tag}_timing.parquet'))
    for ex, p, name in products:
        d = pl.read_parquet(f'{DATA}/{ex}/{p}/daily_main.parquet', columns=['trading_day', 'close'])
        d = d.with_columns(pl.col('trading_day').dt.year().cast(pl.Int64).alias('year')).join(ticks.filter(pl.col('product') == p), on='year', how='left')
        bp = (d['tick'] / d['close']).mean() * 1e4
        for s in ('η(r1)', 'η(r12)', 'η(r1,r12)'):
            g = tim.filter((pl.col('品种') == name) & (pl.col('策略') == s))['每笔平均收益bp'][0]
            rows.append({'组': tag, '品种': name, '策略': s, '每笔平均收益bp': g, '一跳成本bp': bp, '扣一跳后bp': g - bp})
pl.DataFrame(rows).write_parquet(os.path.join(OUT, 'costs.parquet'))
with pl.Config(tbl_rows=100):
    print(pl.DataFrame(rows).filter(pl.col('策略') == 'η(r1,r12)').with_columns(pl.col(pl.Float64).round(2)))
