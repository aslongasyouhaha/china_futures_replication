"""合约乘数估计：乘数 ≈ 成交额 / (成交量 × 价格)，按品种按年取中位数，再对齐到最近的标准乘数。"""
import glob
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
STD = np.array([1, 2, 3, 5, 10, 15, 16, 20, 22, 25, 30, 40, 50, 60, 90, 100, 200, 300, 500, 1000, 10000, 20000], dtype=float)


def snap(x):
    if x is None or not np.isfinite(x) or x <= 0:
        return None
    return float(STD[np.argmin(np.abs(np.log(STD) - np.log(x)))])


def estimate_multipliers():
    rows = []
    for f in glob.glob(f'{DATA}/*/*/daily_contracts.parquet'):
        p = os.path.basename(os.path.dirname(f))
        d = pl.read_parquet(f, columns=['trading_day', 'close', 'volume', 'amount']).filter(
            (pl.col('volume') > 0) & (pl.col('amount') > 0) & (pl.col('close') > 0))
        d = d.with_columns((pl.col('amount') / pl.col('volume') / pl.col('close')).alias('m'))
        for (y,), g in d.group_by(pl.col('trading_day').dt.year()):
            raw = float(g['m'].median())
            rows.append(dict(product=p, year=int(y), raw=raw, mult=snap(raw)))
    return pl.DataFrame(rows).sort(['product', 'year'])


if __name__ == '__main__':
    m = estimate_multipliers()
    m.write_parquet(f'{RESEARCH}/common/output/multipliers.parquet')
    last = m.group_by('product').agg(pl.col('mult').last(), pl.col('raw').last(), pl.col('mult').n_unique().alias('n'),
                                     pl.col('year').max())
    pl.Config.set_tbl_rows(100)
    print(last.filter(pl.col('product').is_in(['RB', 'CU', 'AU', 'AG', 'SC', 'M', 'I', 'J', 'TA', 'MA', 'CF', 'IF', 'IC', 'IM',
                                               'IH', 'T', 'TF', 'TS', 'TL', 'LC', 'SI', 'EC', 'AP', 'FB', 'BB'])).sort('product'))
    ch = m.sort(['product', 'year']).with_columns((pl.col('mult') != pl.col('mult').shift(1).over('product')).alias('chg'))
    print('乘数变化过的品种:'); print(ch.filter(pl.col('chg') & pl.col('mult').shift(1).over('product').is_not_null()).select('product', 'year', 'raw', 'mult'))
