"""Basis-Momentum 复现的数据构造 —— Boons & Porras Prado (2019, Journal of Finance)

论文：第 n 近月合约 = 到期在 t+2 之后的合约里按到期顺序排第 n 个（沿用 Szymanowska et al. 2014，避免持有临近到期的合约）；
      月末价格，持有一个月，超额收益 R = F_{t+1}/F_t − 1（同一合约）。
中国对应：
  paper（主结果）：交割月 ≥ 当前月 + 2 的合约按交割月排序取第 1~4 个（与三因子复现的 F1 规则一致：持有期内不会进入交割月）
  liquid（中国化对照）：第 1 个 = 交割月 ≥ 当前月 + 2 中持仓最大的合约；第 2、3 个 = 更远月份中持仓最大的合约（中国的流动性集中在 1、5、9 等月份）
  价格用官方结算价（缺失时用收盘价）。
输出（output/）：
  curve_{v}.parquet  每个品种每个月末：第 1~4 个合约的代码、价格、持仓、成交量、下月末价格与持有收益
  daily_{v}.parquet  第 1~4 个合约在持有月内的逐日收益（用于滚动 beta、月内波动、偏度等）
"""
import glob
import os
import sys

import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
sys.path.insert(0, os.path.join(HERE, '..', 'carry'))
import carry as C  # noqa: E402
import factors as FA  # noqa: E402

DATA = RAW_DATA
NMAX = 4


def load(f, p):
    d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'volume', 'amount', 'open_interest'])
    d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'), pl.lit(p).alias('product'),
                       pl.col('trading_day').dt.truncate('1mo').alias('m')).filter(pl.col('px') > 0)
    d = d.sort(['symbol', 'trading_day']).with_columns((pl.col('px') / pl.col('px').shift(1).over('symbol') - 1).alias('r'))
    me = d.group_by('m').agg(pl.col('trading_day').max().alias('me'))
    e = d.join(me, on='m').filter(pl.col('trading_day') == pl.col('me'))
    e = e.with_columns(pl.col('symbol').map_elements(C.month_index, return_dtype=pl.Int64).alias('cm'),
                       (pl.col('m').dt.year() * 12 + pl.col('m').dt.month() - 1).alias('now'))
    nxt = e.select('symbol', pl.col('m').dt.offset_by('-1mo').alias('m'), pl.col('px').alias('px_next'))
    e = e.join(nxt, on=['symbol', 'm'], how='left').filter(pl.col('cm') >= pl.col('now') + 2)
    return e.select('product', 'symbol', 'm', 'px', 'px_next', 'volume', 'open_interest', 'cm'), d.select('product', 'symbol', 'trading_day', 'm', 'r', 'px', 'volume', 'amount', 'open_interest')


def pick(e, v):
    """返回 product, m, n, 合约信息。"""
    e = e.sort(['m', 'cm'])
    if v == 'paper':
        return e.with_columns(pl.col('cm').rank('ordinal').over('m').alias('n')).filter(pl.col('n') <= NMAX)
    rows = []
    for (m,), g in e.group_by(['m']):
        last = -1
        for n in range(1, NMAX + 1):
            h = g.filter(pl.col('cm') > last)
            if h.height == 0:
                break
            r = h.sort(['open_interest', 'cm'], descending=[True, False]).row(0, named=True)
            r['n'] = n
            rows.append(r)
            last = r['cm']
    return pl.DataFrame(rows, schema={**e.schema, 'n': pl.UInt32}) if rows else None


def main():
    os.makedirs(OUT, exist_ok=True)
    curves = {'paper': [], 'liquid': []}
    daily = {'paper': [], 'liquid': []}
    for f in sorted(glob.glob(f'{DATA}/*/*/daily_contracts.parquet')):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in FA.EXCLUDE:
            continue
        e, d = load(f, p)
        for v in curves:
            c = pick(e, v)
            if c is None:
                continue
            c = c.with_columns((pl.col('px_next') / pl.col('px') - 1).alias('ret'))
            curves[v].append(c)
            hold = c.filter(pl.col('n') <= NMAX).select('symbol', pl.col('m').dt.offset_by('1mo').alias('m'), pl.col('m').alias('m_form'), 'n')
            daily[v].append(d.join(hold, on=['symbol', 'm']).select('product', 'trading_day', 'm_form', 'n', 'symbol', 'r', 'px', 'volume', 'amount', 'open_interest'))
        print(p, end=' ', flush=True)
    print()
    for v in curves:
        pl.concat(curves[v]).sort(['product', 'm', 'n']).write_parquet(os.path.join(OUT, f'curve_{v}.parquet'))
        pl.concat(daily[v]).sort(['product', 'trading_day', 'n']).write_parquet(os.path.join(OUT, f'daily_{v}.parquet'))
    FA.universe().write_parquet(os.path.join(OUT, 'universe.parquet'))
    print('done')


if __name__ == '__main__':
    main()
