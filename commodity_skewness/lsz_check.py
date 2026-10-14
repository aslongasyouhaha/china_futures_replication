"""对照 Li, Sun & Zhou (2021, Journal of Management Science and Engineering) 的偏度结果：
他们：最近月合约持有到到期再换月（含交割月），比例复权连续价格；全部 47 个品种；2003-01 ~ 2018-06；四分组，多（偏度最低）− 空（偏度最高）；
      偏度多空年化 −9.78%（t = −2.55），即右偏品种跑赢左偏品种。
这里用同一套信号（过去 12 个月日收益偏度）与同一套组合方式，只替换合约构造和品种筛选，找出差异来源：
  A 最近月持有到到期（含交割月），不设持仓门槛         —— Li-Sun-Zhou 口径
  B 交割月 ≥ 当前月 + 2 的第一个合约，不设持仓门槛     —— 只改合约规则
  C 同 B，月末合约持仓 ≥ 100 手且有成交               —— 本库偏度复现的主口径
  D 同 A，月末合约持仓 ≥ 100 手且有成交               —— 只加持仓门槛
输出 output/lsz_check.parquet
"""
import glob
import os
import sys

import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
BM = os.path.join(HERE, '..', 'basis_momentum', 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
sys.path.insert(0, os.path.join(HERE, '..', 'carry'))
import carry as CA  # noqa: E402
from pricing import nw_cov  # noqa: E402

EXCH = ('SHFE', 'DCE', 'CZCE', 'INE')
SKIP = {'EC', 'LF', 'PPF', 'VF'}
LSZ_END = pl.date(2018, 7, 1)
MIN_N = 8


def nw_t(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / np.sqrt(nw_cov(x)[0, 0] / len(x))


def nearest_series():
    """每个品种：每天持有当天仍在交易的最近月合约（到期才换月），日收益 = 同一合约相邻两日结算价之比（换月日用新合约的收益）。"""
    out = []
    for f in glob.glob(f'{RAW_DATA}/*/*/daily_contracts.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex not in EXCH or p in SKIP:
            continue
        d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'volume', 'open_interest'])
        d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px')).filter(pl.col('px') > 0)
        d = d.with_columns(pl.col('symbol').map_elements(CA.month_index, return_dtype=pl.Int64).alias('cm'))
        d = d.sort(['symbol', 'trading_day']).with_columns((pl.col('px') / pl.col('px').shift(1).over('symbol') - 1).alias('r'))
        # 当天的最近月合约
        near = d.sort(['trading_day', 'cm']).group_by('trading_day', maintain_order=True).first()
        # 收益用前一天持有的合约：若该合约今天仍有价格就用它的收益，否则用新最近月的收益
        held = near.select(pl.col('trading_day'), pl.col('symbol').shift(1).alias('held')).drop_nulls()
        r_held = held.join(d.select('symbol', 'trading_day', 'r', 'open_interest', 'volume'), left_on=['held', 'trading_day'], right_on=['symbol', 'trading_day'], how='left')
        x = near.select('trading_day', pl.col('r').alias('r_new'), pl.col('symbol').alias('sym'), pl.col('open_interest').alias('oi_new'), pl.col('volume').alias('vol_new'))
        x = x.join(r_held.select('trading_day', 'r', 'open_interest', 'volume'), on='trading_day', how='left')
        x = x.with_columns(pl.col('r').fill_null(pl.col('r_new')).alias('r'), pl.lit(p).alias('product'))
        out.append(x.select('product', 'trading_day', 'sym', 'r', pl.col('oi_new').alias('oi'), pl.col('vol_new').alias('vol')))
    return pl.concat(out).filter(pl.col('r').is_not_null() & pl.col('r').is_finite())


def monthly_from_daily(dd):
    """月收益与月末偏度（过去 12 个月日对数收益），月末持仓、成交（当天最近月合约）。"""
    dd = dd.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'), (pl.col('r') + 1).log().alias('lr'))
    mret = dd.group_by(['product', 'm']).agg(((pl.col('r') + 1).product() - 1).alias('ret_in'))
    mend = dd.sort('trading_day').group_by(['product', 'm']).agg(pl.col('oi').last(), pl.col('vol').last())
    rows = []
    for (p,), g in dd.sort('trading_day').partition_by(['product'], as_dict=True).items():
        hm = g['m'].to_numpy().astype('datetime64[M]')
        lr = g['lr'].to_numpy()
        for m in np.unique(hm):
            w = (hm > m - 12) & (hm <= m)
            x = lr[w]
            if len(x) >= 150 and x.std() > 0:
                rows.append({'product': p, 'm': np.datetime64(m, 'D').item(), 'SK': float(((x - x.mean()) ** 3).mean() / x.std(ddof=1) ** 3)})
    s = pl.DataFrame(rows).with_columns(pl.col('m').cast(pl.Date))
    nxt = mret.select('product', pl.col('m').dt.offset_by('-1mo').alias('m'), pl.col('ret_in').alias('ret'))
    return s.join(nxt, on=['product', 'm']).join(mend, on=['product', 'm'], how='left')


def rule_panel():
    """交割月 ≥ 当前月 + 2 的第一个合约（basis_momentum paper 曲线），偏度用其持有月内日收益。"""
    c = pl.read_parquet(os.path.join(BM, 'curve_paper.parquet')).filter(pl.col('n') == 1).select('product', 'm', 'ret', pl.col('open_interest').alias('oi'), pl.col('volume').alias('vol'))
    d = pl.read_parquet(os.path.join(BM, 'daily_paper.parquet')).filter((pl.col('n') == 1) & pl.col('r').is_not_null())
    d = d.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('hm'), (pl.col('r') + 1).log().alias('lr')).sort(['product', 'trading_day'])
    rows = []
    for (p,), g in d.partition_by(['product'], as_dict=True).items():
        hm = g['hm'].to_numpy().astype('datetime64[M]')
        lr = g['lr'].to_numpy()
        for m in np.unique(hm):
            w = (hm > m - 12) & (hm <= m)
            x = lr[w]
            if len(x) >= 150 and x.std() > 0:
                rows.append({'product': p, 'm': np.datetime64(m, 'D').item(), 'SK': float(((x - x.mean()) ** 3).mean() / x.std(ddof=1) ** 3)})
    s = pl.DataFrame(rows).with_columns(pl.col('m').cast(pl.Date))
    keep = [p for p in c['product'].unique().to_list() if p not in SKIP]
    return s.join(c, on=['product', 'm']).filter(pl.col('product').is_in(keep) & pl.col('ret').is_not_null() & pl.col('ret').is_finite())


def ls(x):
    rows = []
    for (m,), g in x.drop_nulls(['SK', 'ret']).group_by(['m'], maintain_order=True):
        if g.height < MIN_N:
            continue
        o = g.sort(['SK', 'product'])
        k = max(1, g.height // 4)
        rows.append({'m': m, 'r': o.head(k)['ret'].mean() - o.tail(k)['ret'].mean(), 'lo': o.head(k)['ret'].mean(), 'hi': o.tail(k)['ret'].mean(), 'N': g.height})
    return pl.DataFrame(rows).sort('m')


def main():
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
    A = monthly_from_daily(nearest_series())
    B = rule_panel()
    liq = (pl.col('oi') >= 100) & (pl.col('vol') > 0)
    # 剔除纤维板 2019-12 报价单位变更（两种口径都会受影响）
    fb = ~((pl.col('product') == 'FB') & (pl.col('m') >= pl.date(2019, 9, 1)))
    sets = {'A Li-Sun-Zhou 口径：最近月持有到到期，无门槛': A.filter(fb), 'B 交割月 ≥ 当前月 + 2，无门槛': B.filter(fb),
            'C 交割月 ≥ 当前月 + 2，持仓 ≥ 100 手（本库主口径）': B.filter(fb & liq), 'D 最近月持有到到期，持仓 ≥ 100 手': A.filter(fb & liq)}
    rows = []
    for lab, x in sets.items():
        r = ls(x)
        for per, cond in (('2005-01 ~ 2018-06（对应 Li-Sun-Zhou 样本期）', pl.col('m') < LSZ_END), ('全样本 2005 ~ 2026', pl.lit(True)), ('2018-07 ~ 2026', pl.col('m') >= LSZ_END)):
            q = r.filter(cond)
            y = q['r'].to_numpy()
            rows.append({'设定': lab, '时期': per, '月数': len(y), '每月品种中位数': q['N'].median(), '多空年化%': y.mean() * 1200, 't': nw_t(y),
                         '左偏组年化%': q['lo'].mean() * 1200, '右偏组年化%': q['hi'].mean() * 1200})
    t = pl.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    t.write_parquet(os.path.join(OUT, 'lsz_check.parquet'))
    print(t.with_columns(pl.col(pl.Float64).round(2)))
    # 最近月口径中，交割月 / 退出月持有的比例
    print('A 口径月末持仓中位数', A['oi'].median(), '；C 口径', B.filter(liq)['oi'].median())


if __name__ == '__main__':
    main()
