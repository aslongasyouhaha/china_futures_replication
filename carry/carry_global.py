"""Carry 扩展到股指与国债期货（论文口径：不扣成本、月末收盘成交），并合成全球 carry 因子（GCF）。

期限结构 carry 的定义同商品：ln(F_近 / F_远) × 12 / 相差月数，近月 = 持仓最大的合约，远月 = 比它更远、持仓最大的合约。
  股指期货（IF/IH/IC/IM）现金交割，个人可持有到期：只剔除当月合约（月末时已到期或即将到期）。
  国债期货（TS/TF/T/TL）实物交割，个人不能进入交割月：同商品，剔除交割月和交割前一个月。
  股指 carry ≈ 股息率 − 资金成本（另含对冲需求造成的基差）；国债 carry ≈ 票息 − 资金成本 + 骑乘收益。
每类内部：时序（carry > 0 做多，总敞口 2）为主；截面排名（品种 >= 3 时）作参考。
GCF：商品、股指、国债的截面 carry1-12 三者各按全样本波动缩放到 10%，再等权平均（同论文，用全样本波动缩放）。
"""
import glob
import os

import numpy as np
import polars as pl

import carry as C

EQUITY = ['IF', 'IH', 'IC', 'IM']
BOND = ['TS', 'TF', 'T', 'TL']
OUT = C.OUT


def fin_panel(products, skip_months, min_far_oi=200):
    rows = []
    for p in products:
        f = f'{C.DATA}/CFFEX/{p}/daily_contracts.parquet'
        d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'open_interest'])
        d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'),
                           pl.col('trading_day').dt.truncate('1mo').alias('month'))
        me = d.group_by('month').agg(pl.col('trading_day').max().alias('me'))
        d = d.join(me, on='month').filter(pl.col('trading_day') == pl.col('me'))
        d = d.with_columns(pl.col('symbol').map_elements(C.month_index, return_dtype=pl.Int64).alias('cm'),
                           (pl.col('trading_day').dt.year() * 12 + pl.col('trading_day').dt.month() - 1).alias('now'))
        d = d.filter((pl.col('cm') >= pl.col('now') + skip_months) & (pl.col('px') > 0) & (pl.col('open_interest') > 0))
        for (td,), g in d.group_by(['trading_day']):
            g = g.sort('open_interest', descending=True)
            near = g.row(0, named=True)
            far = g.filter((pl.col('cm') > near['cm']) & (pl.col('open_interest') >= min_far_oi))
            if far.height == 0:
                continue
            far = far.row(0, named=True)
            dm = far['cm'] - near['cm']
            rows.append(dict(product=p, trading_day=td, near=near['symbol'], far=far['symbol'], months_apart=dm,
                             carry=np.log(near['px'] / far['px']) * 12 / dm))
    return pl.DataFrame(rows).sort(['product', 'trading_day'])


def class_backtest(cp, df, signal, kind, min_n):
    old = C.backtest.__defaults__
    port = backtest_n(cp, df, signal, kind, min_n)
    return port


def backtest_n(cp, df, signal, kind, min_n):
    """同 carry.backtest，但允许更少的品种（股指、国债各只有 4 个）。"""
    cp = cp.sort(['product', 'trading_day']).with_columns(
        pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('carry1_12'))
    elig = (df.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
              .group_by(['product', 'm']).agg(pl.all().sort_by('trading_day').last())
              .select('product', 'trading_day', 'vol60', 'n_obs'))
    x = cp.join(elig, on=['product', 'trading_day'], how='inner').filter(
        (pl.col('vol60') >= C.MIN_VOL_LOTS) & (pl.col('n_obs') >= 60) & pl.col(signal).is_not_null())
    out = []
    for (td,), g in x.group_by(['trading_day']):
        n = g.height
        if n < min_n:
            continue
        if kind == 'xs':
            raw = g[signal].rank('average').to_numpy() - (n + 1) / 2
            if not (raw > 0).any():
                continue
            w = raw / raw[raw > 0].sum()
        else:
            w = np.where(g[signal].to_numpy() > 0, 1.0, -1.0) * 2 / n
        out.append(pl.DataFrame({'product': g['product'], 'trading_day': [td] * n, 'w_set': w}))
    W = pl.concat(out)
    d = df.select('product', 'trading_day', 'ret').join(W, on=['product', 'trading_day'], how='left').sort(['product', 'trading_day'])
    me = W.select('trading_day').unique().with_columns(pl.lit(True).alias('_me'))
    d = d.join(me, on='trading_day', how='left').with_columns(
        pl.when(pl.col('_me')).then(pl.col('w_set').fill_null(0.0)).otherwise(pl.col('w_set')).alias('w_set'))
    d = d.with_columns(pl.col('w_set').forward_fill().over('product').alias('wh'))
    d = d.with_columns(pl.col('wh').shift(1).over('product').alias('w'))
    port = (d.filter(pl.col('w').is_not_null()).group_by('trading_day')
              .agg((pl.col('w') * pl.col('ret').fill_null(0)).sum().alias('ret'), (pl.col('w') != 0).sum().alias('n'))
              .sort('trading_day'))
    return port.filter(pl.col('trading_day') >= pl.date(2006, 1, 1))


def main():
    eq = fin_panel(EQUITY, skip_months=1)
    bd = fin_panel(BOND, skip_months=2)
    pl.concat([eq, bd]).write_parquet(f'{OUT}/carry_panel_financial.parquet')
    for name, cp in (('股指', eq), ('国债', bd)):
        print(f'{name} carry 年化（%）均值/标准差:', {p: (round(g['carry'].mean() * 100, 1), round(g['carry'].std() * 100, 1))
                                             for (p,), g in cp.group_by(['product'])})
    df = C.T.load_cache()
    comm = C.carry_panel() if not os.path.exists(f'{OUT}/carry_panel.parquet') else pl.read_parquet(f'{OUT}/carry_panel.parquet')
    res = {}
    res['商品 截面 carry1-12'] = backtest_n(comm, df.filter(~pl.col('product').is_in(list(C.FINANCIAL))), 'carry1_12', 'xs', 4)
    for name, cp, prods in (('股指', eq, EQUITY), ('国债', bd, BOND)):
        sub = df.filter(pl.col('product').is_in(prods))
        for sig, lab in (('carry', 'carry1m'), ('carry1_12', 'carry1-12')):
            res[f'{name} 时序 {lab}'] = backtest_n(cp, sub, sig, 'ts', 1)
            res[f'{name} 截面 {lab}'] = backtest_n(cp, sub, sig, 'xs', 3)
    print()
    pl.concat([v.with_columns(pl.lit(k).alias('strategy')) for k, v in res.items()]).write_parquet(f'{OUT}/paper_by_class.parquet')
    for k, v in res.items():
        print(f'{k:20s} {C.fmt(C.stats(v["ret"]))}  起点 {v["trading_day"].min()}  平均品种数 {v["n"].mean():.1f}')
    # 全球 carry 因子：三类各缩放到 10% 年化波动后等权（月度）
    # 股指常年贴水（对冲需求），时序 carry 几乎总是做多、等于持有股指 → 三类都用截面版本（同论文 GCF）
    parts = {'商品': res['商品 截面 carry1-12'], '股指': res['股指 截面 carry1-12'], '国债': res['国债 截面 carry1-12']}
    m = None
    for k, v in parts.items():
        s = v['ret'].std() * np.sqrt(C.ANN)
        x = v.select('trading_day', (pl.col('ret') * 0.10 / s).alias(k))
        m = x if m is None else m.join(x, on='trading_day', how='outer_coalesce')
    m = m.sort('trading_day')
    cols = list(parts)
    m = m.with_columns(pl.mean_horizontal([pl.col(c) for c in cols]).alias('GCF'))   # 当天有数据的类别等权
    m.write_parquet(f'{OUT}/gcf_daily.parquet')
    print('\n== 全球 carry 因子（三类各缩放到 10% 波动后等权）==')
    print(f'GCF 2006 年起 {C.fmt(C.stats(m["GCF"]))}')
    q = m.drop_nulls(cols)
    print(f'GCF 2020-05 起（三类都有）{C.fmt(C.stats(q["GCF"]))}')
    print(f'仅商品（同样缩放到 10%）同期 {C.fmt(C.stats(q["商品"]))}')
    mm = q.group_by(pl.col('trading_day').dt.truncate('1mo')).agg([((1 + pl.col(c)).product() - 1).alias(c) for c in cols])
    print('三类月度相关性:\n', np.round(np.corrcoef(np.vstack([mm[c] for c in cols])), 2))


if __name__ == '__main__':
    main()
