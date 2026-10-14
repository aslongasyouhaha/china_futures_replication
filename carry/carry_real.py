"""Carry 实盘版：复用 time_series_momentum/tsmom_real.py 的实盘框架（资金 1 亿、次日 VWAP、手续费、半价差、平方根冲击、换月成本、容量上限）。
权重在每个月末、在可交易品种（60 日成交额中位数 >= 1 亿元）里按 carry 排名计算，然后交给框架执行。
"""
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'time_series_momentum'))
import tsmom_real as R  # noqa: E402

import carry as C  # noqa: E402

OUT = os.path.join(HERE, 'output')


def weights(cp, d, signal='carry1_12', kind='xs', adv_min=R.ADV_MIN):
    cp = cp.sort(['product', 'trading_day']).with_columns(
        pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('carry1_12'))
    tradable = d.select('product', 'trading_day', 'adv_val', 'lot_value')
    x = cp.join(tradable, on=['product', 'trading_day'], how='inner').filter(
        (pl.col('adv_val') >= adv_min) & pl.col('lot_value').is_finite() & pl.col(signal).is_not_null())
    out = []
    for (td,), g in x.group_by(['trading_day']):
        n = g.height
        if n < 4:
            continue
        if kind == 'xs':
            raw = g[signal].rank('average').to_numpy() - (n + 1) / 2
            w = raw / raw[raw > 0].sum()
        else:
            w = np.where(g[signal].to_numpy() > 0, 1.0, -1.0) * 2 / n
        out.append(pl.DataFrame({'product': g['product'], 'trading_day': [td] * n, 'w': w}))
    return pl.concat(out)


def summary(r):
    s, c = R.summary(r)
    return s, c, r


def main():
    cp = pl.read_parquet(f'{OUT}/carry_panel.parquet')
    d = R.panel()
    d = d.filter(~pl.col('product').is_in(list(C.FINANCIAL)))
    rows = []
    print('== Carry 实盘版（资金 1 亿、次日 VWAP、扣全部成本，2006 年起）==')
    for name, kw in [('截面 carry1m', dict(signal='carry', kind='xs')), ('截面 carry1-12', dict(signal='carry1_12', kind='xs')),
                     ('时序 carry1m（阈值 0）', dict(signal='carry', kind='ts'))]:
        W = weights(cp, d, **kw)
        r = R.run(d, capital=1e8, weights=W)
        s, c = R.summary(r)
        rows.append((name, r))
        print(f"{name:20s} {R.T.fmt(s)}  年化成本 {c:.2%}（调仓 {r['c_lin'].mean()*243:.2%}+冲击 {r['c_imp'].mean()*243:.2%}+换月 {r['c_roll'].mean()*243:.2%}）"
              f"  持仓 {r['n_pos'].mean():.0f}  杠杆 {r['gross_lev'].median():.1f}")
        for lo in (2010, 2016):
            q = r.filter(pl.col('trading_day').dt.year() >= lo)
            print(f'    {lo} 年起: {R.T.fmt(R.T.stats(q["ret"]))}')
        pl.DataFrame({'trading_day': r['trading_day'], 'ret': r['ret']}).write_parquet(f"{OUT}/real_{kw['signal']}_{kw['kind']}.parquet")

    print('\n== 从论文口径到实盘（截面 carry1-12）==')
    W = weights(cp, d, signal='carry1_12', kind='xs', adv_min=0)
    W1 = weights(cp, d, signal='carry1_12', kind='xs')
    d0 = d.with_columns(pl.lit(0.0).alias('base_cost'))
    ladder = [
        ('① 无摩擦，月末收盘成交，无门槛', d0, W, dict(exec_px='close', lag=0, capital=1e13, y=0, roll_cost=False, cap=1e9, adv_min=0)),
        ('② 改为次日 VWAP 成交', d0, W, dict(capital=1e13, y=0, roll_cost=False, cap=1e9, adv_min=0)),
        ('③ + 成交额门槛 1 亿/日', d0, W1, dict(capital=1e13, y=0, roll_cost=False, cap=1e9)),
        ('④ + 资金 1 亿、整手、容量上限 2%', d0, W1, dict(capital=1e8, y=0, roll_cost=False)),
        ('⑤ + 调仓手续费和半价差', d, W1, dict(capital=1e8, y=0, roll_cost=False)),
        ('⑥ + 换月成本', d, W1, dict(capital=1e8, y=0)),
        ('⑦ + 冲击成本（= 实盘版）', d, W1, dict(capital=1e8)),
    ]
    lad = []
    for name, dd, ww, kw in ladder:
        r = R.run(dd, weights=ww, **kw)
        s, c = R.summary(r)
        lad.append((name, s['sharpe'], s['ann_ret'], c))
        print(f'{name:28s} 夏普 {s["sharpe"]:5.2f}  年化收益 {s["ann_ret"]:6.2%}  年化成本 {c:5.2%}')
    pl.DataFrame(lad, schema=['step', 'sharpe', 'ann_ret', 'cost'], orient='row').write_parquet(f'{OUT}/real_ladder.parquet')

    print('\n== 敏感性（截面 carry1-12，其余同实盘版）==')
    sens = []
    for name, kw in [('资金 1000 万', {'capital': 1e7}), ('资金 10 亿', {'capital': 1e9}), ('冲击 Y=0', {'y': 0.0}), ('冲击 Y=1', {'y': 1.0}),
                     ('价差成本 ×2', {'ticks_mult': 2.0}), ('换月不计成本', {'roll_cost': False})]:
        r = R.run(d, weights=W1, **{'capital': 1e8, **kw})
        s, c = R.summary(r)
        sens.append((name, s['sharpe'], s['ann_ret'], c))
        print(f'{name:14s} 夏普 {s["sharpe"]:5.2f}  年化收益 {s["ann_ret"]:6.2%}  年化成本 {c:5.2%}')
    pl.DataFrame(sens, schema=['case', 'sharpe', 'ann_ret', 'cost'], orient='row').write_parquet(f'{OUT}/real_sensitivity.parquet')


if __name__ == '__main__':
    main()
