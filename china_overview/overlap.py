"""carry、动量、价值在同一框架下的重叠（本文计算，不对应单篇论文）。

统一框架：沿用 value_momentum（Asness-Moskowitz-Pedersen 复现）的品种池与组合构造——
  品种池：主力 60 日平均成交 ≥ 1000 手、上市 ≥ 60 天，不含中金所；2006 年起
  因子：每月末按信号排名加权（权重 ∝ 排名 − 平均排名，多空各 1 元），持有下月，主力后复权收益；每月至少 10 个品种
信号：
  CARRY12  过去 12 个月平均年化 carry（carry 复现的 carry1-12）
  CARRY1   当月年化 carry
  MOM      过去 12 个月收益、剔除最近 1 个月
  VAL      ln(4.5–5.5 年前平均价 ÷ 当前价)（同 AMP 复现：窗口内有数据即可，不要求满 5.5 年）
不计交易成本。
"""
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, f'{R}/value_momentum')
sys.path.insert(0, HERE)
import vme  # noqa: E402
from market import ols_nw  # noqa: E402

FAC = ['CARRY12', 'CARRY1', 'MOM', 'VAL']


def build():
    u = pl.read_parquet(f'{R}/value_momentum/output/panel.parquet')
    cp = pl.read_parquet(vme.FA.CARRY_PANEL).sort(['product', 'trading_day']).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
    cp = cp.with_columns(pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('c12')).select('product', 'm', pl.col('carry').alias('c1'), 'c12')
    x = u.join(cp, on=['product', 'm'], how='left')
    f = None
    for name, sig, data in (('CARRY12', 'c12', x), ('CARRY1', 'c1', x), ('MOM', 'mom', x), ('VAL', 'value', x)):
        s = vme.sorts(data, sig).select('m', pl.col('F').alias(name), pl.col('n').alias(f'n_{name}'))
        f = s if f is None else f.join(s, on='m', how='outer_coalesce')
    b = pl.read_parquet(f'{R}/value_momentum/output/portfolios.parquet').select('m', pl.col('B').alias('MKT'))
    return f.join(b, on='m', how='left').sort('m')


def stats(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    b, t, _, _ = ols_nw(r, np.zeros((len(r), 0)))
    return {'年化%': r.mean() * 1200, 't': float(t[0]), '夏普': r.mean() / r.std(ddof=1) * np.sqrt(12), '月数': len(r)}


def main():
    os.makedirs(OUT, exist_ok=True)
    f = build()
    f.write_parquet(os.path.join(OUT, 'overlap_factors.parquet'))
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    # 1. 各因子单独表现：自身全部可用月份 / 共同样本（价值可用起）/ 分段
    common = f.drop_nulls(FAC + ['MKT'])
    rows = []
    for k in FAC + ['MKT']:
        for lab, g in (('自身样本', f), ('共同样本', common), ('2010–2015', common.filter(pl.col('m').dt.year() <= 2015)), ('2016–2026', common.filter(pl.col('m').dt.year() >= 2016))):
            rows.append({'因子': k, '样本': lab, '起': str(g.filter(pl.col(k).is_not_null())['m'].min()), **stats(g[k].to_numpy())})
    S = pl.DataFrame(rows)
    S.write_parquet(os.path.join(OUT, 'overlap_stats.parquet'))
    print(S.with_columns(pl.col(pl.Float64).round(2)))
    # 2. 相关矩阵（共同样本及分段）
    for lab, g in (('共同样本', common), ('2010–2015', common.filter(pl.col('m').dt.year() <= 2015)), ('2016–2026', common.filter(pl.col('m').dt.year() >= 2016))):
        C = np.corrcoef(g.select(FAC + ['MKT']).to_numpy().T)
        print(lab, g.height)
        print(pl.DataFrame(C, schema=FAC + ['MKT']).with_columns(pl.Series('', FAC + ['MKT'])).with_columns(pl.col(pl.Float64).round(2)))
    # 3. 互相解释：每个因子对其余因子（+ 市场）回归
    specs = [('MOM', ['MKT', 'CARRY12']), ('MOM', ['MKT', 'CARRY12', 'VAL']), ('VAL', ['MKT', 'CARRY12']), ('VAL', ['MKT', 'CARRY12', 'MOM']),
             ('CARRY12', ['MKT', 'MOM']), ('CARRY12', ['MKT', 'MOM', 'VAL']), ('CARRY1', ['MKT', 'CARRY12', 'MOM', 'VAL'])]
    reg = []
    for lab, g in (('共同样本', common), ('2010–2015', common.filter(pl.col('m').dt.year() <= 2015)), ('2016–2026', common.filter(pl.col('m').dt.year() >= 2016))):
        for y, xs in specs:
            b, t, r2, T = ols_nw(g[y].to_numpy(), g.select(xs).to_numpy())
            reg.append({'样本': lab, 'y': y, 'x': '+'.join(xs), 'α': b[0] * 1200, 't_α': t[0], **{f'b_{x}': b[i + 1] for i, x in enumerate(xs)},
                        **{f't_{x}': t[i + 1] for i, x in enumerate(xs)}, 'R2': r2, 'N': T})
    G = pl.DataFrame(reg, infer_schema_length=None)
    G.write_parquet(os.path.join(OUT, 'overlap_spanning.parquet'))
    print(G.with_columns(pl.col(pl.Float64).round(2)))
    # 4. 组合：各因子按全样本波动缩放到相同波动后等权（只用于比较分散效果，事后缩放）
    sd = {k: common[k].std() for k in FAC}
    combos = {'CARRY12': ['CARRY12'], 'MOM': ['MOM'], 'VAL': ['VAL'], 'CARRY12+MOM': ['CARRY12', 'MOM'], 'CARRY12+VAL': ['CARRY12', 'VAL'],
              'MOM+VAL': ['MOM', 'VAL'], 'CARRY12+MOM+VAL': ['CARRY12', 'MOM', 'VAL']}
    crow = []
    for name, ks in combos.items():
        r = sum(common[k] / sd[k] for k in ks) / len(ks) * float(np.mean([sd[k] for k in FAC]))
        for lab, g in (('共同样本', None), ('2010–2015', 2015), ('2016–2026', 2016)):
            rr = r if g is None else (r.filter(common['m'].dt.year() <= 2015) if g == 2015 else r.filter(common['m'].dt.year() >= 2016))
            crow.append({'组合': name, '样本': lab, **stats(rr.to_numpy())})
    CB = pl.DataFrame(crow)
    CB.write_parquet(os.path.join(OUT, 'overlap_combos.parquet'))
    print(CB.with_columns(pl.col(pl.Float64).round(2)))



def md_tables():
    """论文格式的表，写入 tables/xs_overlap_*.md。"""
    sys.path.insert(0, HERE)
    from tables_xs import cell, table
    stars = '\*、\*\*、\*\*\* 分别表示 10%、5%、1% 显著。'
    f = pl.read_parquet(os.path.join(OUT, 'overlap_factors.parquet')).drop_nulls(FAC + ['MKT'])
    lab = {'CARRY12': 'carry1-12', 'CARRY1': '当月 carry', 'MOM': '动量', 'VAL': '价值', 'MKT': '等权商品篮子'}
    # 相关矩阵
    ks = FAC + ['MKT']
    rows = []
    for name, g in (('', f), ('2010–2015', f.filter(pl.col('m').dt.year() <= 2015)), ('2016–2026', f.filter(pl.col('m').dt.year() >= 2016))):
        C = np.corrcoef(g.select(ks).to_numpy().T)
        for i, a in enumerate(ks):
            rows.append(((f'{name}：' if name else '') + lab[a], [f'{C[i, j]:.2f}' if j >= i else '' for j in range(len(ks))]))
    t1 = table([lab[k] for k in ks], rows[:5], [], '月度因子收益的相关系数，2010-07 至 2026-08（194 个月）。本文计算。')
    t1s = table([lab[k] for k in ks], rows[5:], [], '分段的相关系数（2010–2015 年 66 个月，2016–2026 年 128 个月）。本文计算。')
    # 互相解释
    g = pl.read_parquet(os.path.join(OUT, 'overlap_spanning.parquet')).filter(pl.col('样本') == '共同样本')
    rs = list(g.iter_rows(named=True))

    def c(r, k):
        return cell(r.get(f'b_{k}'), r.get(f't_{k}')) if r.get(f'b_{k}') is not None else ''
    rows = [('α（年化 %）', [cell(r['α'], r['t_α'], '{:.2f}') for r in rs])] + [(lab[k], [c(r, k) for r in rs]) for k in ('MKT', 'CARRY12', 'MOM', 'VAL')]
    foot = [('样本数（月）', [str(r['N']) for r in rs]), ('R²', [f"{r['R2']:.3f}" for r in rs])]
    t2 = table([lab[r['y']] for r in rs], rows, foot,
               '被解释变量为各因子的月收益；α 为年化 %。括号内为 Newey-West t 值；' + stars + '2010-07 至 2026-08。本文计算。')
    sub = pl.read_parquet(os.path.join(OUT, 'overlap_spanning.parquet')).filter(pl.col('x').str.count_matches(r'\+') >= 2)
    rows = []
    for y, x in (('MOM', 'MKT+CARRY12+VAL'), ('VAL', 'MKT+CARRY12+MOM'), ('CARRY12', 'MKT+MOM+VAL')):
        cs = []
        for s in ('共同样本', '2010–2015', '2016–2026'):
            r = sub.filter((pl.col('y') == y) & (pl.col('x') == x) & (pl.col('样本') == s)).row(0, named=True)
            cs.append(cell(r['α'], r['t_α'], '{:.2f}'))
        rows.append((f'{lab[y]}：控制另外两个因子和篮子后的 α', cs))
    t3 = table(['2010–2026', '2010–2015', '2016–2026'], rows, [], 'α 为年化 %，括号内为 Newey-West t 值；' + stars + '本文计算。')
    # 组合
    cb = pl.read_parquet(os.path.join(OUT, 'overlap_combos.parquet'))
    names = {'CARRY12': 'carry1-12', 'MOM': '动量', 'VAL': '价值', 'CARRY12+MOM': 'carry + 动量', 'CARRY12+VAL': 'carry + 价值', 'MOM+VAL': '动量 + 价值',
             'CARRY12+MOM+VAL': 'carry + 动量 + 价值'}
    rows = []
    for k, n in names.items():
        cs = []
        for s in ('共同样本', '2010–2015', '2016–2026'):
            r = cb.filter((pl.col('组合') == k) & (pl.col('样本') == s)).row(0, named=True)
            cs += [cell(r['年化%'], r['t'], '{:.2f}'), f"{r['夏普']:.2f}"]
        rows.append((n, cs))
    t4 = table(['2010–2026<br>年化 %', '2010–2026<br>夏普', '2010–2015<br>年化 %', '2010–2015<br>夏普', '2016–2026<br>年化 %', '2016–2026<br>夏普'], rows, [],
               '各因子先按全样本波动缩放到相同波动，再等权相加；缩放是事后的，只用于比较分散效果，不是可执行的策略。不计成本。括号内为 Newey-West t 值；' + stars + '本文计算。')
    for name, s in (('xs_overlap_corr', t1 + '\n' + t1s), ('xs_overlap_span', t2), ('xs_overlap_span_sub', t3), ('xs_overlap_combo', t4)):
        with open(os.path.join(HERE, 'tables', f'{name}.md'), 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(f'==== {name}\n{s}')


if __name__ == '__main__':
    main()
    md_tables()
