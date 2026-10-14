"""商品三因子的经济解释（论文第 6 节），用中国可得的代理变量：
  · 股市波动率：论文用 G20 股市日收益绝对值的月均值；这里用沪深 300 股指期货（IF 主力，2010-04 起）日收益绝对值的月均值
  · 商品波动率：当月品种池内所有商品日收益绝对值的均值（同论文 DVOL_commodity）
  · 投机度：论文用 CFTC 非商业持仓占比；中国没有对应数据，用 成交量 / 持仓量（主力合约，月内日均），等权平均
  Δ 表示当月值减上月值。因子收益属于持有月 h（= 形成月 m 的下一个月），经济变量也取 h 月的变化。
另：多空腿成员重叠（论文 6.5）。
"""
import glob
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st

import pricing as PR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = PR.HERE
OUT = PR.OUT
DATA = RAW_DATA


def daily_main(products):
    parts = []
    for f in glob.glob(f'{DATA}/*/*/daily_main.parquet'):
        p = os.path.basename(os.path.dirname(f))
        if p in products:
            parts.append(pl.read_parquet(f, columns=['trading_day', 'close_adj', 'volume', 'open_interest'])
                           .with_columns(pl.lit(p).alias('product')))
    d = pl.concat(parts).sort(['product', 'trading_day'])
    return d.with_columns((pl.col('close_adj') / pl.col('close_adj').shift(1).over('product') - 1).alias('ret'),
                          pl.col('trading_day').dt.truncate('1mo').alias('h'))


def econ_series(panel):
    """按持有月 h 的经济变量（及其变化）。"""
    uni = panel.with_columns(pl.col('m').dt.offset_by('1mo').alias('h')).select('product', 'h')   # 当月持有的品种池
    d = daily_main(set(panel['product'].unique().to_list()) | {'IF'})
    com = d.join(uni, on=['product', 'h'])
    vol_c = com.group_by('h').agg(pl.col('ret').abs().mean().alias('VOL_com'))
    spec_i = (com.filter(pl.col('open_interest') > 0)
                 .group_by(['product', 'h']).agg((pl.col('volume') / pl.col('open_interest')).mean().alias('spec')))
    spec_i = spec_i.sort(['product', 'h']).with_columns(
        (pl.col('spec').log() - pl.col('spec').log().shift(1).over('product')).alias('dspec_i'),
        (pl.col('h').shift(1).over('product') == pl.col('h').dt.offset_by('-1mo')).alias('cont'))
    spec_i = spec_i.with_columns(pl.when(pl.col('cont')).then(pl.col('dspec_i')).alias('dspec_i'))
    spec = spec_i.group_by('h').agg(pl.col('dspec_i').mean().alias('DSPEC'), pl.col('spec').mean().alias('SPEC'))
    eq = d.filter(pl.col('product') == 'IF').group_by('h').agg(pl.col('ret').abs().mean().alias('VOL_eq'))
    e = vol_c.join(spec, on='h', how='left').join(eq, on='h', how='left').sort('h')
    e = e.with_columns(pl.col('VOL_com').diff().alias('DVOL_com'), pl.col('VOL_eq').diff().alias('DVOL_eq'))
    return e, spec_i


def reg(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (PR.nw_cov(X * e[:, None]) * len(y)) @ XtX
    return b[1], b[1] / np.sqrt(V[1, 1]), 2 * st.norm.sf(abs(b[1] / np.sqrt(V[1, 1]))), 1 - e.var() / y.var()


def main(v):
    fac = pl.read_parquet(f'{OUT}/factors_{v}.parquet').with_columns(pl.col('m').dt.offset_by('1mo').alias('h'))
    port = pl.read_parquet(f'{OUT}/test_portfolios_{v}.parquet').with_columns(pl.col('m').dt.offset_by('1mo').alias('h')).drop('m')
    panel = pl.read_parquet(f'{OUT}/panel_{v}.parquet')
    e, spec_i = econ_series(panel)
    e.write_parquet(f'{OUT}/econ_{v}.parquet')
    d = fac.join(port, on='h').join(e, on='h', how='left').sort('h')
    rows = []
    print(f'\n######## {v} ########')
    print('== 时间序列回归：因子（及多空腿）对经济变量的变化（NW t）==')
    for var, label in (('DVOL_eq', 'Δ股市波动率（IF，2010 起）'), ('DVOL_com', 'Δ商品波动率'), ('DSPEC', 'Δ投机度（成交/持仓）')):
        q = d.drop_nulls([var])
        for f in ('CARRY', 'CARRY_long', 'CARRY_short', 'CMOM', 'CMOM_long', 'CMOM_short', 'AVG'):
            b, t, p, r2 = reg(q[f].to_numpy(), q[var].to_numpy() / q[var].std())
            rows.append({'变量': label, '因子': f, '系数（每 1 标准差，月%）': b * 100, 't': t, 'p': p, 'R2': r2, 'T': q.height})
    ts = pl.DataFrame(rows)
    ts.write_parquet(f'{OUT}/econ_ts_{v}.parquet')
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
    print(ts.with_columns(pl.col('系数（每 1 标准差，月%）', 't', 'p', 'R2').round(3)))

    print('\n== 截面定价：AVG + 一个经济变量，分别定价 carry 组合（P1–P4）和动量组合（Q1–Q5）==')
    fm = []
    for var, label in (('DVOL_eq', 'Δ股市波动率'), ('DVOL_com', 'Δ商品波动率'), ('DSPEC', 'Δ投机度')):
        for tname, cols in (('carry 组合', ['P1', 'P2', 'P3', 'P4']), ('动量组合', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])):
            q = d.drop_nulls([var] + cols)
            r = PR.fama_macbeth(q.select(cols).to_numpy(), q.select('AVG', var).to_numpy())
            fm.append({'经济变量': label, '测试资产': tname, 'λ': r['lam'][1], 'p(NW)': r['p_nw'][1], 'χ2(NW)': r['chi_nw'],
                       'p_χ2(NW)': r['pchi_nw'], 'R2_OLS': r['r2_ols'], 'T': q.height})
            print(f"{label:8s} 定价 {tname:6s} λ={r['lam'][1]:+.4f}（p={r['p_nw'][1]:.2f}）  定价误差 χ² p={r['pchi_nw']:.2f}  R²(OLS) {r['r2_ols']:.0%}  T={q.height}")
    pl.DataFrame(fm).write_parquet(f'{OUT}/econ_fm_{v}.parquet')

    # 个品种层面：赢家/输家的投机度变化（论文：赢家投机持仓上升）
    mem = pl.read_parquet(f'{OUT}/members_{v}.parquet').with_columns(pl.col('m').dt.offset_by('1mo').alias('h'))
    x = mem.filter(pl.col('leg').is_in(['mom_long', 'mom_short'])).join(spec_i.select('product', 'h', 'dspec_i'), on=['product', 'h'])
    print('\n== 动量赢家 / 输家在持有月的投机度变化（对数变化，月均）==')
    for (leg,), g in x.group_by(['leg']):
        dv = g['dspec_i'].drop_nulls().to_numpy()
        print(f"{leg}: 均值 {dv.mean():+.4f}，t = {dv.mean()/dv.std()*np.sqrt(len(dv)):.2f}，n = {len(dv)}")
    # 多空腿成员重叠（论文 6.5）
    ov = []
    for (m,), g in mem.group_by(['m']):
        s = {l: set(g.filter(pl.col('leg') == l)['product']) for l in ('carry_long', 'carry_short', 'mom_long', 'mom_short')}
        ov.append(len(s['carry_long'] & s['mom_long']) + len(s['carry_short'] & s['mom_short']))
    ov = np.array(ov)
    print(f'\n== carry 与动量多空腿的重叠（同向重叠品种数，最多 10）：中位数 {np.median(ov):.0f}，Q3 {np.percentile(ov, 75):.0f}，'
          f'均值 {ov.mean():.1f}（论文 Q3 = 1）')
    cnt = mem.group_by(['leg', 'product']).agg(pl.len().alias('n')).sort('n', descending=True)
    cnt.write_parquet(f'{OUT}/member_counts_{v}.parquet')
    for leg in ('carry_long', 'carry_short', 'mom_long', 'mom_short'):
        top = cnt.filter(pl.col('leg') == leg).head(6)
        print(f"{leg:12s} 最常入选：" + '，'.join(f"{p}({n})" for _, p, n in top.iter_rows()))
    pl.DataFrame({'overlap': ov}).write_parquet(f'{OUT}/overlap_{v}.parquet')

    # 商品下跌月份（论文用 NBER 衰退期；中国没有官方衰退划分，用 AVG 最差 20% 的月份代替）
    print('\n== 按商品市场状态分组（AVG 五分位）的月均收益（%）==')
    st_ = d.with_columns(pl.col('AVG').qcut(5, labels=['最差', '2', '3', '4', '最好']).alias('状态'))
    g = st_.group_by('状态').agg(*[(pl.col(c).mean() * 100).round(2) for c in ('AVG', 'CARRY', 'CARRY_long', 'CARRY_short', 'CMOM', 'CMOM_long', 'CMOM_short')]).sort('状态')
    g.write_parquet(f'{OUT}/by_state_{v}.parquet')
    print(g)


if __name__ == '__main__':
    for v in (sys.argv[1:] or ['paper', 'main']):
        main(v)
