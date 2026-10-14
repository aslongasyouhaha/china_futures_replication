"""《Value and Momentum Everywhere》复现（商品期货部分）—— Asness, Moskowitz & Pedersen (2013, Journal of Finance)

论文方法（商品）：
  收益：最活跃合约的日超额收益复利成月度（这里用主力连续后复权）
  价值：ln(4.5–5.5 年前的平均现货价 ÷ 最近现货价)，即过去 5 年收益的相反数；现货价用未复权主力价格近似（论文用最近期合约价格）
  动量：过去 12 个月累计收益，剔除最近一个月（MOM2–12）
  组合：按信号分三等分 P1（低）、P2、P3（高），等权；P3−P1；排名加权因子 w = c·(rank − 平均 rank)，多空各 1 元（式 1）；
        50/50 组合 = 0.5·价值 + 0.5·动量
  报告：年化均值、t、波动、夏普；对等权商品篮子回归的 α 与 t；价值与动量的相关（对基准回归后的残差）
只能复现论文 8 个资产类别中的“商品期货”：中国股指期货只有 4 个、无账面价值；国债期货只有一个国家。
品种池同其他商品研究：主力 60 日平均成交 ≥ 1000 手、上市 ≥ 60 天；每月至少 10 个品种有信号。
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
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
import factors as FA  # noqa: E402

DATA = RAW_DATA
MIN_N = 10
PAPER = {'价值 P3−P1': (6.3, 1.61, 0.26), '价值 因子': (7.3, 1.92, 0.31), '动量 P3−P1': (12.4, 3.29, 0.53), '动量 因子': (11.5, 3.14, 0.51),
         '50/50 P3−P1': (9.4, 4.42, 0.71), '50/50 因子': (17.1, 4.78, 0.77)}


def nw_t(x):
    x = np.asarray(x, float)
    n = len(x)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xc = x - x.mean()
    s = xc @ xc / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (xc[l:] @ xc[:-l]) / n
    return x.mean() / np.sqrt(s / n)


def panel():
    me = FA.main_monthly()
    parts = []
    for f in glob.glob(f'{DATA}/*/*/daily_main.parquet'):
        parts.append(pl.read_parquet(f, columns=['trading_day', 'close']).with_columns(pl.lit(os.path.basename(os.path.dirname(f))).alias('product')))
    raw = (pl.concat(parts).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
             .group_by(['product', 'm']).agg(pl.col('close').sort_by('trading_day').last().alias('spot')))
    x = me.join(raw, on=['product', 'm']).sort(['product', 'm'])
    # 月份必须连续，才能用 shift 表示“k 个月前”
    x = x.with_columns(((pl.col('m').dt.year() * 12 + pl.col('m').dt.month()) - (pl.col('m').dt.year() * 12 + pl.col('m').dt.month()).shift(1).over('product')).alias('gap'))
    x = x.with_columns(pl.col('gap').fill_null(1).ne(1).cum_sum().over('product').alias('seg'))
    g = ['product', 'seg']
    x = x.with_columns(
        (pl.col('close_adj').shift(-1).over(g) / pl.col('close_adj') - 1).alias('ret_next'),
        (pl.col('close_adj').shift(1).over(g) / pl.col('close_adj').shift(12).over(g) - 1).alias('mom'),
        # 4.5–5.5 年前：第 54–66 个月前的平均价
        pl.mean_horizontal([pl.col('spot').shift(k).over(g) for k in range(54, 67)]).alias('spot_5y'))
    x = x.with_columns((pl.col('spot_5y') / pl.col('spot')).log().alias('value'))
    x = x.with_columns(pl.col('spot').shift(66).over(g).is_not_null().alias('has_value'),
                       pl.col('close_adj').shift(12).over(g).is_not_null().alias('has_mom'))
    uni = x.filter(~pl.col('product').is_in(list(FA.EXCLUDE)) & (pl.col('vol60') >= 1000) & (pl.col('n_obs') >= 60)
                   & pl.col('ret_next').is_not_null() & (pl.col('m') >= pl.date(2006, 1, 1)))
    return uni.select('product', 'm', 'ret_next', 'mom', 'value', 'has_value', 'has_mom')


def sorts(u, sig):
    rows = []
    for (m,), g in u.filter(pl.col(sig).is_not_null()).group_by(['m'], maintain_order=True):
        n = g.height
        if n < MIN_N:
            continue
        s = g[sig].to_numpy()
        r = g['ret_next'].to_numpy()
        o = np.argsort(s, kind='stable')
        parts = np.array_split(o, 3)
        rk = s.argsort().argsort() + 1.0
        w = rk - rk.mean()
        w = w / w[w > 0].sum()
        rows.append({'m': m, 'n': n, 'P1': r[parts[0]].mean(), 'P2': r[parts[1]].mean(), 'P3': r[parts[2]].mean(), 'F': float(w @ r)})
    d = pl.DataFrame(rows)
    return d.with_columns((pl.col('P3') - pl.col('P1')).alias('P3−P1'))


def stats(x, b):
    x = np.asarray(x, float)
    X = np.column_stack([np.ones(len(b)), b])
    co = np.linalg.lstsq(X, x, rcond=None)[0]
    e = x - X @ co
    XtX = np.linalg.inv(X.T @ X)
    L = int(np.floor(4 * (len(x) / 100) ** (2 / 9)))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    V = XtX @ S @ XtX
    return dict(均值=x.mean() * 1200, t=nw_t(x), 波动=x.std(ddof=1) * np.sqrt(12) * 100, 夏普=x.mean() / x.std(ddof=1) * np.sqrt(12),
                α=co[0] * 1200, tα=co[0] / np.sqrt(V[0, 0])), e


def main():
    os.makedirs(OUT, exist_ok=True)
    u = panel()
    u.write_parquet(os.path.join(OUT, 'panel.parquet'))
    bench = u.group_by('m').agg(pl.col('ret_next').mean().alias('B'))
    v = sorts(u, 'value').rename({c: f'V_{c}' for c in ('n', 'P1', 'P2', 'P3', 'F', 'P3−P1')})
    mo = sorts(u, 'mom').rename({c: f'M_{c}' for c in ('n', 'P1', 'P2', 'P3', 'F', 'P3−P1')})
    d = v.join(mo, on='m').join(bench, on='m').sort('m').with_columns(
        (0.5 * pl.col('V_P3−P1') + 0.5 * pl.col('M_P3−P1')).alias('C_P3−P1'), (0.5 * pl.col('V_F') + 0.5 * pl.col('M_F')).alias('C_F'))
    d.write_parquet(os.path.join(OUT, 'portfolios.parquet'))
    print(f"样本 {d['m'].min()} ~ {d['m'].max()}（形成月），{d.height} 个月；每月有价值信号的品种 {d['V_n'].min()}~{d['V_n'].max()}（中位数 {d['V_n'].median():.0f}）")
    B = d['B'].to_numpy()
    rows, res = [], {}
    for lab, col in (('价值 P1', 'V_P1'), ('价值 P2', 'V_P2'), ('价值 P3', 'V_P3'), ('价值 P3−P1', 'V_P3−P1'), ('价值 因子', 'V_F'),
                     ('动量 P1', 'M_P1'), ('动量 P2', 'M_P2'), ('动量 P3', 'M_P3'), ('动量 P3−P1', 'M_P3−P1'), ('动量 因子', 'M_F'),
                     ('50/50 P3−P1', 'C_P3−P1'), ('50/50 因子', 'C_F'), ('等权商品篮子', 'B')):
        s, e = stats(d[col].to_numpy(), B)
        res[col] = e
        pp = PAPER.get(lab)
        rows.append({'组合': lab, **s, '论文均值': pp[0] if pp else None, '论文t': pp[1] if pp else None, '论文夏普': pp[2] if pp else None})
    t = pl.DataFrame(rows)
    t.write_parquet(os.path.join(OUT, 'table1.parquet'))
    pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(200)
    print(t.with_columns(pl.exclude('组合').round(2)))
    c1 = np.corrcoef(res['V_P3−P1'], res['M_P3−P1'])[0, 1]
    c2 = np.corrcoef(res['V_F'], res['M_F'])[0, 1]
    print(f'价值与动量的相关（对基准回归后的残差）：P3−P1 {c1:.2f}，因子 {c2:.2f}（论文 −0.39、−0.46）')
    sub = []
    for lo, hi in ((2010, 2015), (2016, 2026)):
        q = d.filter(pl.col('m').dt.year().is_between(lo, hi))
        for lab, col in (('价值 因子', 'V_F'), ('动量 因子', 'M_F'), ('50/50 因子', 'C_F')):
            x = q[col].to_numpy()
            sub.append({'时段': f'{lo}–{hi}', '组合': lab, '均值': x.mean() * 1200, 't': nw_t(x), '夏普': x.mean() / x.std() * np.sqrt(12), '月数': len(x)})
    sp = pl.DataFrame(sub)
    sp.write_parquet(os.path.join(OUT, 'subperiods.parquet'))
    print(sp.with_columns(pl.exclude('时段', '组合', '月数').round(2)))
    pl.DataFrame({'口径': ['P3−P1', '因子'], '相关': [c1, c2]}).write_parquet(os.path.join(OUT, 'correlation.parquet'))
    carry_link(u, d)


def carry_link(u, d):
    """补充（非论文内容）：与 carry1-12（过去 12 个月平均年化 carry，按排名加权，同一品种池）的关系。"""
    cp = pl.read_parquet(FA.CARRY_PANEL).sort(['product', 'trading_day']).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
    cp = cp.with_columns(pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('carry12')).select('product', 'm', 'carry12')
    x = u.join(cp, on=['product', 'm'], how='left')
    car = sorts(x.rename({'carry12': 'c12'}), 'c12').select('m', pl.col('F').alias('K_F'))
    j = d.join(car, on='m').sort('m')
    rows = []
    for lab, col in (('动量 因子', 'M_F'), ('价值 因子', 'V_F'), ('50/50 因子', 'C_F')):
        y = j[col].to_numpy()
        X = np.column_stack([np.ones(len(y)), j['B'].to_numpy(), j['K_F'].to_numpy()])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        XtX = np.linalg.inv(X.T @ X)
        L = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for l in range(1, L + 1):
            G = Xe[l:].T @ Xe[:-l]
            S += (1 - l / (L + 1)) * (G + G.T)
        V = XtX @ S @ XtX
        rows.append({'组合': lab, '原始年化%': y.mean() * 1200, '与 carry 相关': float(np.corrcoef(y, j['K_F'])[0, 1]),
                     '控制 carry 后 α%': b[0] * 1200, 't': b[0] / np.sqrt(V[0, 0]), 'carry β': b[2]})
    k = j['K_F'].to_numpy()
    rows.append({'组合': 'carry1-12 因子（参照）', '原始年化%': k.mean() * 1200, '与 carry 相关': 1.0, '控制 carry 后 α%': None, 't': nw_t(k), 'carry β': None})
    t = pl.DataFrame(rows)
    t.write_parquet(os.path.join(OUT, 'carry_link.parquet'))
    print('\n== 补充：与 carry1-12 的关系（同一品种池、同一月份）==')
    print(t.with_columns(pl.exclude('组合').round(2)))


if __name__ == '__main__':
    main()
