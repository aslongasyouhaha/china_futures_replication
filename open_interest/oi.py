"""《What Does Futures Market Interest Tell Us about the Macroeconomy and Asset Prices?》复现（商品部分为主）
—— Hong & Yogo (2012, Journal of Financial Economics)

论文方法（商品）：
  收益：每个商品取剩余期限 1–12 个月的所有合约（剔除 1 个月内到期），月度收益等权；商品在板块内等权；4 个板块等权
  商品市场持仓：持仓金额 = 价格 × 持仓量，板块内加总，月度增长率，4 个板块等权，再取 12 个月几何平均
  其他预测变量：短期利率、期限利差、经济活动指数（CFNAI）、商品基差（中位数）、套保者多空失衡（CFTC）、过去 12 个月商品收益（几何平均）
  回归：下月商品收益对标准化预测变量回归，系数 = 每 1 个标准差对应的月收益变化（%）
中国对应与差异：
  板块：农产品（含鸡蛋、生猪，畜牧品种太少）、有色贵金属、黑色、能源化工
  持仓金额 = 结算价 × 持仓量 × 合约乘数；为避免新品种上市造成的跳升，板块增长只用前后两个月都存在的品种
  基差：(F_T / F_近)^(1/(T−近)) − 1，以最近月合约代替现货
  合约须月初持仓 ≥ 100 手（避免冷门合约的陈旧结算价）
  无利率、经济活动指数、CFTC 套保者持仓数据，这些对照变量省略
另：论文脚注的替代口径（只用合约张数、不乘价格）；股指（IF）与国债（T）期货的持仓增长对各自收益的预测。
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
DATA = RAW_DATA
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
import factors as FA  # noqa: E402

MULT = os.path.join(HERE, '..', 'common', 'output', 'multipliers.parquet')
BLACK = {'RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM', 'WR', 'ZC'}
MIN_OI = 100


def sector(p):
    if p in BLACK:
        return '黑色'
    if p in FA.METAL:
        return '有色贵金属'
    if p in FA.AGRI:
        return '农产品'
    return '能源化工'


def nw(y, X, L=12):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX @ S @ XtX))
    return b, b / se, 1 - e.var() / y.var()


def month_end_contracts(f, p):
    d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'open_interest'])
    d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'), pl.col('trading_day').dt.truncate('1mo').alias('m'))
    me = d.group_by('m').agg(pl.col('trading_day').max().alias('me'))
    d = d.join(me, on='m').filter((pl.col('trading_day') == pl.col('me')) & (pl.col('px') > 0))
    return d.with_columns(pl.col('symbol').map_elements(FA.C.month_index, return_dtype=pl.Int64).alias('cm'),
                          (pl.col('m').dt.year() * 12 + pl.col('m').dt.month() - 1).alias('now'), pl.lit(p).alias('product')
                          ).select('product', 'symbol', 'm', 'px', 'open_interest', 'cm', 'now')


def commodity_panel():
    mult = pl.read_parquet(MULT).select('product', pl.col('year').cast(pl.Int32), 'mult')
    rets, ois, basis = [], [], []
    for f in glob.glob(f'{DATA}/*/*/daily_contracts.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in FA.EXCLUDE:
            continue
        c = month_end_contracts(f, p).sort(['symbol', 'm'])
        c = c.with_columns(pl.col('m').dt.year().cast(pl.Int32).alias('year')).join(mult, on=['product', 'year'], how='left').with_columns(
            pl.col('mult').forward_fill().backward_fill())
        # 合约月度收益：上月末 → 本月末，同一合约；上月末剩余期限 2–12 个月、持仓 ≥ 100
        c = c.with_columns(pl.col('px').shift(1).over('symbol').alias('px_prev'), pl.col('open_interest').shift(1).over('symbol').alias('oi_prev'),
                           pl.col('now').shift(1).over('symbol').alias('now_prev'))
        r = c.filter((pl.col('now_prev') == pl.col('now') - 1) & (pl.col('cm') - pl.col('now_prev') >= 2) & (pl.col('cm') - pl.col('now_prev') <= 12)
                     & (pl.col('oi_prev') >= MIN_OI))
        rets.append(r.group_by(['product', 'm']).agg((pl.col('px') / pl.col('px_prev') - 1).mean().alias('ret'), pl.len().alias('k')))
        ois.append(c.group_by(['product', 'm']).agg((pl.col('px') * pl.col('open_interest') * pl.col('mult')).sum().alias('doi'),
                                                    pl.col('open_interest').sum().alias('noi')))
        # 基差：以最近月合约代替现货
        live = c.filter((pl.col('cm') >= pl.col('now')) & (pl.col('open_interest') >= MIN_OI))
        near = live.sort('cm').group_by(['product', 'm']).agg(pl.col('px').first().alias('s'), pl.col('cm').first().alias('cm0'))
        b = live.join(near, on=['product', 'm']).filter(pl.col('cm') > pl.col('cm0'))
        b = b.with_columns(((pl.col('px') / pl.col('s')) ** (1 / (pl.col('cm') - pl.col('cm0'))) - 1).alias('basis'))
        basis.append(b.group_by(['product', 'm']).agg(pl.col('basis').median()))
    x = pl.concat(ois).join(pl.concat(rets), on=['product', 'm'], how='left').join(pl.concat(basis), on=['product', 'm'], how='left')
    return x.with_columns(pl.col('product').map_elements(sector, return_dtype=pl.Utf8).alias('sector')).sort(['product', 'm'])


def aggregate(x):
    # 板块持仓增长：只用前后两个月都存在的品种
    x = x.with_columns(pl.col('doi').shift(1).over('product').alias('doi_prev'), pl.col('noi').shift(1).over('product').alias('noi_prev'),
                       pl.col('m').shift(1).over('product').alias('m_prev'))
    g = x.filter(pl.col('m_prev') == pl.col('m').dt.offset_by('-1mo')).group_by(['sector', 'm']).agg(
        (pl.col('doi').sum() / pl.col('doi_prev').sum() - 1).alias('g_doi'), (pl.col('noi').sum() / pl.col('noi_prev').sum() - 1).alias('g_noi'))
    sr = x.drop_nulls('ret').group_by(['sector', 'm']).agg(pl.col('ret').mean(), pl.col('basis').median(), pl.len().alias('n'))
    s = g.join(sr, on=['sector', 'm'], how='outer_coalesce')
    agg = s.group_by('m').agg(pl.col('g_doi').mean(), pl.col('g_noi').mean(), pl.col('ret').mean(), pl.col('basis').mean(),
                              pl.col('sector').n_unique().alias('n_sector'), pl.col('n').sum().alias('n_comm')).sort('m')
    agg = agg.filter(pl.col('n_sector') == 4)
    lg = lambda c: ((1 + pl.col(c)).log().rolling_mean(12, min_periods=12).exp() - 1)
    agg = agg.with_columns(lg('g_doi').alias('OI12'), lg('g_noi').alias('OI12_n'), lg('ret').alias('R12'), pl.col('ret').shift(-1).alias('ret_next'))
    return s, agg


def regressions(agg, ycol='ret_next', label='商品'):
    d = agg.drop_nulls(['OI12', 'R12', 'basis', ycol])
    z = lambda c: ((d[c] - d[c].mean()) / d[c].std()).to_numpy()
    y = d[ycol].to_numpy() * 100
    specs = [('(1) 基差', ['basis']), ('(2) 基差 + 持仓增长', ['basis', 'OI12']), ('(3) 基差 + 过去收益', ['basis', 'R12']),
             ('(4) 基差 + 持仓增长 + 过去收益', ['basis', 'OI12', 'R12']), ('(5) 持仓增长（只用合约张数）', ['OI12_n']),
             ('(6) 持仓张数增长 + 过去收益', ['OI12_n', 'R12'])]
    rows = []
    for name, cols in specs:
        b, t, r2 = nw(y, np.column_stack([z(c) for c in cols]))
        row = {'资产': label, '回归': name, 'R2%': r2 * 100, '月数': len(y)}
        for c, bb, tt in zip(cols, b[1:], t[1:]):
            row[c] = f'{bb:+.2f}（{tt:.2f}）'
        rows.append(row)
    return rows, d


def fin_markets():
    """股指（IF）与国债（T）期货：自身持仓增长（合约张数，12 个月几何平均）预测下月收益；商品持仓增长预测国债收益。"""
    out = {}
    for ex, p in (('CFFEX', 'IF'), ('CFFEX', 'T')):
        c = month_end_contracts(f'{DATA}/{ex}/{p}/daily_contracts.parquet', p).sort(['symbol', 'm'])
        n = c.group_by('m').agg(pl.col('open_interest').sum().alias('noi')).sort('m')
        main = pl.read_parquet(f'{DATA}/{ex}/{p}/daily_main.parquet', columns=['trading_day', 'close_adj']).with_columns(
            pl.col('trading_day').dt.truncate('1mo').alias('m')).group_by('m').agg(pl.col('close_adj').sort_by('trading_day').last()).sort('m')
        x = n.join(main, on='m').sort('m').with_columns((pl.col('noi') / pl.col('noi').shift(1) - 1).alias('g'),
                                                       (pl.col('close_adj') / pl.col('close_adj').shift(1) - 1).alias('r'))
        x = x.with_columns(((1 + pl.col('g')).log().rolling_mean(12, min_periods=12).exp() - 1).alias(f'{p}_OI12'),
                           ((1 + pl.col('r')).log().rolling_mean(12, min_periods=12).exp() - 1).alias(f'{p}_R12'),
                           pl.col('r').shift(-1).alias(f'{p}_ret_next'))
        out[p] = x.select('m', f'{p}_OI12', f'{p}_R12', f'{p}_ret_next')
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    x = commodity_panel()
    x.write_parquet(os.path.join(OUT, 'panel.parquet'))
    s, agg = aggregate(x)
    s.write_parquet(os.path.join(OUT, 'sectors.parquet'))
    agg.write_parquet(os.path.join(OUT, 'aggregate.parquet'))
    d = agg.drop_nulls(['OI12', 'R12', 'basis', 'ret'])
    print(f"商品组合：{d['m'].min()} ~ {d['m'].max()}，{d.height} 个月；月均收益 {d['ret'].mean()*100:.2f}%，月波动 {d['ret'].std()*100:.2f}%，"
          f"一阶自相关 {np.corrcoef(d['ret'][1:], d['ret'][:-1])[0,1]:.2f}")
    print(f"持仓增长（12 个月几何平均）：月均 {d['OI12'].mean()*100:.2f}%，标准差 {d['OI12'].std()*100:.2f}%，一阶自相关 {np.corrcoef(d['OI12'][1:], d['OI12'][:-1])[0,1]:.2f}；"
          f"与过去 12 个月收益相关 {np.corrcoef(d['OI12'], d['R12'])[0,1]:.2f}，与基差相关 {np.corrcoef(d['OI12'], d['basis'])[0,1]:.2f}")
    rows, used = regressions(agg)
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(12)
    t = pl.DataFrame(rows)
    t.write_parquet(os.path.join(OUT, 'table6.parquet'))
    print('\n== 下月商品收益（%）对标准化预测变量回归（NW t，滞后 12）==')
    print(t)
    # 分段
    sub = []
    for lo, hi in ((2010, 2015), (2016, 2026)):
        q = agg.filter(pl.col('m').dt.year().is_between(lo, hi))
        r, _ = regressions(q, label=f'{lo}–{hi}')
        sub += [rr for rr in r if rr['回归'].startswith(('(2)', '(4)'))]
    print(pl.DataFrame(sub))
    pl.DataFrame(sub).write_parquet(os.path.join(OUT, 'subperiods.parquet'))
    # 金融期货
    fm = fin_markets()
    fin_rows = []
    j = agg.select('m', 'OI12').join(fm['T'], on='m').join(fm['IF'], on='m', how='left')
    for lab, y, xs in (('国债期货收益 ~ 商品持仓增长', 'T_ret_next', ['OI12']), ('国债期货收益 ~ 国债持仓增长', 'T_ret_next', ['T_OI12']),
                       ('国债期货收益 ~ 商品持仓 + 国债持仓', 'T_ret_next', ['OI12', 'T_OI12'])):
        q = j.drop_nulls([y] + xs)
        b, tt, r2 = nw(q[y].to_numpy() * 100, np.column_stack([((q[c] - q[c].mean()) / q[c].std()).to_numpy() for c in xs]))
        fin_rows.append({'回归': lab, '系数（t）': '；'.join(f'{c} {bb:+.2f}（{t_:.2f}）' for c, bb, t_ in zip(xs, b[1:], tt[1:])), 'R2%': r2 * 100, '月数': q.height})
    q = fm['IF'].join(agg.select('m', 'OI12'), on='m', how='left').drop_nulls(['IF_ret_next', 'IF_OI12'])
    for lab, xs in (('股指期货收益 ~ 股指持仓增长', ['IF_OI12']), ('股指期货收益 ~ 股指持仓增长 + 过去收益', ['IF_OI12', 'IF_R12'])):
        qq = q.drop_nulls(xs)
        b, tt, r2 = nw(qq['IF_ret_next'].to_numpy() * 100, np.column_stack([((qq[c] - qq[c].mean()) / qq[c].std()).to_numpy() for c in xs]))
        fin_rows.append({'回归': lab, '系数（t）': '；'.join(f'{c} {bb:+.2f}（{t_:.2f}）' for c, bb, t_ in zip(xs, b[1:], tt[1:])), 'R2%': r2 * 100, '月数': qq.height})
    ft = pl.DataFrame(fin_rows)
    ft.write_parquet(os.path.join(OUT, 'financial.parquet'))
    print(ft)


if __name__ == '__main__':
    main()
