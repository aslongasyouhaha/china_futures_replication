"""中国商品期货的“市场因子”：它是什么、能否择时（跨论文研究，不对应单篇论文）

1. 构造（月度，主力合约复权收益，简单收益）
   EW   全部商品品种等权（上市满 6 个月）
   OIW  按上月末持仓金额加权
   FIX  固定品种池：2006 年以前已上市的品种，等权（成分不随时间变化）
   FIX2 固定品种池二：2010 年以前上市、且 2021 年以来月末持仓金额中位数 ≥ 10 亿元的品种，等权，从 2010-01 起（成分不变，且都仍活跃）
   另与南华商品综合指数（NHCI）对照
2. 宏观对照（同期）：市场因子月收益对 ΔPMI、ΔPPI 同比、ΔM1 同比、沪深 300 月收益、USDCNH 月变化、原油（INE SC，2018 起）回归
3. 择时（预测下月）：整体基差、整体持仓金额增长、过去 12 个月 / 1 个月市场收益、ΔPMI、ΔPPI、ΔM1（公布滞后一月）、沪深 300 上月收益
   样本内：单变量预测回归（NW t）；样本外：扩展窗口（2010 年起），OOS R²（对历史均值）与 Clark-West；按预测符号做多 / 做空的择时收益
"""
import glob
import os
import sys

import numpy as np
import polars as pl
from scipy import stats as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'carry'))
sys.path.insert(0, os.path.join(HERE, '..', 'commodity_three_factor'))
sys.path.insert(0, os.path.join(HERE, '..', 'open_interest'))
import carry as CA  # noqa: E402
from oi import sector  # noqa: E402
from pricing import nw_cov  # noqa: E402

RAW = RAW_DATA
REF = f'{RAW}/参考数据'
SKIP = {'EC', 'LF', 'PPF', 'VF', 'FB'}
MULT = os.path.join(HERE, '..', 'common', 'output', 'multipliers.parquet')
OOS_START = __import__('datetime').date(2010, 1, 1)


def nwl(T):
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def ols_nw(y, X):
    y, X = np.asarray(y, float), np.asarray(X, float).reshape(len(y), -1)
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    y, X = y[ok], np.column_stack([np.ones(ok.sum()), X[ok]])
    T = len(y)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    Q = np.linalg.inv(X.T @ X / T)
    V = Q @ nw_cov(X * e[:, None], nwl(T)) @ Q / T
    return b, b / np.sqrt(np.diag(V)), 1 - e.var() / y.var(), T


# ---------------------------------------------------------------- 1. 市场因子
def products():
    mult = pl.read_parquet(MULT).select('product', 'year', 'mult')
    rows = []
    for f in glob.glob(f'{RAW}/*/*/daily_main.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in SKIP:
            continue
        d = pl.read_parquet(f, columns=['trading_day', 'close', 'close_adj']).sort('trading_day')
        oi = pl.read_parquet(f.replace('daily_main', 'daily_index'), columns=['trading_day', 'open_interest'])
        d = d.join(oi, on='trading_day', how='left').with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'), pl.lit(p).alias('product'),
                                                                   pl.col('trading_day').dt.year().cast(pl.Int64).alias('year'))
        d = d.join(mult, on=['product', 'year'], how='left')
        m = d.group_by('product', 'm').agg(pl.col('close_adj').last(), (pl.col('open_interest') * pl.col('close') * pl.col('mult').fill_null(1.0)).last().alias('oiv'))
        rows.append(m.sort('m'))
    x = pl.concat(rows).sort(['product', 'm'])
    ok = pl.col('m').shift(1).over('product') == pl.col('m').dt.offset_by('-1mo')
    x = x.with_columns(pl.when(ok).then(pl.col('close_adj') / pl.col('close_adj').shift(1).over('product') - 1).alias('r'),
                       pl.when(ok).then(pl.col('oiv').shift(1).over('product')).alias('oiv_prev'),
                       pl.col('m').min().over('product').alias('m0'),
                       pl.col('product').map_elements(sector, return_dtype=pl.Utf8).alias('sector'))
    age = (pl.col('m').dt.year() - pl.col('m0').dt.year()) * 12 + pl.col('m').dt.month() - pl.col('m0').dt.month()
    return x.with_columns(age.alias('age')).filter(pl.col('r').is_not_null() & (pl.col('age') >= 6) & (pl.col('r').abs() < 1))


def factors(x):
    fix = x.filter(pl.col('m0') < pl.date(2006, 1, 1))['product'].unique()
    ew = x.group_by('m').agg(pl.col('r').mean().alias('EW'), pl.len().alias('N'))
    oiw = x.filter(pl.col('oiv_prev') > 0).group_by('m').agg(((pl.col('r') * pl.col('oiv_prev')).sum() / pl.col('oiv_prev').sum()).alias('OIW'))
    fx = x.filter(pl.col('product').is_in(fix)).group_by('m').agg(pl.col('r').mean().alias('FIX'), pl.len().alias('N_FIX'))
    act = x.filter(pl.col('m') >= pl.date(2021, 1, 1)).group_by('product').agg(pl.col('oiv').median().alias('oiv_med'))
    fix2 = x.filter(pl.col('m0') < pl.date(2010, 1, 1)).select('product').unique().join(act.filter(pl.col('oiv_med') >= 1e9), on='product')['product']
    fx2 = x.filter(pl.col('product').is_in(fix2) & (pl.col('m') >= pl.date(2010, 1, 1))).group_by('m').agg(pl.col('r').mean().alias('FIX2'), pl.len().alias('N_FIX2'))
    sec = x.group_by('m', 'sector').agg(pl.col('r').mean()).pivot(values='r', index='m', columns='sector')
    nh = pl.read_parquet(f'{REF}/nanhua_index.parquet').filter(pl.col('ts_code') == 'NHCI.NH').with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
    nh = nh.sort('trading_day').group_by('m').agg(pl.col('close').last()).sort('m').with_columns((pl.col('close') / pl.col('close').shift(1) - 1).alias('NHCI'))
    f = ew.join(oiw, on='m', how='left').join(fx, on='m', how='left').join(fx2, on='m', how='left').join(sec, on='m', how='left').join(nh.select('m', 'NHCI'), on='m', how='left')
    return f.sort('m').filter(pl.col('m') >= pl.date(2005, 2, 1)), sorted(fix.to_list()), sorted(fix2.to_list())


# ---------------------------------------------------------------- 2. 宏观
def macro():
    mon = lambda c: pl.col(c).str.to_date('%Y%m').alias('m')   # noqa: E731
    pmi = pl.read_parquet(f'{REF}/cn_pmi.parquet').select(mon('month'), pl.col('PMI010000').cast(pl.Float64).alias('PMI'))
    ppi = pl.read_parquet(f'{REF}/cn_ppi.parquet').select(mon('month'), pl.col('ppi_yoy').cast(pl.Float64).alias('PPI'))
    mm = pl.read_parquet(f'{REF}/cn_money.parquet').select(mon('month'), pl.col('m1_yoy').cast(pl.Float64).alias('M1'))
    hs = pl.read_parquet(f'{REF}/hs300_index.parquet').with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m')).sort('trading_day')
    hs = hs.group_by('m').agg(pl.col('close').last()).sort('m').with_columns((pl.col('close') / pl.col('close').shift(1) - 1).alias('HS300'))
    fx = pl.read_parquet(f'{REF}/fx_usdcnh.parquet').with_columns(((pl.col('bid_close') + pl.col('ask_close')) / 2).alias('mid'), pl.col('trading_day').dt.truncate('1mo').alias('m'))
    fx = fx.sort('trading_day').group_by('m').agg(pl.col('mid').last()).sort('m').with_columns((pl.col('mid') / pl.col('mid').shift(1) - 1).alias('USDCNH'))
    sc = pl.read_parquet(f'{RAW}/INE/SC/daily_main.parquet', columns=['trading_day', 'close_adj']).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
    sc = sc.sort('trading_day').group_by('m').agg(pl.col('close_adj').last()).sort('m').with_columns((pl.col('close_adj') / pl.col('close_adj').shift(1) - 1).alias('OIL'))
    z = pmi.join(ppi, on='m', how='outer_coalesce').join(mm, on='m', how='outer_coalesce').sort('m')
    z = z.with_columns(pl.col('PMI').diff().alias('dPMI'), pl.col('PPI').diff().alias('dPPI'), pl.col('M1').diff().alias('dM1'))
    return z.join(hs.select('m', 'HS300'), on='m', how='left').join(fx.select('m', 'USDCNH'), on='m', how='left').join(sc.select('m', 'OIL'), on='m', how='left')


# ---------------------------------------------------------------- 3. 择时变量
def aggregate_state(x):
    """月末整体基差（主力 vs 下一个合约，年化，等权）与整体持仓金额 12 个月对数增长（等权）。"""
    c = pl.read_parquet(os.path.join(HERE, '..', 'basis_momentum', 'output', 'curve_liquid.parquet'))
    c = c.filter(pl.col('n').is_in([1, 2])).select('product', 'm', 'n', 'px', 'cm')
    b = c.filter(pl.col('n') == 1).join(c.filter(pl.col('n') == 2), on=['product', 'm'], suffix='2')
    b = b.filter((pl.col('cm2') > pl.col('cm')) & (pl.col('px') > 0) & (pl.col('px2') > 0))
    b = b.with_columns(((pl.col('px2').log() - pl.col('px').log()) / (pl.col('cm2') - pl.col('cm')) * 12).alias('basis'))
    B = b.filter(~pl.col('product').is_in(list(SKIP))).group_by('m').agg(pl.col('basis').clip(-1, 1).mean().alias('BASIS'))
    o = x.select('product', 'm', 'oiv').sort(['product', 'm'])
    o = o.with_columns(pl.when(pl.col('m').shift(12).over('product') == pl.col('m').dt.offset_by('-12mo'))
                       .then(pl.col('oiv').log() - pl.col('oiv').shift(12).over('product').log()).alias('g'))
    O = o.filter(pl.col('g').is_finite()).group_by('m').agg(pl.col('g').clip(-3, 3).mean().alias('OIG'))
    return B.join(O, on='m', how='outer_coalesce').sort('m')


def timing(f, mac, state, col):
    """预测 col 的下月收益。所有预测变量都是月末 t 可知（宏观数据滞后一月）。"""
    d = f.select('m', pl.col(col).alias('y')).sort('m')
    d = d.with_columns((pl.col('y') + 1).log().rolling_sum(12).alias('TS12'), pl.col('y').alias('TS1'))
    d = d.join(state, on='m', how='left').join(mac.select('m', 'dPMI', 'dPPI', 'dM1', 'HS300'), on='m', how='left').sort('m')
    d = d.with_columns(pl.col('dPMI').shift(1), pl.col('dPPI').shift(1), pl.col('dM1').shift(1))   # 统计月 t 的数据在 t+1 才公布
    d = d.with_columns(pl.col('y').shift(-1).alias('y_next'))
    preds = ['BASIS', 'OIG', 'TS12', 'TS1', 'dPMI', 'dPPI', 'dM1', 'HS300']
    rows = []
    ms = d['m'].to_list()
    for p in preds:
        y, xv = d['y_next'].to_numpy().astype(float), d[p].to_numpy().astype(float)
        b, t, r2, T = ols_nw(y, xv)
        # 样本外：扩展窗口，至少 36 个月
        fc, hm, real, sig = [], [], [], []
        for i, m in enumerate(ms):
            if m < OOS_START or np.isnan(y[i]) or np.isnan(xv[i]):
                continue
            ok = ~np.isnan(y[:i]) & ~np.isnan(xv[:i])
            if ok.sum() < 36:
                continue
            X = np.column_stack([np.ones(ok.sum()), xv[:i][ok]])
            bb = np.linalg.lstsq(X, y[:i][ok], rcond=None)[0]
            fc.append(bb[0] + bb[1] * xv[i])
            hm.append(y[:i][ok].mean())
            real.append(y[i])
            sig.append(np.sign(bb[1] * (xv[i] - xv[:i][ok].mean())))   # 预测高于均值 → 做多
        fc, hm, real, sig = map(np.array, (fc, hm, real, sig))
        oos = 1 - ((real - fc) ** 2).sum() / ((real - hm) ** 2).sum()
        cw = (real - hm) ** 2 - ((real - fc) ** 2 - (hm - fc) ** 2)
        cwt = cw.mean() / np.sqrt(nw_cov(cw, nwl(len(cw)))[0, 0] / len(cw))
        tr = sig * real
        rows.append({'因子': col, '预测变量': p, 'β': float(b[1]), 't': float(t[1]), 'R2%': r2 * 100, '月数': T, 'OOS月数': len(real),
                     'OOS_R2%': oos * 100, 'CW_t': float(cwt), 'CW_p': float(st.norm.sf(cwt)),
                     '择时年化%': tr.mean() * 1200, '择时夏普': tr.mean() / tr.std(ddof=1) * np.sqrt(12), '同期做多年化%': real.mean() * 1200})
    return pl.DataFrame(rows), d


def main():
    os.makedirs(OUT, exist_ok=True)
    x = products()
    f, fix, fix2 = factors(x)
    mac = macro()
    state = aggregate_state(x)
    f.write_parquet(os.path.join(OUT, 'factors.parquet'))
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    print('固定池', len(fix), fix)
    print('固定池二', len(fix2), fix2)
    # 1. 描述
    cols = ['EW', 'OIW', 'FIX', 'FIX2', 'NHCI', '农产品', '有色贵金属', '黑色', '能源化工']
    desc = []
    for c in cols:
        r = f[c].drop_nulls().to_numpy()
        b, t, _, _ = ols_nw(r, np.zeros((len(r), 0)))
        desc.append({'序列': c, '起': str(f.filter(pl.col(c).is_not_null())['m'].min()), '月数': len(r), '年化%': r.mean() * 1200, 't': float(t[0]),
                     '年化波动%': r.std(ddof=1) * np.sqrt(12) * 100, '与EW相关': float(np.corrcoef(*f.select(pl.col('EW').alias('a'), pl.col(c).alias('b')).drop_nulls().to_numpy().T)[0, 1])})
    desc = pl.DataFrame(desc)
    desc.write_parquet(os.path.join(OUT, 'desc.parquet'))
    print(desc)
    # 分段
    seg = []
    for lab, lo, hi in [('2005–2009', 2005, 2009), ('2010–2015', 2010, 2015), ('2016–2020', 2016, 2020), ('2021–2026', 2021, 2026)]:
        g = f.filter((pl.col('m').dt.year() >= lo) & (pl.col('m').dt.year() <= hi))
        seg.append({'时期': lab, **{c: (g[c].mean() or np.nan) * 1200 for c in ('EW', 'OIW', 'FIX', 'FIX2', 'NHCI')}, 'EW波动%': g['EW'].std() * np.sqrt(12) * 100, '品种数': g['N'].mean()})
    seg = pl.DataFrame(seg)
    seg.write_parquet(os.path.join(OUT, 'segments.parquet'))
    print(seg)
    # 2. 宏观对照
    z = f.join(mac, on='m', how='left')
    rows = []
    for c in ('EW', 'OIW', '黑色', '有色贵金属', '能源化工', '农产品'):
        for v in ('dPMI', 'dPPI', 'dM1', 'HS300', 'USDCNH', 'OIL'):
            b, t, r2, T = ols_nw(z[c].to_numpy(), z[v].to_numpy())
            rows.append({'因子': c, '变量': v, 'β': float(b[1]), 't': float(t[1]), 'R2%': r2 * 100, '月数': T,
                         '相关': float(np.corrcoef(*z.select(c, v).drop_nulls().to_numpy().T)[0, 1])})
        allv = ['dPMI', 'dPPI', 'dM1', 'HS300']
        b, t, r2, T = ols_nw(z[c].to_numpy(), z.select(allv).to_numpy())
        rows.append({'因子': c, '变量': '合计(PMI+PPI+M1+沪深300)', 'β': np.nan, 't': np.nan, 'R2%': r2 * 100, '月数': T, '相关': np.nan})
        zz = z.filter(pl.col('OIL').is_not_null())
        b, t, r2, T = ols_nw(zz[c].to_numpy(), zz.select(allv + ['USDCNH', 'OIL']).to_numpy())
        rows.append({'因子': c, '变量': '合计+汇率+原油(2018起)', 'β': np.nan, 't': np.nan, 'R2%': r2 * 100, '月数': T, '相关': np.nan})
    mc = pl.DataFrame(rows)
    mc.write_parquet(os.path.join(OUT, 'macro.parquet'))
    print(mc.with_columns(pl.col(pl.Float64).round(3)))
    # 3. 择时
    tm = pl.concat([timing(f, mac, state, c)[0] for c in ('EW', 'OIW', 'FIX', 'FIX2')])
    tm.write_parquet(os.path.join(OUT, 'timing.parquet'))
    state.write_parquet(os.path.join(OUT, 'state.parquet'))
    print(tm.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == '__main__':
    main()
