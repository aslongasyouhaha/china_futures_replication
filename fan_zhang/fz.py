"""《The Untold Story of Commodity Futures in China》复现 —— Fan & Zhang (2020, Journal of Futures Markets)

论文方法（附录 A、Table 3–4）：
  合约：近月持有到到期前一个月的最后一个交易日再换月；第 m 近月（m = 2, 3, 4）同样到近月换月时换月。官方结算价。
  样本：至少 4 年数据的品种、每月至少 8 个品种；2004-02 ~ 2017-05。
  组合：月末按信号分四组，做多信号最好的一组、做空最差的一组，等权，持有 1 个月；多空收益 = 多头 − 空头（50% 保证金口径）。
  十个信号：
    TS   展期收益 ln F_front − ln F_2（做多高）
    HP   套保比率 |ΔOI| / 成交量（月度；做多高）
    MOM  过去 12 个月月对数收益均值（做多高）
    VOLA 变异系数 = 过去 36 个月日收益方差 / |均值|（做多高）
    OI   整条曲线总持仓的月度变化 OI_t − OI_{t−1}（做多高）
    LIQ  Amivest = 过去 2 个月日成交额 / |日收益| 的均值（做多低，即流动性差的）
    FX   对人民币有效汇率月收益的 42 个月滚动 beta（做多低）
    INF  对意外通胀的 42 个月滚动 beta（做多高）
    SKEW 过去 12 个月日收益偏度（做多低）
    VAL  ln(3.5~4.5 年前的日均价格 / 当前价格)（做多高）
中国数据对应：第 m 近月 = basis_momentum/data.py 的 paper 规则（交割月 ≥ 当前月 + 2 的合约按到期排序，正是“近月持有到到期前一个月”）。
  品种入选：同论文，整个样本中至少有 48 个月数据（一次性筛选，带有前视）；不设持仓门槛（同论文），另做“持仓 ≥ 100 手”的稳健性。
  成交额：volume × 价格 × 合约乘数（源数据 2026-04 后成交额为空）。
  FX：人民币对美元月升值（−Δln USDCNH，2012-02 起；论文用 BIS 人民币有效汇率）；INF：CPI 同比的月度变化（论文用实际 − 预测的意外通胀），beta 滞后一个月以避免用到未公布的 CPI。
用法：python fz.py
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
from pricing import nw_cov  # noqa: E402

MULT = os.path.join(HERE, '..', 'common', 'output', 'multipliers.parquet')
PAPER_END = pl.date(2017, 5, 1)          # 形成月 ≤ 2017-04（收益月 ≤ 2017-05）
MIN_N = 8
SIG = {'TS': True, 'HP': True, 'MOM': True, 'VOLA': True, 'OI': True, 'LIQ': False, 'FX': False, 'INF': True, 'SKEW': False, 'VAL': True}
NAMES = {'TS': '期限结构', 'HP': '套保压力', 'MOM': '动量', 'VOLA': '波动率', 'OI': '持仓变化', 'LIQ': '流动性', 'FX': '汇率', 'INF': '通胀', 'SKEW': '偏度', 'VAL': '价值'}


def nw_t(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / np.sqrt(nw_cov(x)[0, 0] / len(x))


def stats(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 12:
        return {'年化%': np.nan, 't': np.nan, '年化波动%': np.nan, '夏普': np.nan, '月数': len(r)}
    mu, sd = r.mean() * 12, r.std(ddof=1) * np.sqrt(12)
    return {'年化%': mu * 100, 't': nw_t(r), '年化波动%': sd * 100, '夏普': mu / sd, '月数': len(r)}


# ---------------------------------------------------------------- 数据
def total_oi():
    rows = []
    for f in glob.glob(f'{RAW_DATA}/*/*/daily_index.parquet'):
        p = os.path.basename(os.path.dirname(f))
        x = pl.read_parquet(f, columns=['trading_day', 'open_interest']).with_columns(pl.lit(p).alias('product'), pl.col('trading_day').dt.truncate('1mo').alias('m'))
        rows.append(x.group_by(['product', 'm']).agg(pl.col('open_interest').sort_by('trading_day').last().alias('toi')))
    t = pl.concat(rows).sort(['product', 'm'])
    return t.with_columns((pl.col('toi') - pl.col('toi').shift(1).over('product')).alias('OI'),
                          (pl.col('m').shift(1).over('product') == pl.col('m').dt.offset_by('-1mo')).alias('oi_ok'))


def macro():
    """月度宏观序列：FX = 人民币对美元月升值（−Δln USDCNH，代替 BIS 人民币有效汇率）；INF = CPI 同比较上月的变化（代替“实际 − 预测”的意外通胀）。"""
    ref = f'{RAW_DATA}/参考数据'
    fx = pl.read_parquet(f'{ref}/fx_usdcnh.parquet').with_columns(((pl.col('bid_close') + pl.col('ask_close')) / 2).alias('mid'),
                                                                   pl.col('trading_day').dt.truncate('1mo').alias('m'))
    fx = fx.sort('trading_day').group_by('m').agg(pl.col('mid').last()).sort('m').with_columns((-(pl.col('mid').log().diff())).alias('FXr'))
    cpi = pl.read_parquet(f'{ref}/cn_cpi.parquet').select(pl.col('month').str.to_date('%Y%m').alias('m'), pl.col('nt_yoy').cast(pl.Float64)).sort('m')
    cpi = cpi.with_columns(pl.col('nt_yoy').diff().alias('INFr'))
    return fx.select('m', 'FXr'), cpi.select('m', 'INFr')


def rolling_beta(cn, fac, col, lag):
    """形成月 t 的 beta：过去 42 个月（至少 36 个）月对数收益对因子回归；lag = 1 表示只用到 t−1 月的因子（CPI 次月才公布）。"""
    x = cn.select('product', 'm', 'lr_in').join(fac, on='m', how='left').sort(['product', 'm'])
    out = []
    for (p,), g in x.partition_by(['product'], as_dict=True).items():
        y = g['lr_in'].to_numpy()
        f = g[col].to_numpy()
        ms = g['m'].to_list()
        for i in range(len(g)):
            j = i - lag
            lo = max(0, j - 41)
            if j < 0:
                continue
            yy, ff = y[lo:j + 1], f[lo:j + 1]
            ok = ~np.isnan(yy) & ~np.isnan(ff)
            if ok.sum() >= 36 and ff[ok].var() > 0:
                out.append({'product': p, 'm': ms[i], col.replace('r', '') + '_b': float(np.cov(yy[ok], ff[ok])[0, 1] / ff[ok].var(ddof=1))})
    return pl.DataFrame(out, schema={'product': pl.Utf8, 'm': pl.Date, col.replace('r', '') + '_b': pl.Float64})


def build():
    c = pl.read_parquet(os.path.join(BM, 'curve_paper.parquet'))
    d = pl.read_parquet(os.path.join(BM, 'daily_paper.parquet')).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('hm'),
                                                                               pl.col('trading_day').dt.year().cast(pl.Int64).alias('year'))
    mult = pl.read_parquet(MULT).select('product', 'year', 'mult')
    d = d.join(mult, on=['product', 'year'], how='left').with_columns((pl.col('volume') * pl.col('px') * pl.col('mult').fill_null(1.0)).alias('rmb'),
                                                                      (pl.col('r') + 1).log().alias('lr'))
    front = c.filter(pl.col('n') == 1).select('product', 'm', pl.col('px').alias('f1'))
    second = c.filter(pl.col('n') == 2).select('product', 'm', pl.col('px').alias('f2'))
    ts = front.join(second, on=['product', 'm']).select('product', 'm', (pl.col('f1').log() - pl.col('f2').log()).alias('TS'))
    toi = total_oi()
    fxs, infs = macro()
    # 品种上市月数（事前可知）
    fb = pl.read_parquet(f'{RAW_DATA}/参考数据/futures_contracts.parquet').select('fut_code', 'list_date').drop_nulls()
    fb = fb.with_columns(pl.col('fut_code').str.to_uppercase(), pl.col('list_date').str.to_date('%Y%m%d'))
    lst = fb.group_by('fut_code').agg(pl.col('list_date').min().dt.truncate('1mo').alias('m_list')).rename({'fut_code': 'product'})
    first = c.group_by('product').agg(pl.col('m').min().alias('m_data')).join(lst, on='product', how='left')
    first = first.select('product', pl.min_horizontal('m_data', pl.col('m_list').fill_null(pl.col('m_data'))).alias('m0'))
    out = []
    for n in (1, 2, 3, 4):
        cn = c.filter(pl.col('n') == n).select('product', 'm', 'symbol', 'ret', 'open_interest', 'volume', 'px').sort(['product', 'm'])
        cn = cn.with_columns((pl.col('ret') + 1).log().alias('lret'))
        # 过去 12 个月的月对数收益：形成月 s 的 ret 属于 s+1
        hist = cn.select('product', pl.col('m').dt.offset_by('1mo').alias('m'), pl.col('lret').alias('lr_in'))
        cn = cn.join(hist, on=['product', 'm'], how='left').sort(['product', 'm'])
        cn = cn.with_columns(pl.col('lr_in').rolling_mean(12).over('product').alias('MOM'),
                             (pl.col('m').shift(11).over('product') == pl.col('m').dt.offset_by('-11mo')).alias('mom_ok'))
        cn = cn.with_columns(pl.when(pl.col('mom_ok')).then(pl.col('MOM')).alias('MOM'))
        # 日度信号
        dn = d.filter((pl.col('n') == n) & pl.col('r').is_not_null()).sort(['product', 'trading_day'])
        rows = []
        for (p,), g in dn.partition_by(['product'], as_dict=True).items():
            hm = g['hm'].to_numpy().astype('datetime64[M]')
            lr = g['lr'].to_numpy()
            rmb = g['rmb'].to_numpy()
            px = g['px'].to_numpy()
            vol = g['volume'].to_numpy()
            oi = g['open_interest'].to_numpy()
            for m in np.unique(hm):
                w = lambda k, s=0: (hm > m - k) & (hm <= m - s)   # noqa: E731
                w1, w2, w12, w36 = w(1), w(2), w(12), w(36)
                row = {'product': p, 'm': np.datetime64(m, 'D').item()}
                x = lr[w12]
                row['SKEW'] = float(((x - x.mean()) ** 3).mean() / x.std(ddof=1) ** 3) if len(x) >= 150 and x.std() > 0 else np.nan
                y = lr[w36]
                row['VOLA'] = float(y.var(ddof=1) / abs(y.mean())) if len(y) >= 500 and y.mean() != 0 else np.nan
                z = w2 & (np.abs(lr) > 0)
                row['LIQ'] = float(np.mean(rmb[z] / np.abs(lr[z]))) if z.sum() >= 10 else np.nan
                wv = (hm > m - 54) & (hm <= m - 42)
                row['VAL'] = float(np.log(np.nanmean(px[wv]) / px[hm <= m][-1])) if wv.sum() >= 150 else np.nan
                # 套保比率：当月 |ΔOI| / 当月成交量（该合约序列）
                mo = np.where(w1)[0]
                prev = np.where((hm == m - 1))[0]
                if len(mo) and len(prev) and vol[mo].sum() > 0:
                    row['HP'] = float(abs(oi[mo[-1]] - oi[prev[-1]]) / vol[mo].sum())
                else:
                    row['HP'] = np.nan
                rows.append(row)
        sd = pl.DataFrame(rows).with_columns(pl.col('m').cast(pl.Date))
        cn = cn.join(sd, on=['product', 'm'], how='left').join(ts, on=['product', 'm'], how='left')
        cn = cn.join(toi.filter(pl.col('oi_ok')).select('product', 'm', 'OI'), on=['product', 'm'], how='left')
        cn = cn.join(first, on='product').with_columns(((pl.col('m').dt.year() - pl.col('m0').dt.year()) * 12 + pl.col('m').dt.month() - pl.col('m0').dt.month()).alias('age'))
        cn = cn.filter(pl.col('ret').is_not_null() & pl.col('ret').is_finite())
        # FX、INF：用该合约序列过去 42 个月的已实现月收益估计 beta（FX 同月，INF 滞后一月）
        cn = cn.join(rolling_beta(cn, fxs, 'FXr', 0).rename({'FX_b': 'FX'}), on=['product', 'm'], how='left')
        cn = cn.join(rolling_beta(cn, infs, 'INFr', 1).rename({'INF_b': 'INF'}), on=['product', 'm'], how='left')
        out.append(cn.with_columns(pl.lit(n).alias('n')))
    x = pl.concat(out, how='diagonal_relaxed')
    # 论文：整个样本中至少有 4 年数据的品种（一次性筛选）
    months = x.filter(pl.col('n') == 1).group_by('product').agg(pl.col('m').n_unique().alias('nm'))
    x = x.join(months.filter(pl.col('nm') >= 48).select('product'), on='product')
    # 纤维板 FB 在 2019-12-02 报价单位由 元/张 改为 元/立方米（价格跳约 16.6 倍），之后的价格信号与收益都被污染，只用 2019-09 以前
    return x.filter(~((pl.col('product') == 'FB') & (pl.col('m') >= pl.date(2019, 9, 1))))


EXCLUDE_DIRS = {'CFFEX'}


def commodity_filter(x):
    ex = {os.path.basename(os.path.dirname(os.path.dirname(f))): None for f in []}
    prods = {}
    for f in glob.glob(f'{RAW_DATA}/*/*/daily_contracts.parquet'):
        prods[os.path.basename(os.path.dirname(f))] = os.path.basename(os.path.dirname(os.path.dirname(f)))
    keep = [p for p, e in prods.items() if e not in EXCLUDE_DIRS and p not in {'EC', 'LF', 'PPF', 'VF'}]
    return x.filter(pl.col('product').is_in(keep))


# ---------------------------------------------------------------- 组合
def long_short(x, sig, high_long):
    rows = []
    for (m,), g in x.drop_nulls(sig).group_by(['m'], maintain_order=True):
        if g.height < MIN_N:
            continue
        o = g.sort([sig, 'product'], descending=high_long)
        k = max(1, g.height // 4)
        rows.append({'m': m, 'r': o.head(k)['ret'].mean() - o.tail(k)['ret'].mean(), 'N': g.height})
    return pl.DataFrame(rows, schema={'m': pl.Date, 'r': pl.Float64, 'N': pl.Int64})


def run(x, label):
    res = []
    for n in (1, 2, 3, 4):
        xn = x.filter(pl.col('n') == n)
        for s, hl in SIG.items():
            ls = long_short(xn, s, hl)
            for per, cond in (('2004-02~2017-05（论文样本期）', pl.col('m') < PAPER_END), ('2017-06~2026-08（样本外）', pl.col('m') >= PAPER_END), ('全样本', pl.lit(True))):
                q = ls.filter(cond)
                res.append({'设定': label, 'm': n, '策略': NAMES[s], '时期': per, **stats(q['r'].to_numpy()), '每月品种中位数': q['N'].median() if q.height else None})
    return pl.DataFrame(res)


def long_only(x):
    res = []
    for n in (1, 2, 3, 4):
        e = x.filter(pl.col('n') == n).group_by('m').agg(pl.col('ret').mean().alias('r'), pl.len().alias('N')).filter(pl.col('N') >= MIN_N).sort('m')
        for per, cond in (('2004-02~2017-05（论文样本期）', pl.col('m') < PAPER_END), ('2017-06~2026-08（样本外）', pl.col('m') >= PAPER_END), ('全样本', pl.lit(True))):
            q = e.filter(cond)
            res.append({'m': n, '时期': per, **stats(q['r'].to_numpy())})
    return pl.DataFrame(res)


def main():
    os.makedirs(OUT, exist_ok=True)
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
    x = commodity_filter(build())
    x.write_parquet(os.path.join(OUT, 'panel.parquet'))
    print('品种-月', x.height, '；起止', x['m'].min(), x['m'].max(), '；品种', x['product'].n_unique())
    print('2017-05 前每月品种数中位数', x.filter((pl.col('n') == 1) & (pl.col('m') < PAPER_END)).group_by('m').len()['len'].median())
    lo = long_only(x)
    lo.write_parquet(os.path.join(OUT, 'long_only.parquet'))
    print(lo.with_columns(pl.col(pl.Float64).round(2)))
    t4 = run(x, '论文方法（无持仓门槛）')
    t4b = run(x.filter((pl.col('open_interest') >= 100) & (pl.col('volume') > 0)), '稳健性：合约持仓 ≥ 100 手且有成交')
    t = pl.concat([t4, t4b])
    t.write_parquet(os.path.join(OUT, 'table4.parquet'))
    print(t.filter(pl.col('时期').str.starts_with('2004')).with_columns(pl.col(pl.Float64).round(2)))


if __name__ == '__main__':
    main()
