"""商品三因子复现（因子与测试组合）—— Bakshi, Gao & Rossi (2019, Management Science)
《Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns》

两个版本，月末收盘构造、持有一个月、等权、不扣成本：
  paper（论文方法，主结果）
    · 持有合约 F1：排在最近月 F0 之后、且下个月不会进入交割月的第一个合约（通常是次近月）；结算价，持有到下月末
    · 期限结构 ln(y) = ln(F1/F0)，不年化；ln(y) < 0 为贴水
    · 动量：过去 6 个月按上述规则滚动持有的累计收益
    · 中国数据的必要调整：F1 当天须持仓 ≥ 500 手且有成交（否则价格不可靠）
  main（中国化口径，对照）
    · 收益：主力连续后复权；期限结构：年化 carry（近 = 剔除交割月后持仓最大，远 = 更远的持仓最大）；动量：主力过去 6 个月涨幅
两个版本的品种池相同：商品（不含中金所、月均价合约、集运指数），主力 60 日平均成交 ≥ 1000 手、上市 ≥ 60 天；2006 年起，每月 ≥ 10 个品种。
因子：AVG（等权做多全部）、CARRY（C5：最贴水 5 个 − 最升水 5 个）、CMOM（M5：过去 6 个月最高 5 个 − 最低 5 个）。
测试组合（11 个）：P1–P4（贴水品种按贴水程度分两半 P1 最深、P2；升水分两半 P3、P4 最深）、Q1（最差）–Q5、农产品、金属。
  能源化工不作测试组合：其次近月多不活跃，论文方法下有 28 个月整个板块没有合格品种（论文同样因历史不完整排除能源板块）。
"""
import glob
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
sys.path.insert(0, os.path.join(HERE, '..', 'time_series_momentum'))
sys.path.insert(0, os.path.join(HERE, '..', 'carry'))
import carry as C  # noqa: E402
import tsmom as T  # noqa: E402

CARRY_PANEL = os.path.join(HERE, '..', 'carry', 'output', 'carry_panel.parquet')
START = pl.date(2006, 1, 1)
EXCLUDE = {'EC', 'LF', 'PPF', 'VF'}
N_LEG = 5
MIN_OI = 500
VARIANTS = ('paper', 'main')
METAL = {'CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG', 'BC', 'AO', 'AD', 'LC', 'PD', 'PT', 'PS', 'SI', 'SS',
         'RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM', 'WR', 'ZC'}           # 有色、贵金属、新能源金属、黑色（含煤焦）
AGRI = {'A', 'B', 'C', 'CS', 'M', 'Y', 'P', 'RM', 'OI', 'RS', 'CF', 'CY', 'SR', 'AP', 'CJ', 'JD', 'LH', 'PK', 'WH', 'WT',
        'PM', 'RI', 'LR', 'JR', 'RR', 'GN'}                              # 其余为能源化工（含橡胶、纸浆、木材）


def sector(p):
    return '金属' if p in METAL else '农产品' if p in AGRI else '能源化工'


def main_monthly():
    df = T.load_cache().filter(pl.col('cls') == 'commodity')
    return (df.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
              .group_by(['product', 'm']).agg(pl.all().sort_by('trading_day').last()).sort(['product', 'm']))


def universe():
    me = main_monthly()
    return me.filter(~pl.col('product').is_in(list(EXCLUDE)) & (pl.col('vol60') >= 1000) & (pl.col('n_obs') >= 60)
                     & (pl.col('m') >= START)).select('product', 'm')


def main_panel(uni):
    me = main_monthly().with_columns(
        (pl.col('close_adj').shift(-1).over('product') / pl.col('close_adj') - 1).alias('ret_next'),
        (pl.col('m').shift(-1).over('product') == pl.col('m').dt.offset_by('1mo')).alias('next_ok'),
        (pl.col('close_adj') / pl.col('close_adj').shift(6).over('product') - 1).alias('mom6'),
        (pl.col('m').shift(6).over('product') == pl.col('m').dt.offset_by('-6mo')).alias('mom_ok'))
    cp = pl.read_parquet(CARRY_PANEL).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m')).select('product', 'm', 'carry')
    x = me.join(uni, on=['product', 'm']).join(cp, on=['product', 'm'], how='left')
    x = x.filter(pl.col('next_ok') & pl.col('ret_next').is_not_null() & pl.col('mom_ok') & pl.col('mom6').is_not_null()
                 & pl.col('carry').is_not_null())
    return x.select('product', 'm', 'ret_next', 'mom6', 'carry')


def contract_month_ends():
    parts = []
    for f in glob.glob(f'{T.DATA}/*/*/daily_contracts.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in EXCLUDE:
            continue
        d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'volume', 'open_interest'])
        d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'), pl.lit(p).alias('product'),
                           pl.col('trading_day').dt.truncate('1mo').alias('m'))
        me = d.group_by('m').agg(pl.col('trading_day').max().alias('me'))
        d = d.join(me, on='m').filter((pl.col('trading_day') == pl.col('me')) & (pl.col('px') > 0))
        d = d.with_columns(pl.col('symbol').map_elements(C.month_index, return_dtype=pl.Int64).alias('cm'),
                           (pl.col('m').dt.year() * 12 + pl.col('m').dt.month() - 1).alias('now'))
        parts.append(d.select('product', 'symbol', 'm', 'px', 'volume', 'open_interest', 'cm', 'now'))
    return pl.concat(parts)


def paper_contracts():
    """每个品种每个月末的 F0、F1 及 F1 的下月收益（不论是否满足流动性，供诊断用）。"""
    c = contract_month_ends()
    nxt = c.select('product', 'symbol', pl.col('m').dt.offset_by('-1mo').alias('m'), pl.col('px').alias('px_next'))
    rows = []
    for (p,), g in c.group_by(['product']):
        g = g.join(nxt.filter(pl.col('product') == p).drop('product'), on=['symbol', 'm'], how='left')
        for (m,), h in g.group_by(['m']):
            h = h.filter(pl.col('cm') >= pl.col('now')).sort('cm')           # 尚未到期
            if h.height < 2:
                continue
            f0 = h.row(0, named=True)
            # 排在 F0 之后、下个月不进入交割月的第一个合约；隔月挂牌的品种若 F0 本身已满足条件，则持有第二个合约（同论文）
            cand = h.filter((pl.col('cm') >= pl.col('now') + 2) & (pl.col('cm') > f0['cm']))
            if cand.height == 0:
                continue
            f1 = cand.row(0, named=True)
            rows.append({'product': p, 'm': m, 'f0': f0['symbol'], 'f1': f1['symbol'], 'gap': f1['cm'] - f0['cm'],
                         'lny': float(np.log(f1['px'] / f0['px'])),
                         'ret_next': f1['px_next'] / f1['px'] - 1 if f1['px_next'] is not None else None,
                         'f1_oi': f1['open_interest'], 'f1_vol': f1['volume'],
                         'ok': f1['open_interest'] >= MIN_OI and f1['volume'] > 0 and f1['px_next'] is not None})
    lp = pl.DataFrame(rows).sort(['product', 'm'])
    # 动量：过去 6 个月按论文规则滚动持有的收益连乘（ret_next 属于 m 的下一个月）
    hist = lp.select('product', pl.col('m').dt.offset_by('1mo').alias('m'), pl.col('ret_next').alias('r_in'))
    lp = lp.join(hist, on=['product', 'm'], how='left').sort(['product', 'm'])
    return lp.with_columns(((1 + pl.col('r_in')).log().rolling_sum(6).over('product').exp() - 1).alias('mom6'),
                           (pl.col('m').shift(5).over('product') == pl.col('m').dt.offset_by('-5mo')).alias('mom_ok'))


def paper_panel(uni, lp):
    x = lp.join(uni, on=['product', 'm']).filter(pl.col('ok') & pl.col('mom_ok') & pl.col('mom6').is_not_null()
                                                 & pl.col('ret_next').is_not_null())
    return x.select('product', 'm', 'ret_next', 'mom6', (-pl.col('lny')).alias('carry'))     # carry 高 = 贴水深


def build(x):
    """x: product, m, ret_next, mom6, carry。返回 (因子表, 测试组合表, 成员表)；m 为形成月，收益属于下个月。"""
    fac, port, member = [], [], []
    for (m,), g in x.sort('product').group_by(['m'], maintain_order=True):
        n = g.height
        if n < 2 * N_LEG:
            continue
        r = g['ret_next'].to_numpy()
        c = g['carry'].to_numpy()
        mo = g['mom6'].to_numpy()
        prods = g['product'].to_list()
        oc = np.argsort(-c, kind='stable')
        om = np.argsort(-mo, kind='stable')
        row = {'m': m, 'n': n, 'AVG': r.mean(),
               'CARRY': r[oc[:N_LEG]].mean() - r[oc[-N_LEG:]].mean(),
               'CMOM': r[om[:N_LEG]].mean() - r[om[-N_LEG:]].mean()}
        for k in (1, 2, 3, 4):                                  # C1..C4、M1..M4（论文 Table 1）
            row[f'C{k}'] = r[oc[:k]].mean() - r[oc[-k:]].mean()
            row[f'M{k}'] = r[om[:k]].mean() - r[om[-k:]].mean()
        row['CARRY_long'], row['CARRY_short'] = r[oc[:N_LEG]].mean(), -r[oc[-N_LEG:]].mean()
        row['CMOM_long'], row['CMOM_short'] = r[om[:N_LEG]].mean(), -r[om[-N_LEG:]].mean()
        fac.append(row)
        pr = {'m': m}
        back = np.where(c > 0)[0]
        cont = np.where(c <= 0)[0]
        if len(back) >= 2:
            idx = back[np.argsort(-c[back], kind='stable')]
            h = len(idx) // 2
            pr['P1'], pr['P2'] = r[idx[:h]].mean(), r[idx[h:]].mean()
        if len(cont) >= 2:
            idx = cont[np.argsort(-c[cont], kind='stable')]     # 升水由浅到深
            h = len(idx) - len(idx) // 2
            pr['P3'], pr['P4'] = r[idx[:h]].mean(), r[idx[h:]].mean()
        for k, idx in enumerate(np.array_split(om[::-1], 5)):  # 过去收益由低到高
            pr[f'Q{k+1}'] = r[idx].mean()
        for s in ('农产品', '金属', '能源化工'):
            sel = np.array([sector(p) == s for p in prods])
            if sel.any():
                pr[s] = r[sel].mean()
        port.append(pr)
        member += [{'m': m, 'product': prods[i], 'leg': leg} for leg, idx in
                   (('carry_long', oc[:N_LEG]), ('carry_short', oc[-N_LEG:]), ('mom_long', om[:N_LEG]), ('mom_short', om[-N_LEG:]))
                   for i in idx]
    return pl.DataFrame(fac), pl.DataFrame(port), pl.DataFrame(member)


def main():
    os.makedirs(OUT, exist_ok=True)
    uni = universe()
    lp = paper_contracts()
    lp.write_parquet(f'{OUT}/paper_contracts.parquet')
    panels = {'paper': paper_panel(uni, lp), 'main': main_panel(uni)}
    for v, x in panels.items():
        x = x.with_columns(pl.col('product').map_elements(sector, return_dtype=pl.Utf8).alias('sector'))
        x.write_parquet(f'{OUT}/panel_{v}.parquet')
        fac, port, mem = build(x)
        fac.write_parquet(f'{OUT}/factors_{v}.parquet')
        port.write_parquet(f'{OUT}/test_portfolios_{v}.parquet')
        mem.write_parquet(f'{OUT}/members_{v}.parquet')
        print(f"{v}: 品种-月 {x.height}，因子 {fac['m'].min()} ~ {fac['m'].max()}，{fac.height} 个月，每月品种数中位数 {fac['n'].median():.0f}；"
              f"测试组合缺失月数 {({c: port[c].null_count() for c in port.columns if c != 'm' and port[c].null_count()})}")


if __name__ == '__main__':
    main()
