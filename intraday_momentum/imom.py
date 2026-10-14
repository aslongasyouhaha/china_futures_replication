"""《Market Intraday Momentum》复现 —— Gao, Han, Li & Zhou (2018, Journal of Financial Economics)

论文方法（SPY，1993–2013；网络附录称标普 500 期货结果类似）：
  r1：前一交易日收盘 → 开盘后 30 分钟（含隔夜信息）；r13：最后半小时；r12：倒数第二个半小时
  回归 r13 = α + β1·r1 + β2·r12；NW t；样本外：前 5 年起递推、每月扩展，R²_OS 相对历史均值
  分组：金融危机期、首半小时波动三分位、首半小时成交量三分位（按年分组）
  经济价值：择时 η(r1)（r1>0 做多最后半小时，否则做空）、η(r1,r12)（同号才交易）；常多与买入持有对照；
            均值-方差投资者 w = r̂/(γσ̂²)，γ=5，w∈[−0.5,1.5]，确定性等价收益（CER）；交易成本（盈亏平衡）
中国对应：
  主样本：沪深 300 股指期货（IF，主力合约）；另有上证 50、中证 500、中证 1000、国债期货（对应论文的其他 ETF）；商品期货作补充
  半小时按每天实际交易时间：首半小时 = 前收盘 → 开盘（有夜盘的为夜盘开盘）后 30 分钟；末两个半小时 = 收盘前 60→30、30→0 分钟
  前收盘取同一合约的前一交易日收盘价（日线），避免换月日用错合约；分钟 K 线按开始时间标记，时点 T 的价格 = 开始于 T−1 分钟那根的收盘
  年化系数 243；“危机期”为 2015 年股灾（2015-06-15 至 2016-01-31）
"""
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
DATA = RAW_DATA
ANN = 243
GAMMA = 5
INIT_DAYS = 1215                       # 约 5 年
MAIN = [('CFFEX', 'IF', '沪深300'), ('CFFEX', 'IH', '上证50'), ('CFFEX', 'IC', '中证500'), ('CFFEX', 'IM', '中证1000'),
        ('CFFEX', 'T', '10年国债'), ('CFFEX', 'TF', '5年国债'), ('CFFEX', 'TS', '2年国债')]
COMMOD = [('SHFE', 'CU', '铜'), ('SHFE', 'AL', '铝'), ('SHFE', 'ZN', '锌'), ('SHFE', 'AU', '黄金'), ('SHFE', 'AG', '白银'), ('SHFE', 'RB', '螺纹钢'),
          ('SHFE', 'RU', '橡胶'), ('SHFE', 'BU', '沥青'), ('DCE', 'M', '豆粕'), ('DCE', 'Y', '豆油'), ('DCE', 'P', '棕榈油'), ('DCE', 'C', '玉米'),
          ('DCE', 'I', '铁矿石'), ('DCE', 'J', '焦炭'), ('DCE', 'L', '塑料'), ('DCE', 'V', 'PVC'), ('CZCE', 'TA', 'PTA'), ('CZCE', 'SR', '白糖'),
          ('CZCE', 'CF', '棉花'), ('CZCE', 'MA', '甲醇'), ('CZCE', 'RM', '菜粕'), ('CZCE', 'FG', '玻璃'), ('INE', 'SC', '原油')]
CRISIS = (pl.date(2015, 6, 15), pl.date(2016, 1, 31))


def half_hours(ex, p):
    m = pl.read_parquet(f'{DATA}/{ex}/{p}/minute_main.parquet', columns=['trading_day', 'datetime', 'symbol', 'close', 'volume']).sort('datetime')
    dc = pl.read_parquet(f'{DATA}/{ex}/{p}/daily_contracts.parquet', columns=['symbol', 'trading_day', 'close']).sort(['symbol', 'trading_day'])
    dc = dc.with_columns(pl.col('close').shift(1).over('symbol').alias('prev_close'), pl.col('trading_day').shift(1).over('symbol').alias('prev_day'))
    m = m.with_columns((pl.col('datetime') + pl.duration(minutes=1)).alias('t_end'))
    g = m.group_by('trading_day').agg(pl.col('symbol').last(), pl.col('t_end').min().alias('open_t'), pl.col('t_end').max().alias('close_t'))
    g = g.with_columns((pl.col('open_t') - pl.duration(minutes=1)).alias('open_t'))
    m = m.join(g.select('trading_day', pl.col('symbol').alias('sym_day'), 'open_t', 'close_t'), on='trading_day').filter(pl.col('symbol') == pl.col('sym_day')).sort('t_end')
    m = m.with_columns(((pl.col('t_end') - pl.col('open_t')).dt.total_minutes()).alias('since_open'),
                       ((pl.col('close_t') - pl.col('t_end')).dt.total_minutes()).alias('to_close'))
    # 各时点价格：时点之前最后一根 K 线的收盘
    first = m.filter(pl.col('since_open') <= 30).group_by('trading_day').agg(
        pl.col('close').last().alias('p30'), pl.col('since_open').max().alias('chk30'),
        pl.col('volume').sum().alias('vol1'),
        (pl.col('close').log().diff().drop_nulls() ** 2).sum().sqrt().alias('rv1'))
    pts = m.group_by('trading_day').agg(
        pl.col('close').filter(pl.col('to_close') >= 60).last().alias('p_m60'),
        pl.col('close').filter(pl.col('to_close') >= 30).last().alias('p_m30'),
        pl.col('close').last().alias('p_close'),
        pl.col('to_close').filter(pl.col('to_close') >= 30).min().alias('chk_m30'),
        pl.col('to_close').filter(pl.col('to_close') >= 60).min().alias('chk_m60'),
        pl.col('to_close').min().alias('chk_close'))
    d = first.join(pts, on='trading_day').join(g.select('trading_day', 'symbol', 'open_t', 'close_t'), on='trading_day')
    d = d.join(dc.select('symbol', 'trading_day', 'prev_close'), on=['symbol', 'trading_day'], how='left')
    # 价格时点必须基本准确（允许最多 2 分钟偏差，避免午休或停顿）
    d = d.filter((pl.col('chk30') >= 28) & (pl.col('chk_m30') <= 32) & (pl.col('chk_m60') <= 62) & (pl.col('chk_close') <= 2) & (pl.col('prev_close') > 0))
    d = d.with_columns((pl.col('p30') / pl.col('prev_close') - 1).alias('r1'), (pl.col('p_m30') / pl.col('p_m60') - 1).alias('r12'),
                       (pl.col('p_close') / pl.col('p_m30') - 1).alias('r13'), (pl.col('p_close') / pl.col('prev_close') - 1).alias('r_day'))
    return d.select('trading_day', 'symbol', 'r1', 'r12', 'r13', 'r_day', 'rv1', 'vol1').sort('trading_day')


def ols_nw(y, X, L=None):
    n = len(y)
    X = np.column_stack([np.ones(n), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    L = int(np.floor(4 * (n / 100) ** (2 / 9))) if L is None else L
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX @ S @ XtX))
    return b, b / se, 1 - e.var() / y.var()


def reg_block(d, label, prod):
    rows = []
    y = d['r13'].to_numpy()
    for name, cols in (('r1', ['r1']), ('r12', ['r12']), ('r1+r12', ['r1', 'r12'])):
        b, t, r2 = ols_nw(y, d.select(cols).to_numpy())
        row = {'品种': prod, '样本': label, '预测变量': name, '天数': len(y), '截距（年化%）': b[0] * ANN * 100, 't截距': t[0], 'R2%': r2 * 100}
        for c, bb, tt in zip(cols, b[1:], t[1:]):
            row[f'β_{c}×100'] = bb * 100
            row[f't_{c}'] = tt
        rows.append(row)
    return rows


def oos(d, prod):
    """递推样本外：前 INIT_DAYS 天起，每月重估一次；R²_OS 相对历史均值。同时给出择时与效用增益所需的预测。"""
    if d.height < INIT_DAYS + 250:
        return [], None
    d = d.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m')).with_row_index('i')
    y = d['r13'].to_numpy()
    X1 = d.select('r1').to_numpy()
    X2 = d.select('r1', 'r12').to_numpy()
    months = d['m'].to_numpy()
    start = INIT_DAYS
    preds = {'hist': np.full(len(y), np.nan), 'r1': np.full(len(y), np.nan), 'r12': np.full(len(y), np.nan), 'r1+r12': np.full(len(y), np.nan)}
    var = np.full(len(y), np.nan)
    i = start
    while i < len(y):
        j = i
        while j < len(y) and months[j] == months[i]:
            j += 1
        tr = slice(0, i)
        preds['hist'][i:j] = y[tr].mean()
        var[i:j] = y[tr].var()
        for k, X in (('r1', X1), ('r12', d.select('r12').to_numpy()), ('r1+r12', X2)):
            Xa = np.column_stack([np.ones(i), X[tr]])
            b = np.linalg.lstsq(Xa, y[tr], rcond=None)[0]
            preds[k][i:j] = np.column_stack([np.ones(j - i), X[i:j]]) @ b
        i = j
    sel = np.arange(start, len(y))
    yy = y[sel]
    den = ((yy - preds['hist'][sel]) ** 2).sum()
    rows = [{'品种': prod, '预测变量': k, 'R2_OS%': (1 - ((yy - preds[k][sel]) ** 2).sum() / den) * 100, '样本外起点': str(d['trading_day'][start]), '天数': len(sel)}
            for k in ('r1', 'r12', 'r1+r12')]
    # 效用增益（论文 Table 7）
    util = []
    for k in ('hist', 'r1', 'r1+r12'):
        w = np.clip(preds[k][sel] / (GAMMA * var[sel]), -0.5, 1.5)
        rp = w * yy
        mu, sd = rp.mean() * ANN, rp.std() * np.sqrt(ANN)
        util.append({'品种': prod, '预测': k, '年化收益%': mu * 100, '年化波动%': sd * 100, '夏普': mu / sd if sd > 0 else np.nan, 'U': mu - GAMMA / 2 * sd ** 2})
    u0 = util[0]['U']
    for u in util:
        u['CER%'] = (u['U'] - u0) * 100
    return rows, util


def timing(d, prod):
    y = d['r13'].to_numpy()
    s1 = np.where(d['r1'].to_numpy() > 0, 1.0, -1.0)
    s12 = np.where(d['r12'].to_numpy() > 0, 1.0, -1.0)
    both = np.where(s1 == s12, s1, 0.0)
    out = []
    for name, r in (('η(r1)', s1 * y), ('η(r12)', s12 * y), ('η(r1,r12)', both * y), ('常多（最后半小时）', y), ('买入持有（全天）', d['r_day'].to_numpy())):
        mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
        b, t, _ = ols_nw(r, np.zeros((len(r), 0)))
        n_trade = (both != 0).sum() if name == 'η(r1,r12)' else len(r)
        out.append({'品种': prod, '策略': name, '年化收益%': mu * 100, 't': t[0], '年化波动%': sd * 100, '夏普': mu / sd,
                    '偏度': float(((r - r.mean()) ** 3).mean() / r.std() ** 3), '峰度': float(((r - r.mean()) ** 4).mean() / r.std() ** 4),
                    '成功率%': (r >= 0).mean() * 100 if name != 'η(r1,r12)' else (r[both != 0] >= 0).mean() * 100,
                    '每笔平均收益bp': r.sum() / max(n_trade, 1) * 1e4})
    return out


def run(products, tag):
    reg, sub, os_rows, util, tim, splits = [], [], [], [], [], []
    for ex, p, name in products:
        d = half_hours(ex, p)
        d.write_parquet(os.path.join(OUT, f'halfhours_{p}.parquet'))
        reg += reg_block(d, '全样本', name)
        c = d.filter(pl.col('trading_day').is_between(*CRISIS))
        if c.height > 50:
            sub += reg_block(c, '2015 股灾期', name) + reg_block(d.filter(~pl.col('trading_day').is_between(*CRISIS)), '剔除股灾期', name)
        for lab, col, by_year in (('首半小时波动', 'rv1', False), ('首半小时成交量', 'vol1', True)):
            q = d.with_columns((pl.col(col).rank().over(pl.col('trading_day').dt.year()) / pl.len().over(pl.col('trading_day').dt.year())).alias('pct')
                               if by_year else (pl.col(col).rank() / pl.len()).alias('pct'))
            for g, lo, hi in (('低', 0, 1 / 3), ('中', 1 / 3, 2 / 3), ('高', 2 / 3, 1.01)):
                qq = q.filter((pl.col('pct') > lo) & (pl.col('pct') <= hi))
                r = reg_block(qq, f'{lab}：{g}', name)[2]
                splits.append(r)
        o, u = oos(d, name)
        os_rows += o
        util += u or []
        tim += timing(d, name)
        print(name, d.height, str(d['trading_day'].min()), flush=True)
    for nm, rows in (('reg', reg), ('subsample', sub), ('oos', os_rows), ('utility', util), ('timing', tim), ('splits', splits)):
        if rows:
            pl.DataFrame(rows).write_parquet(os.path.join(OUT, f'{tag}_{nm}.parquet'))
    return reg, sub, os_rows, util, tim, splits


def main():
    os.makedirs(OUT, exist_ok=True)
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(240)
    for products, tag in ((MAIN, 'fin'), (COMMOD, 'com')):
        reg, sub, o, u, t, s = run(products, tag)
        print(f'\n######## {tag} ########')
        print(pl.DataFrame(reg).with_columns(pl.col(pl.Float64).round(2)))
        if o:
            print(pl.DataFrame(o).with_columns(pl.col('R2_OS%').round(2)))
            print(pl.DataFrame(u).with_columns(pl.col(pl.Float64).round(2)))
        print(pl.DataFrame(t).with_columns(pl.col(pl.Float64).round(2)))
        if tag == 'fin':
            print(pl.DataFrame(sub).with_columns(pl.col(pl.Float64).round(2)))
            print(pl.DataFrame(s).filter(pl.col('品种') == '沪深300').with_columns(pl.col(pl.Float64).round(2)))


if __name__ == '__main__':
    main()
