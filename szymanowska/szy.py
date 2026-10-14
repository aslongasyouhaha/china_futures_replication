"""《An Anatomy of Commodity Futures Risk Premia》复现 —— Szymanowska, de Roon, Nijman & van den Goorbergh (2014, Journal of Finance)

论文方法：
  双月频率；剔除交割月及前一个月的价格（在 T−1 月之前换月）；最近月合约代替现货
  一期（两个月）收益 r(1)：买入“到期还有一期”的合约，持有一期；第 k 期合约的一期收益 r(k)
  Short Roll(n)  = 连续 n 期的 r(1) 之和           → 只含现货溢价
  Holding(n)     = 买入第 n 期合约持有 n 期        → 现货溢价 + 期限溢价
  Excess Holding = Holding − Short Roll             → 只含期限溢价
  Spreading(n)   = (1/n) Σ_k Σ_j [r(k) − r(1)]      → 期限溢价
  每期按基差（及其他变量）分四组，等权平均对数收益；P4 − P1；年化 = 每期均值 × 6 / n
  Table II 基差排序（含 B.1 按全样本平均基差、B.2 按偏离平均的部分）；Table III 其他排序；
  Table IV 基差因子（Holding，P3+P4 − P1−P2）解释 Short Roll 与 Excess Holding；Table V 用 Spreading 的一个 / 两个基差因子解释 Excess Holding；
  Table VI 其他变量的因子解释 Short Roll
中国对应：
  双月日期 = 偶数月末；“第 n 期合约” = 交割月 ≥ 当前月 + 2 + 2n 的第一个合约（n = 0 为现货代理），保证持有到期时仍未进入交割前一月；
  官方结算价（缺失用收盘）；基差 y = [ln F(1) − ln F(0)] / (两合约相隔的期数)
  排序变量：基差；动量（t−12 ~ t−1 月主力对数收益之和）；变异系数（t−36 ~ t−1 月主力月收益的方差 / 均值）；
            通胀 beta（60 个月滚动，对 CPI 环比的变化，滞后一月）；美元 beta（60 个月滚动，对 USDCNH 月变化，2012 年起）；
            套保压力（前 20 名会员 (空 − 多) / (空 + 多)，代替 CFTC）；持仓（总持仓 12 个月增长）；流动性（过去两个月日均 Amivest = 成交额 / |收益|）
  主样本不设流动性门槛（同论文）；稳健性：进场时所用合约持仓 ≥ 100 手
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
import carry as CA  # noqa: E402
from pricing import nw_cov  # noqa: E402

RAW = RAW_DATA
SKIP = {'EC', 'LF', 'PPF', 'VF'}
NMAX = 4
MIN_N = 12          # 论文：每组至少 3 个品种
MULT = os.path.join(HERE, '..', 'common', 'output', 'multipliers.parquet')


def nw_t(x, L):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / np.sqrt(nw_cov(x, L)[0, 0] / len(x))


# ---------------------------------------------------------------- 合约面板
def contracts():
    parts = []
    for f in glob.glob(f'{RAW}/*/*/daily_contracts.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in SKIP:
            continue
        d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'volume', 'open_interest'])
        d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px'), pl.col('trading_day').dt.truncate('1mo').alias('m')).filter(pl.col('px') > 0)
        me = d.group_by('m').agg(pl.col('trading_day').max().alias('me'))
        d = d.join(me, on='m').filter(pl.col('trading_day') == pl.col('me'))
        d = d.with_columns(pl.col('symbol').map_elements(CA.month_index, return_dtype=pl.Int64).alias('cm'),
                           (pl.col('m').dt.year() * 12 + pl.col('m').dt.month() - 1).alias('mi'), pl.lit(p).alias('product'))
        parts.append(d.select('product', 'symbol', 'm', 'mi', 'cm', 'px', 'volume', 'open_interest'))
    return pl.concat(parts).filter(pl.col('product') != 'FB')   # 纤维板 2019-12 改报价单位，整体剔除（论文样本也无此类品种）


def periods(c):
    """每个品种每个双月日期 t：第 0..NMAX 期合约及其在 t、t+1..t+NMAX 的价格。"""
    c = c.filter(pl.col('m').dt.month() % 2 == 0)
    price = {(r[0], r[1], r[2]): (r[3], r[4], r[5]) for r in c.select('product', 'symbol', 'mi', 'px', 'open_interest', 'volume').iter_rows()}
    rows = []
    for (p,), g in c.partition_by(['product'], as_dict=True).items():
        for (mi,), h in g.partition_by(['mi'], as_dict=True).items():
            h = h.filter(pl.col('cm') >= mi + 2).sort('cm')
            if h.height == 0:
                continue
            cms, syms = h['cm'].to_list(), h['symbol'].to_list()
            rec = {'product': p, 'mi': mi}
            for n in range(NMAX + 1):
                k = next((i for i, x in enumerate(cms) if x >= mi + 2 + 2 * n), None)
                if k is None:
                    break
                s = syms[k]
                rec[f's{n}'] = s
                rec[f'cm{n}'] = cms[k]
                rec[f'p{n}'] = price[(p, s, mi)][0]
                rec[f'oi{n}'] = price[(p, s, mi)][1]
                # 该合约在 j 期后的价格（j = 1..NMAX）
                for j in range(1, NMAX + 1):
                    q = price.get((p, s, mi + 2 * j))
                    rec[f'p{n}_{j}'] = q[0] if q else None
            rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None).sort(['product', 'mi'])


def returns(P):
    """r(k) 一期收益、Short Roll、Holding、Excess Holding、Spreading（均为对数收益）。"""
    P = P.with_columns(*[(pl.col(f'p{k}_1').log() - pl.col(f'p{k}').log()).alias(f'r{k}') for k in range(1, NMAX + 1)],
                       *[(pl.col(f'p{n}_{n}').log() - pl.col(f'p{n}').log()).alias(f'HOLD{n}') for n in range(1, NMAX + 1)],
                       ((pl.col('p1').log() - pl.col('p0').log()) / ((pl.col('cm1') - pl.col('cm0')) / 2)).alias('BASIS'))
    P = P.sort(['product', 'mi'])
    lead = lambda c, j: pl.col(c).shift(-j).over('product')   # noqa: E731
    okj = lambda j: (pl.col('mi').shift(-j).over('product') == pl.col('mi') + 2 * j)   # noqa: E731
    out = []
    for n in range(1, NMAX + 1):
        sr = sum(pl.when(okj(j)).then(lead('r1', j)) for j in range(n))
        spr = sum(pl.when(okj(j)).then(lead(f'r{k}', j) - lead('r1', j)) for k in range(1, n + 1) for j in range(n)) / n
        out += [sr.alias(f'SR{n}'), (pl.col(f'HOLD{n}') - sr).alias(f'EH{n}'), spr.alias(f'SP{n}')]
    return P.with_columns(out)


# ---------------------------------------------------------------- 排序变量（月度，主力合约）
def monthly():
    """各品种月末：主力复权对数收益、动量、变异系数、Amivest（过去两个月日均 成交额/|收益|）、总持仓、前 20 名套保压力、通胀 / 美元 beta。"""
    mult = pl.read_parquet(MULT).select('product', 'year', 'mult')
    rows, hps = [], []
    for f in glob.glob(f'{RAW}/*/*/daily_main.parquet'):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in SKIP or p == 'FB':
            continue
        d = pl.read_parquet(f, columns=['trading_day', 'close', 'volume', 'close_adj']).sort('trading_day')
        oi = pl.read_parquet(f.replace('daily_main', 'daily_index'), columns=['trading_day', 'open_interest']).rename({'open_interest': 'toi'})
        d = d.join(oi, on='trading_day', how='left').with_columns(pl.lit(p).alias('product'), pl.col('trading_day').dt.year().cast(pl.Int64).alias('year'),
                                                                   pl.col('close_adj').log().diff().alias('lr'), pl.col('trading_day').dt.truncate('1mo').alias('m'))
        d = d.join(mult, on=['product', 'year'], how='left').with_columns((pl.col('volume') * pl.col('close') * pl.col('mult').fill_null(1.0)).alias('rmb'))
        d = d.with_columns(pl.when(pl.col('lr').abs() > 0).then(pl.col('rmb') / pl.col('lr').abs()).alias('amv'))
        rows.append(d.group_by('product', 'm').agg(pl.col('close_adj').sort_by('trading_day').last().alias('ca'), pl.col('toi').sort_by('trading_day').last().alias('OI'),
                                                   pl.col('amv').sum().alias('amv_s'), pl.col('amv').count().alias('amv_n')))
        hf = os.path.join(os.path.dirname(f), 'holding_rank.parquet')
        if os.path.exists(hf):
            h = pl.read_parquet(hf, columns=['trading_day', 'level', 'row_type', 'long_hld', 'short_hld']).filter((pl.col('level') == '合约') & (pl.col('row_type') == '会员'))
            h = h.group_by('trading_day').agg(pl.col('long_hld').fill_null(0).sum().alias('L'), pl.col('short_hld').fill_null(0).sum().alias('S'))
            h = h.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m')).sort('trading_day').group_by('m').agg(pl.col('L').last(), pl.col('S').last())
            hps.append(h.select(pl.lit(p).alias('product'), 'm', ((pl.col('S') - pl.col('L')) / (pl.col('S') + pl.col('L'))).alias('HP')))
    m = pl.concat(rows).sort(['product', 'm'])
    ok = lambda k: pl.col('m').shift(k).over('product') == pl.col('m').dt.offset_by(f'-{k}mo')   # noqa: E731
    m = m.with_columns(pl.when(ok(1)).then(pl.col('ca').log() - pl.col('ca').shift(1).over('product').log()).alias('lret'))
    m = m.with_columns(pl.when(ok(11)).then(pl.col('lret').rolling_sum(12).over('product')).alias('MOM'),
                       pl.when(ok(35)).then(pl.col('lret').rolling_var(36, min_periods=24).over('product')
                                            / pl.col('lret').rolling_mean(36, min_periods=24).over('product')).alias('CV'),
                       pl.when(ok(1)).then((pl.col('amv_s') + pl.col('amv_s').shift(1).over('product'))
                                           / (pl.col('amv_n') + pl.col('amv_n').shift(1).over('product'))).alias('LIQ'))
    m = m.join(pl.concat(hps), on=['product', 'm'], how='left')
    # 通胀 beta：对“CPI 环比的变化”（论文：change in one-month CPI inflation），CPI 次月公布 → 滞后一月；美元 beta：对 Δln USDCNH（2012 年起）
    ref = f'{RAW}/参考数据'
    cpi = pl.read_parquet(f'{ref}/cn_cpi.parquet').select(pl.col('month').str.to_date('%Y%m').alias('m'), pl.col('nt_mom').cast(pl.Float64)).sort('m')
    cpi = cpi.with_columns(pl.col('nt_mom').diff().alias('INFr'))
    fx = pl.read_parquet(f'{ref}/fx_usdcnh.parquet').with_columns(((pl.col('bid_close') + pl.col('ask_close')) / 2).alias('mid'),
                                                                   pl.col('trading_day').dt.truncate('1mo').alias('m'))
    fx = fx.sort('trading_day').group_by('m').agg(pl.col('mid').last()).sort('m').with_columns(pl.col('mid').log().diff().alias('FXr'))
    m = m.join(beta(m, cpi.select('m', 'INFr'), 'INFr', 1, 'INF'), on=['product', 'm'], how='left')
    m = m.join(beta(m, fx.select('m', 'FXr'), 'FXr', 0, 'FX'), on=['product', 'm'], how='left')
    return m.select('product', 'm', 'MOM', 'CV', 'INF', 'FX', 'HP', 'OI', 'LIQ')


def beta(m, fac, col, lag, name, win=60, mn=36):
    """60 个月滚动（至少 36 个月）月对数收益对因子的 beta。"""
    x = m.select('product', 'm', 'lret').join(fac, on='m', how='left').sort(['product', 'm'])
    out = []
    for (p,), g in x.partition_by(['product'], as_dict=True).items():
        y, f, ms = g['lret'].to_numpy().astype(float), g[col].to_numpy().astype(float), g['m'].to_list()
        for i in range(len(g)):
            j = i - lag
            if j < mn - 1:
                continue
            yy, ff = y[max(0, j - win + 1):j + 1], f[max(0, j - win + 1):j + 1]
            ok = ~np.isnan(yy) & ~np.isnan(ff)
            if ok.sum() >= mn and ff[ok].var() > 0:
                out.append({'product': p, 'm': ms[i], name: float(np.cov(yy[ok], ff[ok])[0, 1] / ff[ok].var(ddof=1))})
    return pl.DataFrame(out, schema={'product': pl.Utf8, 'm': pl.Date, name: pl.Float64})


def add_signals(P, M):
    P = P.with_columns(pl.date(pl.col('mi') // 12, pl.col('mi') % 12 + 1, 1).alias('m')).join(M, on=['product', 'm'], how='left')
    mb = P.group_by('product').agg(pl.col('BASIS').mean().alias('BASIS_MEAN'))
    return P.join(mb, on='product').with_columns((pl.col('BASIS') - pl.col('BASIS_MEAN')).alias('BASIS_DEV'))


# ---------------------------------------------------------------- 排序与检验
VARS = ['BASIS', 'MOM', 'CV', 'INF', 'FX', 'HP', 'OI', 'LIQ']
VNAME = {'BASIS': '基差', 'MOM': '动量', 'CV': '变异系数', 'INF': '通胀 beta', 'FX': '美元 beta', 'HP': '套保压力', 'OI': '持仓', 'LIQ': '流动性',
         'BASIS_MEAN': '平均基差（B.1）', 'BASIS_DEV': '基差偏离（B.2）'}


def sort_ports(P, var, ret, min_n=MIN_N):
    """每个双月日期按 var 分四组（等权），返回 mi × [P1..P4] 的 ret。"""
    x = P.filter(pl.col(var).is_not_null() & pl.col(var).is_finite() & pl.col(ret).is_not_null() & pl.col(ret).is_finite())
    x = x.with_columns(pl.len().over('mi').alias('N'), (pl.col(var).rank('ordinal').over('mi') - 1).alias('rk'))
    x = x.filter(pl.col('N') >= min_n).with_columns((pl.col('rk') * 4 // pl.col('N') + 1).alias('q'))
    if x.height == 0:
        return pl.DataFrame(schema={'mi': pl.Int64, **{f'P{q}': pl.Float64 for q in range(1, 5)}})
    w =x.group_by('mi', 'q').agg(pl.col(ret).mean()).pivot(values=ret, index='mi', columns='q').sort('mi')
    return w.rename({str(q): f'P{q}' for q in range(1, 5)}).select('mi', 'P1', 'P2', 'P3', 'P4')


def hml(w):
    return (w['P3'] + w['P4']) / 2 - (w['P1'] + w['P2']) / 2


def nwl(T, n):
    return max(n - 1, 0) + int(np.floor(4 * (T / 100) ** (2 / 9)))   # n 期收益每期采样，重叠 n−1 期


def ann(x, n):
    return float(np.nanmean(x)) * 6 / n * 100


def tstat(x, n):
    x = np.asarray(x, float)
    return nw_t(x, nwl(np.sum(~np.isnan(x)), n))


def wald(R, F, n):
    """时间序列回归 R_i = α_i + β_i F + ε，α 联合为零的 Wald 检验（GMM，Newey-West）。"""
    R, F = np.asarray(R, float), np.asarray(F, float).reshape(len(R), -1)
    ok = ~np.isnan(R).any(1) & ~np.isnan(F).any(1)
    R, F = R[ok], F[ok]
    T, N = R.shape
    X = np.column_stack([np.ones(T), F])
    k = X.shape[1]
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    E = R - X @ B
    G = np.column_stack([X * E[:, [i]] for i in range(N)])
    S = nw_cov(G, nwl(T, n))
    Ainv = np.kron(np.eye(N), np.linalg.inv(X.T @ X / T))
    V = Ainv @ S @ Ainv / T
    idx = [i * k for i in range(N)]
    a, Va = B[0], V[np.ix_(idx, idx)]
    # 排序组合本身构成因子时（如基差排序 + 基差因子，n = 1）残差线性相关，协方差奇异 → 广义逆，自由度取秩
    sc = np.sqrt(np.diag(Va))
    C = Va / np.outer(sc, sc)
    df = np.linalg.matrix_rank(C, tol=1e-8)
    W = float((a / sc) @ np.linalg.pinv(C, rcond=1e-8) @ (a / sc))
    return dict(alpha=a * 6 / n * 100, t=a / sc, W=W, p=float(st.chi2.sf(W, df)), r2=1 - E.var(0) / R.var(0), T=T)


def tables(P, tag):
    srt, tst = [], []
    # Table II / III：各排序的 Short Roll、Excess Holding（Holding、Spreading 备查，同论文网络附录）
    for var in ['BASIS', 'BASIS_MEAN', 'BASIS_DEV'] + VARS[1:]:
        for a in ('SR', 'EH', 'HOLD', 'SP'):
            for n in range(1, NMAX + 1):
                if a == 'EH' and n == 1:
                    continue
                w = sort_ports(P, var, f'{a}{n}')
                if w.height < 24:
                    continue
                row = {'样本': tag, '变量': var, '收益': a, 'n': n, '期数': w.height, '起': w['mi'].min()}
                for q in ('P1', 'P2', 'P3', 'P4'):
                    row[q] = ann(w[q].to_numpy(), n)
                    row[q + '_t'] = tstat(w[q].to_numpy(), n)
                hl = (w['P4'] - w['P1']).to_numpy()
                row['P4-P1'], row['P4-P1_t'] = ann(hl, n), tstat(hl, n)
                srt.append(row)
    # 因子：基差 Holding（P3+P4 − P1−P2）、基差 Spreading（一个 / 高低两个）、其他变量的 Holding 因子
    fac = {}
    for n in range(1, NMAX + 1):
        h = sort_ports(P, 'BASIS', f'HOLD{n}')
        s = sort_ports(P, 'BASIS', f'SP{n}')
        f = h.select('mi', hml(h).alias('F')).join(s.select('mi', hml(s).alias('FS'), ((s['P3'] + s['P4']) / 2).alias('FH'), ((s['P1'] + s['P2']) / 2).alias('FL')),
                                                   on='mi', how='outer_coalesce')
        mk = P.filter(pl.col(f'HOLD{n}').is_finite()).group_by('mi').agg(pl.col(f'HOLD{n}').mean().alias('MKT'))
        f = f.join(mk, on='mi', how='left')
        for v in VARS[1:]:
            g = sort_ports(P, v, f'HOLD{n}')
            f = f.join(g.select('mi', hml(g).alias(f'F_{v}')), on='mi', how='left')
        fac[n] = f
    models = [('SR', ['F'], 'IV 基差因子'), ('EH', ['F'], 'IV 基差因子'), ('EH', ['FS'], 'V 基差 Spreading 因子'), ('EH', ['FH', 'FL'], 'V 高/低基差 Spreading 双因子'),
              ('SR', ['F', 'MKT'], '中国补充 基差因子+等权市场'), ('EH', ['FH', 'FL', 'MKT'], '中国补充 双因子+等权市场')] + \
             [('SR', [f'F_{v}'], f'VI {VNAME[v]}因子') for v in VARS[1:]]
    for var in VARS:
        for n in range(1, NMAX + 1):
            for a, fcols, model in models:
                if a == 'EH' and n == 1:
                    continue
                w = sort_ports(P, var, f'{a}{n}').join(fac[n], on='mi', how='inner').drop_nulls(fcols)
                if w.height < 24:
                    continue
                r = wald(w.select('P1', 'P2', 'P3', 'P4').to_numpy(), w.select(fcols).to_numpy(), n)
                tst.append({'样本': tag, '排序': var, '收益': a, 'n': n, '模型': model, '期数': r['T'],
                            **{f'a{q}': float(r['alpha'][q - 1]) for q in range(1, 5)}, **{f't{q}': float(r['t'][q - 1]) for q in range(1, 5)},
                            '平均|α|': float(np.mean(np.abs(r['alpha']))), 'Wald': r['W'], 'p': r['p'], '平均R2': float(np.mean(r['r2']))})
    return pl.DataFrame(srt), pl.DataFrame(tst)


def table1(P):
    """板块等权指数（对应 Table I）。"""
    sys.path.insert(0, os.path.join(HERE, '..', 'open_interest'))
    from oi import sector
    x = P.with_columns(pl.col('product').map_elements(sector, return_dtype=pl.Utf8).alias('sector'))
    rows = []
    for a in ('SR', 'EH', 'HOLD', 'SP'):
        for n in range(1, NMAX + 1):
            if a == 'EH' and n == 1:
                continue
            for s in ['全部'] + sorted(x['sector'].unique().to_list()):
                g = x if s == '全部' else x.filter(pl.col('sector') == s)
                w = g.filter(pl.col(f'{a}{n}').is_finite()).group_by('mi').agg(pl.col(f'{a}{n}').mean().alias('r'), pl.len().alias('k')).filter(pl.col('k') >= 2).sort('mi')
                r = w['r'].to_numpy()
                rows.append({'收益': a, 'n': n, '板块': s, '年化%': ann(r, n), 't': tstat(r, n), '年化波动%': float(np.nanstd(r, ddof=1) * np.sqrt(6 / n) * 100),
                             '期数': len(r), '平均品种数': float(w['k'].mean())})
    return pl.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    P = returns(periods(contracts()))
    P = add_signals(P, monthly())
    P.write_parquet(os.path.join(OUT, 'periods.parquet'))
    t1 = table1(P)
    t1.write_parquet(os.path.join(OUT, 'table1.parquet'))
    srt, tst = [], []
    for tag, Q in [('全样本', P), ('2005–2014', P.filter(pl.col('mi') < 2015 * 12)), ('2015–2026', P.filter(pl.col('mi') >= 2015 * 12))]:
        a, b = tables(Q, tag)
        srt.append(a)
        tst.append(b)
    # 稳健性：进场所用合约持仓 ≥ 100 手
    R = P.with_columns(*[pl.when((pl.col('oi1') >= 100) & (pl.col(f'oi{n}') >= 100)).then(pl.col(f'{a}{n}')).alias(f'{a}{n}')
                         for a in ('SR', 'EH', 'HOLD', 'SP') for n in range(1, NMAX + 1)])
    a, b = tables(R, '持仓≥100')
    srt.append(a)
    tst.append(b)
    S, T = pl.concat(srt), pl.concat(tst)
    S.write_parquet(os.path.join(OUT, 'sorts.parquet'))
    T.write_parquet(os.path.join(OUT, 'tests.parquet'))
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(30)
    print(t1.filter(pl.col('板块') == '全部'))
    f = S.filter((pl.col('样本') == '全样本') & pl.col('收益').is_in(['SR', 'EH']))
    print(f.select('变量', '收益', 'n', '期数', *[pl.col(c).round(2) for c in ('P1', 'P2', 'P3', 'P4', 'P4-P1', 'P4-P1_t')]))
    print(T.filter((pl.col('样本') == '全样本') & ~pl.col('模型').str.starts_with('VI'))
          .select('排序', '收益', 'n', '模型', pl.col('平均|α|').round(2), pl.col('p').round(3), pl.col('平均R2').round(2)))
