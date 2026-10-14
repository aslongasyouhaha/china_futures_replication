"""论文方法（次近月）与中国化口径（主力合约）的对比分析：
  1. 论文方法下持有合约与主力的差别（是否同一合约、流动性、两种期限结构的排序相关）
  2. 2×2 拆解：排序信号（论文 / 主力）× 持有合约（次近月 / 主力），附稳健性
  3. 近月超额（次近月收益 − 主力收益）能被什么预测
  4. carry 与动量能否互相解释（对 AVG 与另一个因子回归后的 α）
  5. TSMOM（主力合约、商品）拆成截面部分与整体方向部分
  6. 主力合约上的截面动量：组合构造方式 × 回看期；按多空分歧程度分组
  7. 主力合约过去 12 个月后复权涨幅 = 价格变化（未复权）+ 换月展期收益，各自的预测力
  8. 独立检验：12 个月动量 vs 过去 12 个月平均贴水（carry1-12，直接由期限结构算出）
"""
import glob
import os

import numpy as np
import polars as pl

import factors as FA
import pricing as PR

OUT = FA.OUT


def nw_t(x):
    x = np.asarray(x, float)
    return x.mean() / np.sqrt(PR.nw_cov(x)[0, 0] / len(x))


def joint():
    lp = pl.read_parquet(f'{OUT}/panel_paper.parquet')
    mp = pl.read_parquet(f'{OUT}/panel_main.parquet')
    return lp.select('product', 'm', 'sector', pl.col('ret_next').alias('r_lit'), pl.col('carry').alias('c_lit'), pl.col('mom6').alias('m_lit')).join(
        mp.select('product', 'm', pl.col('ret_next').alias('r_main'), pl.col('carry').alias('c_main'), pl.col('mom6').alias('m_main')),
        on=['product', 'm'])


def diagnostics():
    lc = pl.read_parquet(f'{OUT}/paper_contracts.parquet')
    x = pl.read_parquet(f'{OUT}/panel_paper.parquet').join(lc.select('product', 'm', 'f1', 'f1_vol', 'gap'), on=['product', 'm'])
    parts = []
    for f in glob.glob(f'{FA.T.DATA}/*/*/daily_main.parquet'):
        p = os.path.basename(os.path.dirname(f))
        parts.append(pl.read_parquet(f, columns=['trading_day', 'symbol', 'volume']).with_columns(pl.lit(p).alias('product')))
    ms = (pl.concat(parts).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
            .group_by(['product', 'm']).agg(pl.all().sort_by('trading_day').last()))
    j = x.join(ms.select('product', 'm', pl.col('symbol').alias('main'), pl.col('volume').alias('main_vol')), on=['product', 'm'], how='left')
    jj = joint()
    rc = jj.group_by('m').agg(pl.corr(pl.col('c_lit').rank(), pl.col('c_main').rank()).alias('rho'))
    out = {'持有合约即主力的比例': float((j['f1'] == j['main']).mean()),
           '持有合约成交量 ÷ 主力成交量（中位数）': float((j['f1_vol'] / j['main_vol']).median()),
           '两种期限结构的截面排序相关（中位数）': float(rc['rho'].median()),
           '贴水 / 升水判断一致的比例': float((np.sign(jj['c_lit']) == np.sign(jj['c_main'])).mean())}
    print('== 论文方法与主力合约的差别 ==')
    for k, v in out.items():
        print(f'  {k}：{v:.2f}')
    pl.DataFrame([{'项目': k, '值': v} for k, v in out.items()]).write_parquet(f'{OUT}/cmp_diagnostics.parquet')


def decompose():
    j = joint()
    lo, hi = j['r_lit'].quantile(0.01), j['r_lit'].quantile(0.99)
    lo2, hi2 = j['r_main'].quantile(0.01), j['r_main'].quantile(0.99)
    cases = {'全部': j, '收益截尾 1%': j.with_columns(pl.col('r_lit').clip(lo, hi), pl.col('r_main').clip(lo2, hi2)),
             '去掉煤焦': j.filter(~pl.col('product').is_in(['J', 'JM', 'ZC'])),
             '2019 年前': j.filter(pl.col('m') < pl.date(2020, 1, 1)), '2020 年起': j.filter(pl.col('m') >= pl.date(2020, 1, 1))}
    rows = []
    for case, jj in cases.items():
        for sig in ('lit', 'main'):
            for ret in ('lit', 'main'):
                z = jj.select('product', 'm', 'sector', pl.col(f'r_{ret}').alias('ret_next'), pl.col(f'c_{sig}').alias('carry'),
                              pl.col(f'm_{sig}').alias('mom6'))
                f, _, _ = FA.build(z)
                sr = lambda c: float(f[c].mean() / f[c].std() * np.sqrt(12))
                rows.append({'样本': case, '信号': '论文方法' if sig == 'lit' else '主力合约', '持有合约': '次近月' if ret == 'lit' else '主力',
                             '月数': f.height, 'CARRY%': float(f['CARRY'].mean() * 1200), 'CARRY夏普': sr('CARRY'),
                             'CMOM%': float(f['CMOM'].mean() * 1200), 'CMOM夏普': sr('CMOM')})
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/cmp_2x2.parquet')
    print('\n== 2×2 拆解（两种口径都有数据的品种-月）==')
    print(t.filter(pl.col('样本') == '全部').with_columns(pl.col('CARRY%', 'CARRY夏普', 'CMOM%', 'CMOM夏普').round(2)))


def near_excess():
    j = joint().with_columns((pl.col('r_lit') - pl.col('r_main')).alias('d'),
                             ((1 + pl.col('m_lit')).log() - (1 + pl.col('m_main')).log()).alias('past_d'))
    names = {'c_lit': '当前贴水程度（ln(F0/F1)）', 'past_d': '过去 6 个月的近月超额', 'm_main': '过去 6 个月主力涨幅'}
    rows = []
    for y, ylab in (('d', '下月近月超额'), ('r_main', '下月主力收益')):
        for xs in (['c_lit'], ['past_d'], ['m_main'], ['c_lit', 'past_d', 'm_main']):
            co = []
            for (m,), g in j.drop_nulls([y] + xs).group_by(['m']):
                if g.height < 10:
                    continue
                X = np.column_stack([np.ones(g.height)] + [((g[c] - g[c].mean()) / g[c].std()).to_numpy() for c in xs])
                co.append(np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0][1:])
            co = np.array(co)
            for k, c in enumerate(xs):
                rows.append({'因变量': ylab, '回归': '单变量' if len(xs) == 1 else '三者一起', '变量': names[c],
                             '系数%': float(co[:, k].mean() * 100), 't': float(nw_t(co[:, k])), 'T': len(co)})
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/cmp_near_excess.parquet')
    print('\n== 近月超额的预测（截面 Fama-MacBeth，每 1 个截面标准差，月%）==')
    for r in t.iter_rows(named=True):
        print(f"  {r['因变量']} {r['回归']:4s} {r['变量']:14s} {r['系数%']:+.2f}%  t={r['t']:.2f}")


def spanning():
    rows = []
    for v, lab in (('main', '主力合约'), ('paper', '论文方法（次近月）')):
        d = pl.read_parquet(f'{OUT}/factors_{v}.parquet')
        for y, oth in (('CMOM', 'CARRY'), ('CARRY', 'CMOM')):
            Y = d[y].to_numpy()
            X = np.column_stack([np.ones(len(Y)), d['AVG'].to_numpy(), d[oth].to_numpy()])
            b = np.linalg.lstsq(X, Y, rcond=None)[0]
            e = Y - X @ b
            XtX = np.linalg.inv(X.T @ X)
            V = XtX @ (PR.nw_cov(X * e[:, None]) * len(Y)) @ XtX
            rows.append({'口径': lab, '因子': y, '原始年化%': float(Y.mean() * 1200), 'α年化%': float(b[0] * 1200),
                         't(α)': float(b[0] / np.sqrt(V[0, 0])), '另一因子β': float(b[2]), 'R2': float(1 - e.var() / Y.var())})
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/cmp_spanning.parquet')
    print('\n== carry 与动量能否互相解释（对 AVG + 另一个因子回归）==')
    print(t.with_columns(pl.exclude('口径', '因子').round(2)))


def tsmom_decomp():
    """TSMOM（主力合约、商品，品种池同上）= 截面部分（仓位减去当月平均仓位）+ 整体方向部分（平均仓位 × 商品平均收益）。"""
    me = FA.main_monthly().with_columns(
        (pl.col('close_adj').shift(-1).over('product') / pl.col('close_adj') - 1).alias('r'),
        *[(pl.col('close_adj') / pl.col('close_adj').shift(k).over('product') - 1).alias(f'p{k}') for k in (6, 12)])
    u = me.join(pl.read_parquet(f'{OUT}/panel_main.parquet').select('product', 'm'), on=['product', 'm']).drop_nulls(['r', 'p6', 'p12', 'sigma'])
    u = u.filter(pl.col('sigma') > 0)
    rows = []
    for k in (12, 6):
        for wname in ('等权 ±1', '40%/σ（同 TSMOM 论文）'):
            parts = []
            for (m,), g in u.group_by(['m']):
                if g.height < 10:
                    continue
                r = g['r'].to_numpy()
                w = np.sign(g[f'p{k}'].to_numpy())
                if wname != '等权 ±1':
                    w = w * FA.T.TARGET_VOL / g['sigma'].to_numpy()
                net = w.mean()
                parts.append((np.mean(w * r), np.mean((w - net) * r), net * r.mean()))
            a = np.array(parts)
            rows.append({'回看月数': k, '仓位': wname, 'TSMOM%': a[:, 0].mean() * 1200, 't(TSMOM)': nw_t(a[:, 0]),
                         '截面部分%': a[:, 1].mean() * 1200, 't(截面)': nw_t(a[:, 1]), '整体方向%': a[:, 2].mean() * 1200, 't(方向)': nw_t(a[:, 2])})
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/cmp_tsmom.parquet')
    print('\n== TSMOM 拆解（主力合约、商品）==')
    print(t.with_columns(pl.exclude('回看月数', '仓位').round(2)))


def construction():
    """主力合约、同一品种池：不同组合构造方式与回看期下的截面动量（多空两边各 1 元名义）；按当月多空分歧程度分组。"""
    me = FA.main_monthly().with_columns(
        (pl.col('close_adj').shift(-1).over('product') / pl.col('close_adj') - 1).alias('r'),
        *[(pl.col('close_adj') / pl.col('close_adj').shift(k).over('product') - 1).alias(f'p{k}') for k in (6, 12)])
    u = me.join(pl.read_parquet(f'{OUT}/panel_main.parquet').select('product', 'm'), on=['product', 'm']).drop_nulls(['r', 'p6', 'p12'])
    rows, split = [], []
    for k in (6, 12):
        res = {n: [] for n in ('两端各 5 个（论文 M5）', '两端各 20%', '两端各 1/3', '全部按排名加权')}
        sgn, disp = [], []
        for (m,), g in u.group_by(['m']):
            n = g.height
            if n < 10:
                continue
            r = g['r'].to_numpy()
            s = g[f'p{k}'].to_numpy()
            o = np.argsort(-s, kind='stable')
            res['两端各 5 个（论文 M5）'].append(r[o[:5]].mean() - r[o[-5:]].mean())
            q = n // 5
            res['两端各 20%'].append(r[o[:q]].mean() - r[o[-q:]].mean())
            q = n // 3
            res['两端各 1/3'].append(r[o[:q]].mean() - r[o[-q:]].mean())
            rk = s.argsort().argsort() - (n - 1) / 2
            res['全部按排名加权'].append(rk @ r / np.abs(rk).sum() * 2)
            w = np.sign(s) - np.sign(s).mean()
            gross = np.abs(w).sum()
            sgn.append(w @ r / gross * 2 if gross > 0 else 0.0)
            disp.append(np.abs(w).mean())
        for name, x in res.items():
            x = np.array(x)
            rows.append({'回看月数': k, '构造': name, '年化%': x.mean() * 1200, '年化波动%': x.std() * np.sqrt(12) * 100,
                         '夏普': x.mean() / x.std() * np.sqrt(12), 't': nw_t(x)})
        sgn, disp = np.array(sgn), np.array(disp)
        hi = disp > np.median(disp)
        for lab, sel in (('多空分歧大的月份', hi), ('多空分歧小的月份', ~hi)):
            split.append({'回看月数': k, '月份': lab, '年化%': sgn[sel].mean() * 1200, 't': nw_t(sgn[sel]), '月数': int(sel.sum())})
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/cmp_construction.parquet')
    sp = pl.DataFrame(split)
    sp.write_parquet(f'{OUT}/cmp_dispersion.parquet')
    print('\n== 主力合约上的截面动量：构造方式 × 回看期 ==')
    print(t.with_columns(pl.exclude('回看月数', '构造').round(2)))
    print('== 按当月多空分歧程度分组（符号加权、归一）==')
    print(sp.with_columns(pl.col('年化%', 't').round(2)))


def roll_split():
    """后复权涨幅 = 未复权主力价格的变化 + 换月展期收益（换月时新旧合约价差）。分别排序看预测力。"""
    parts = []
    for f in glob.glob(f'{FA.T.DATA}/*/*/daily_main.parquet'):
        parts.append(pl.read_parquet(f, columns=['trading_day', 'close']).with_columns(pl.lit(os.path.basename(os.path.dirname(f))).alias('product')))
    raw = (pl.concat(parts).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
             .group_by(['product', 'm']).agg(pl.col('close').sort_by('trading_day').last()))
    me = FA.main_monthly().join(raw, on=['product', 'm']).sort(['product', 'm']).with_columns(
        (pl.col('close_adj').shift(-1).over('product') / pl.col('close_adj') - 1).alias('r'),
        (pl.col('close_adj').log() - pl.col('close_adj').shift(12).over('product').log()).alias('adj12'),
        (pl.col('close').log() - pl.col('close').shift(12).over('product').log()).alias('raw12'),
        (pl.col('m').shift(12).over('product') == pl.col('m').dt.offset_by('-12mo')).alias('ok'))
    x = me.filter(pl.col('ok')).join(pl.read_parquet(f'{OUT}/panel_main.parquet').select('product', 'm', 'carry'), on=['product', 'm'])
    x = x.drop_nulls(['r', 'adj12', 'raw12', 'carry']).with_columns((pl.col('adj12') - pl.col('raw12')).alias('roll12'))
    names = {'adj12': '后复权涨幅（TSMOM 用的）', 'raw12': '价格变化（未复权）', 'roll12': '换月展期收益', 'carry': '当前 carry'}
    rows = []
    for sig, lab in names.items():
        ts, xs = [], []
        for (m,), g in x.group_by(['m']):
            if g.height < 10:
                continue
            r = g['r'].to_numpy()
            s = g[sig].to_numpy()
            ts.append(np.mean(np.sign(s) * r))
            rk = s.argsort().argsort() - (len(s) - 1) / 2
            xs.append(rk @ r / np.abs(rk).sum() * 2)
        rows.append({'信号': lab, '时间序列%': np.mean(ts) * 1200, 't(时间序列)': nw_t(ts), '截面%': np.mean(xs) * 1200, 't(截面)': nw_t(xs)})
    co = []
    for (m,), g in x.group_by(['m']):
        if g.height < 10:
            continue
        X = np.column_stack([np.ones(g.height)] + [((g[c] - g[c].mean()) / g[c].std()).to_numpy() for c in ('raw12', 'roll12', 'carry')])
        co.append(np.linalg.lstsq(X, g['r'].to_numpy(), rcond=None)[0][1:])
    co = np.array(co)
    fm = [{'变量': names[c], '系数%': float(co[:, i].mean() * 100), 't': float(nw_t(co[:, i]))} for i, c in enumerate(('raw12', 'roll12', 'carry'))]
    t = pl.DataFrame(rows)
    t.write_parquet(f'{OUT}/cmp_roll_split.parquet')
    pl.DataFrame(fm).write_parquet(f'{OUT}/cmp_roll_split_fm.parquet')
    print('\n== 主力合约过去 12 个月：价格变化 vs 换月展期收益 ==')
    print(t.with_columns(pl.exclude('信号').round(2)))
    print('三者一起的截面回归（每 1 标准差，月%）：', [(r['变量'], round(r['系数%'], 2), round(r['t'], 2)) for r in fm])


def mom_vs_carry12():
    me = FA.main_monthly().with_columns(
        (pl.col('close_adj').shift(-1).over('product') / pl.col('close_adj') - 1).alias('r'),
        (pl.col('close_adj').log() - pl.col('close_adj').shift(12).over('product').log()).alias('mom12'),
        (pl.col('m').shift(12).over('product') == pl.col('m').dt.offset_by('-12mo')).alias('ok'))
    cp = pl.read_parquet(FA.CARRY_PANEL).sort(['product', 'trading_day']).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
    cp = cp.with_columns(pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('carry12'))
    x = (me.filter('ok').join(pl.read_parquet(f'{OUT}/panel_main.parquet').select('product', 'm'), on=['product', 'm'])
           .join(cp.select('product', 'm', 'carry12'), on=['product', 'm']).drop_nulls(['r', 'mom12', 'carry12']))
    M, Cc = [], []
    for (m,), g in x.group_by(['m'], maintain_order=True):
        if g.height < 10:
            continue
        r = g['r'].to_numpy()
        for s_, L in ((g['mom12'].to_numpy(), M), (g['carry12'].to_numpy(), Cc)):
            rk = s_.argsort().argsort() - (len(s_) - 1) / 2
            L.append(rk @ r / np.abs(rk).sum() * 2)
    M, Cc = np.array(M), np.array(Cc)
    X = np.column_stack([np.ones(len(M)), Cc])
    b = np.linalg.lstsq(X, M, rcond=None)[0]
    e = M - X @ b
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (PR.nw_cov(X * e[:, None]) * len(M)) @ XtX
    out = {'截面相关': float(x.group_by('m').agg(pl.corr('mom12', 'carry12'))['mom12'].mean()),
           '动量%': M.mean() * 1200, 't动量': nw_t(M), 'carry12%': Cc.mean() * 1200, 't_carry12': nw_t(Cc),
           'α%': b[0] * 1200, 'tα': b[0] / np.sqrt(V[0, 0]), 'R2': 1 - e.var() / M.var()}
    pl.DataFrame([out]).write_parquet(f'{OUT}/cmp_mom_vs_carry12.parquet')
    print('\n== 12 个月动量 vs 过去 12 个月平均贴水（排名加权）==', {k: round(v, 2) for k, v in out.items()})


if __name__ == '__main__':
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(12)
    diagnostics()
    decompose()
    near_excess()
    spanning()
    tsmom_decomp()
    construction()
    roll_split()
    mom_vs_carry12()
