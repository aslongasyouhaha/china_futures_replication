"""论文结论部分提出的实证方向：有平仓压力时，近月相对次月的偏离是否更大？

中国规则提供天然对照：
  压力月：交割月的前一个月（个人客户须在月底前平仓）
  对照月：再往前一个月（同一对合约，同样用“窗口开始前近月/次月比值不变”的基准，但没有强制退出）
度量（不依赖模型）：窗口内近月相对基准的最大绝对偏离 max|ln(F_obs/F_lic)|、窗口末的偏离、窗口内近月持仓变化。
另：2020 年 4 月上海原油（与 WTI 跌成负数同月）的路径。
"""
import glob
import os

import numpy as np
import polars as pl

import china_events as CE

OUT = CE.OUT


def nw_t(x):
    x = np.asarray(x, float)
    n = len(x)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xc = x - x.mean()
    s = xc @ xc / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (xc[l:] @ xc[:-l]) / n
    return x.mean() / np.sqrt(s / n)


def window_stats(px, sym, s2, w_days, t0):
    if (sym, t0) not in px or (s2, t0) not in px:
        return None
    lam = px[(sym, t0)][0] / px[(s2, t0)][0]
    ds = [np.log(px[(sym, t)][0] / (lam * px[(s2, t)][0])) for t in w_days if (sym, t) in px and (s2, t) in px]
    if len(ds) < 5:
        return None
    oi0 = px[(sym, t0)][1]
    oi1 = px.get((sym, w_days[-1]), (None, None))[1]
    return dict(max_abs=float(np.max(np.abs(ds))), end=float(ds[-1]), oi_change=(oi1 / oi0 - 1) if oi0 and oi1 is not None else None)


def scan(p, d):
    days = d['trading_day'].unique().sort().to_list()
    pos = {x: i for i, x in enumerate(days)}
    px = {(s, t): (v, oi) for s, t, v, oi in d.select('symbol', 'trading_day', 'px', 'open_interest').iter_rows()}
    by_sym = {k[0]: v.sort('trading_day') for k, v in d.partition_by(['symbol'], as_dict=True).items()}
    cms = d.select('symbol', 'cm').unique().sort('cm').rows()
    out = []
    for k, (sym, cm) in enumerate(cms):
        g = by_sym[sym]
        wins = {lab: g.filter(pl.col('now') == cm - off)['trading_day'].to_list() for lab, off in (('压力月', 1), ('对照月', 2))}
        if any(len(w) < 5 for w in wins.values()):
            continue
        t0s = {lab: days[pos[w[0]] - 1] for lab, w in wins.items() if pos[w[0]] > 0}
        if len(t0s) < 2 or (sym, t0s['对照月']) not in px or px[(sym, t0s['对照月'])][1] < CE.MIN_OI_NEAR:
            continue
        nxt = [s for s, c in cms[k + 1:] if all((s, t) in px and px[(s, t)][1] >= CE.MIN_OI_NEXT for t in t0s.values())]
        if not nxt:
            continue
        row = {'product': p, 'symbol': sym, 'next': nxt[0], 'cm': cm}
        ok = True
        for lab, w in wins.items():
            st = window_stats(px, sym, nxt[0], w, t0s[lab])
            if st is None:
                ok = False
                break
            row.update({f'{lab}_{a}': v for a, v in st.items()})
        if ok:
            out.append(row)
    return out


def main():
    rows = []
    for f in sorted(glob.glob(f'{CE.DATA}/*/*/daily_contracts.parquet')):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in CE.EXCLUDE:
            continue
        rows += scan(p, CE.load(f))
    t = pl.DataFrame(rows)
    t.write_parquet(os.path.join(OUT, 'exit_vs_placebo.parquet'))
    print(f'同时有压力月与对照月的合约到期：{t.height} 个（{t["product"].n_unique()} 个品种）')
    diff = (t['压力月_max_abs'] - t['对照月_max_abs']).to_numpy()
    print(f"窗口内最大绝对偏离：压力月 中位数 {t['压力月_max_abs'].median()*100:.2f}%  对照月 中位数 {t['对照月_max_abs'].median()*100:.2f}%；"
          f"压力月更大的比例 {(diff > 0).mean():.1%}")
    for q in (0.9, 0.99):
        print(f"  {int(q*100)}% 分位：压力月 {t['压力月_max_abs'].quantile(q)*100:.1f}%  对照月 {t['对照月_max_abs'].quantile(q)*100:.1f}%")
    for th in (0.05, 0.10, 0.20):
        print(f"  偏离超过 {th:.0%} 的到期：压力月 {(t['压力月_max_abs'] > th).sum()} 个，对照月 {(t['对照月_max_abs'] > th).sum()} 个")
    print(f"窗口内近月持仓变化（中位数）：压力月 {t['压力月_oi_change'].median():.1%}，对照月 {t['对照月_oi_change'].median():.1%}")
    # 偏离方向：压力月末，近月相对基准是被压低还是抬高
    e = t['压力月_end'].to_numpy()
    print(f"压力月末偏离方向：被压低 {(e < 0).mean():.1%}，被抬高 {(e > 0).mean():.1%}；平均 {e.mean()*100:+.2f}%（t={nw_t(e):.2f}）")
    # 按品种：压力月比对照月偏离更大的程度
    bp = (t.with_columns((pl.col('压力月_max_abs') - pl.col('对照月_max_abs')).alias('d'))
           .group_by('product').agg(pl.len().alias('n'), (pl.col('d').median() * 100).alias('压力月多出的偏离%（中位数）'),
                                    (pl.col('压力月_max_abs').median() * 100).alias('压力月偏离%'))
           .filter(pl.col('n') >= 20).sort('压力月多出的偏离%（中位数）', descending=True))
    bp.write_parquet(os.path.join(OUT, 'exit_by_product.parquet'))
    pl.Config.set_tbl_rows(20)
    print(bp.head(10).with_columns(pl.exclude('product', 'n').round(2)))
    print(bp.tail(5).with_columns(pl.exclude('product', 'n').round(2)))
    # 持仓下降越多，偏离越大？
    q = t.drop_nulls('压力月_oi_change').with_columns(pl.col('压力月_oi_change').qcut(4, labels=['降得最少', '2', '3', '降得最多']).alias('g'))
    print(q.group_by('g').agg(pl.len(), (pl.col('压力月_max_abs').median() * 100).alias('偏离%中位数')).sort('g'))

    # 2020 年 4 月：上海原油
    sc = CE.load(f'{CE.DATA}/INE/SC/daily_contracts.parquet').filter(
        pl.col('trading_day').is_between(pl.date(2020, 3, 25), pl.date(2020, 4, 30)) & pl.col('symbol').is_in(['SC2005', 'SC2006']))
    w = sc.pivot(index='trading_day', columns='symbol', values='px').sort('trading_day')
    oi = sc.pivot(index='trading_day', columns='symbol', values='open_interest').sort('trading_day')
    print('\n== 2020 年 4 月上海原油 SC2005 / SC2006 ==')
    print(w.join(oi, on='trading_day', suffix='_持仓').with_columns((pl.col('SC2005') / pl.col('SC2006')).round(3).alias('比值')))


if __name__ == '__main__':
    main()
