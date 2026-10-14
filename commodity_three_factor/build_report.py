"""生成商品三因子复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 读取。输出: report.html
第一部分：按论文方法复现（次近月合约、ln(F1/F0)）；第二部分：换成中国习惯的主力合约后有何变化、为什么。"""
import html
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pricing as PR  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = PR.OUT
PAPER_FAC = {'AVG': (6.27, 14.32, 0.44), 'CARRY': (16.34, 22.26, 0.73), 'CMOM': (16.11, 26.28, 0.61)}
PAPER_FM = {'三因子': ('96%', '0.22', '0.22'), '去掉动量': ('83%', '0.01', '0.15'), '去掉 carry': ('68%', '0.00', '0.00'), '只有 AVG': ('—', '—', '—')}
NAMES = {'AVG': '平均因子 AVG', 'CARRY': 'Carry 因子 C5', 'CMOM': '动量因子 M5'}


def rd(name, v):
    return pl.read_parquet(f'{O}/{name}_{v}.parquet') if v else pl.read_parquet(f'{O}/{name}.parquet')


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def cum_svg(fac):
    W, Hh, left, y0, ph = 760, 320, 50, 20, 250
    m = fac['m'].to_list()
    series = {k: np.cumsum(np.log1p(fac[k].to_numpy())) for k in ('CMOM', 'CARRY', 'AVG')}
    lo = min(0, min(v.min() for v in series.values())) - 0.1
    hi = max(v.max() for v in series.values()) + 0.1
    n = len(m)
    xs = lambda i: left + i / (n - 1) * (W - left - 120)
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="三个因子的累计收益">']
    for t in np.arange(np.ceil(lo), hi, 1.0):
        s.append(f'<line x1="{left}" x2="{W-120}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="{"axis" if abs(t) < 1e-9 else "grid"}"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{np.exp(t):.1f}×</text>')
    for i, d in enumerate(m):
        if d.month == 1 and d.year % 4 == 2:
            s.append(f'<text x="{xs(i):.1f}" y="{y0+ph+18}" text-anchor="middle" class="sub">{d.year}</text>')
    style = {'CARRY': 'accent', 'CMOM': 'fg', 'AVG': 'muted'}
    ends = []
    for k, v in series.items():
        pts = ' '.join(f'{xs(i):.1f},{yv(y):.1f}' for i, y in enumerate(v))
        s.append(f'<polyline points="{pts}" class="ln {style[k]}" stroke-width="{2.2 if k != "AVG" else 1.6}"/>')
        ends.append([yv(v[-1]), k])
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 15)
    for y, k in ends:
        s.append(f'<text x="{W-114}" y="{y+4:.1f}" class="lbl {style[k]}t">{NAMES[k]}</text>')
    s.append('</svg>')
    return ''.join(s)


def scatter_svg(pe, models):
    W, Hh, left, y0, ph, pw, gap = 760, 340, 56, 20, 270, 300, 90
    vals = pe.filter(pl.col('模型').is_in(models)).select('实际%', '模型预测%').to_numpy()
    lo, hi = vals.min() - 1, vals.max() + 1
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="实际与模型预测的平均收益">']
    for k, name in enumerate(models):
        x0 = left + k * (pw + gap)
        xs = lambda v: x0 + (v - lo) / (hi - lo) * pw
        yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
        s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" class="frame"/>'
                 f'<line x1="{xs(lo):.1f}" y1="{yv(lo):.1f}" x2="{xs(hi):.1f}" y2="{yv(hi):.1f}" class="ref dash"/>'
                 f'<text x="{x0+pw/2}" y="{y0+14}" text-anchor="middle" class="lbl">{name}</text>')
        for t in range(int(np.ceil(lo / 5) * 5), int(hi) + 1, 5):
            s.append(f'<text x="{x0-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t}%</text>'
                     f'<text x="{xs(t):.1f}" y="{y0+ph+16}" text-anchor="middle" class="sub">{t}%</text>')
        for r in pe.filter(pl.col('模型') == name).iter_rows(named=True):
            cls = 'accent' if r['组合'].startswith('P') else 'fg' if r['组合'].startswith('Q') else 'muted'
            s.append(f'<circle cx="{xs(r["模型预测%"]):.1f}" cy="{yv(r["实际%"]):.1f}" r="5" class="dot {cls}">'
                     f'<title>{r["组合"]}：实际 {r["实际%"]:.1f}%，预测 {r["模型预测%"]:.1f}%</title></circle>'
                     f'<text x="{xs(r["模型预测%"])+7:.1f}" y="{yv(r["实际%"])+4:.1f}" class="sm">{html.escape(r["组合"])}</text>')
    s.append(f'<text x="{W/2}" y="{Hh-4}" text-anchor="middle" class="sub">横轴：模型预测的年化收益；纵轴：实际年化收益；虚线 = 完全定价。蓝 = carry 组合，黑 = 动量组合，灰 = 板块</text>')
    s.append('</svg>')
    return ''.join(s)


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--muted:#a9a69b;--accent:#2f6fd0;--axis:#9a978e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:24px 0 6px}.part{font-size:22px;margin:56px 0 0;padding:10px 14px;background:var(--card);border-left:4px solid var(--accent);border-radius:6px}
.meta{color:var(--sub);font-size:13px;margin-bottom:12px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--bg)}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}figcaption{color:var(--sub);font-size:12px;margin-top:6px}
figure .title{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--line)}svg .axis{stroke:var(--axis)}svg .ref{stroke:var(--axis)}svg .dash{stroke-dasharray:4 3}svg .frame{fill:none;stroke:var(--line)}
svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}svg .ln.fg{stroke:var(--fg)}svg .ln.muted{stroke:var(--muted)}
svg .dot.accent{fill:var(--accent)}svg .dot.fg{fill:var(--fg)}svg .dot.muted{fill:var(--muted)}
svg text{fill:var(--fg);font-size:12px}svg .sm{font-size:10px;fill:var(--sub)}svg .sub{fill:var(--sub)}svg .accentt{fill:var(--accent);font-weight:600}svg .fgt{fill:var(--fg);font-weight:600}svg .mutedt{fill:var(--sub)}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    V = 'paper'
    fac = rd('factors', V).sort('m')
    D = {r['组合']: r for r in rd('describe', V).iter_rows(named=True)}
    DM = {r['组合']: r for r in rd('describe', 'main').iter_rows(named=True)}
    F = {r['模型']: r for r in rd('fama_macbeth', V).iter_rows(named=True)}
    FM = {r['模型']: r for r in rd('fama_macbeth', 'main').iter_rows(named=True)}
    pe = rd('pricing_errors', V)
    ts = rd('time_series', V)
    pca = rd('pca', V)
    sub = rd('subperiod', V)
    ets = rd('econ_ts', V)
    efm = rd('econ_fm', V)
    ov = rd('overlap', V)['overlap'].to_numpy()
    ovm = rd('overlap', 'main')['overlap'].to_numpy()
    bys = rd('by_state', V)
    cnt = rd('member_counts', V)
    diag = {r['项目']: r['值'] for r in rd('cmp_diagnostics', None).iter_rows(named=True)}
    x2 = rd('cmp_2x2', None)
    ne = rd('cmp_near_excess', None)
    spn = rd('cmp_spanning', None)
    tsd = rd('cmp_tsmom', None)
    d = fac.join(rd('test_portfolios', V), on='m').drop_nulls(PR.TEST)
    corr = np.corrcoef(d['CARRY'], d['CMOM'])[0, 1]
    T = F['三因子']['T']
    p = []
    p.append('<h1>中国商品期货三因子模型复现报告</h1><div class="meta">复现论文：Gurdip Bakshi, Xiaohui Gao &amp; Alberto G. Rossi，'
             '《Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns》，<i>Management Science</i>，2019 · '
             f'中国商品期货 2006–2026 · 不计交易成本（同论文）</div>')
    p.append(f'<p class="lead">按论文方法复现，中国商品的结果与论文基本一致：carry 与动量因子收益都显著（夏普 {D["CARRY"]["夏普"]:.2f} 与 {D["CMOM"]["夏普"]:.2f}，论文 0.73 与 0.61），'
             f'去掉其中任何一个，模型都被拒绝，两者都是解释商品截面收益所必需的。但论文给出的经济解释（carry 对应股市波动率风险、动量对应投机活动）在中国找不到支持。'
             f'换成中国市场交易集中的主力合约后，carry 依然显著，按论文方式（6 个月回看）构造的动量因子却不再显著，原因是中国商品的价格趋势主要体现在临近交割的近月合约上；若改看 12 个月，主力合约上的“动量”显著，但大部分与过去一年的 carry 重叠，扣掉后剩下的价格动量不显著。</p>')
    p.append(f'<div class="kpis"><div class="kpi"><b>{D["CARRY"]["夏普"]:.2f} / {D["CMOM"]["夏普"]:.2f}</b><span>carry / 动量因子夏普（论文 0.73 / 0.61）</span></div>'
             f'<div class="kpi"><b>p = {F["去掉动量"]["p_χ2_NW"]:.2f}</b><span>去掉动量：模型被拒绝（论文 0.01）</span></div>'
             f'<div class="kpi"><b>p = {F["去掉 carry"]["p_χ2_NW"]:.2f}</b><span>去掉 carry：模型被拒绝（论文 0.00）</span></div>'
             f'<div class="kpi"><b>{DM["CMOM"]["夏普"]:.2f}</b><span>换成主力合约后的动量夏普（不显著）</span></div></div>')

    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['样本', '美国 29 个商品期货，1970-01 至 2011-09，每月 15~28 个', f'中国商品期货（不含中金所、月均价合约、集运指数），2006 至 2026-08，可用于定价检验 {T} 个月，每月 10~66 个'],
        ['持有合约与收益', '月末持有“下个月不会进入交割的次近月合约”，持有到下月末', '同左：排在最近月之后、下个月不进入交割月的第一个合约（隔月挂牌品种同样处理）；用结算价'],
        ['期限结构', 'ln(y) = ln(F₁/F₀)，次近月 ÷ 最近月，不年化', '同左'],
        ['因子', 'AVG（等权做多全部）；CARRY = C5（最贴水 5 个 − 最升水 5 个）；CMOM = M5（过去 6 个月最高 5 个 − 最低 5 个）', '同左；动量用同样滚动持有次近月的过去 6 个月收益'],
        ['测试组合', '4 个 carry 组合 + 5 个动量五分位 + 农产品、畜牧、金属（能源因历史太短排除），共 12 个', '4 + 5 + 农产品、金属，共 11 个：中国畜牧（鸡蛋 2013、生猪 2021）历史太短；能源化工的次近月多不活跃，27 个月整个板块没有合格品种，同论文排除'],
        ['截面检验', 'GMM / Fama-MacBeth（NW 与 Shanken），GLS 与 OLS R²，定价误差 χ²，HJ 距离；嵌套模型', '同左（Fama-MacBeth 的 λ 与一步 GMM 相同）；HJ 距离的 p 值按 Jagannathan-Wang 模拟'],
        ['时间序列检验', '各组合对三因子回归，α 联合检验', '同左（GRS 检验）'],
        ['经济解释', 'G20 股市波动率；CFTC 非商业持仓（投机度）；NBER 衰退期；多空腿成员', '沪深 300 股指期货波动率（2010 起）；成交量 ÷ 持仓量；商品市场涨跌状态；多空腿成员'],
        ['中国数据的必要调整', '—', '次近月当天须持仓 ≥ 500 手且有成交，否则价格不可靠；品种池要求主力 60 日平均成交 ≥ 1000 手、上市 ≥ 60 天']]))

    p.append('<div class="part">第一部分：按论文方法复现</div>')
    p.append('<h2>一、因子表现</h2>')
    p.append(f'<figure><div class="title">carry 与动量因子都稳定上升，平均因子基本走平</div>{cum_svg(fac)}'
             '<figcaption>来源：factors.py · 月度对数收益累计，纵轴为净值倍数（对数刻度）</figcaption></figure>')
    rows = []
    for k in ('AVG', 'CARRY', 'CMOM'):
        r = D[k]
        pa = PAPER_FAC[k]
        rows.append([NAMES[k], f"{r['年化均值%']:.1f}%", f"{r['95%下限']:.1f}% ~ {r['95%上限']:.1f}%", f"{r['年化波动%']:.1f}%",
                     f"<b>{r['夏普']:.2f}</b>", f"{r['t(NW)']:.2f}", f"{r['偏度']:.2f}", f"{pa[0]:.1f}% / {pa[1]:.1f}% / {pa[2]:.2f}"])
    p.append(table(['因子', '年化均值', '95% 置信区间（自助法）', '年化波动', '夏普', 't 值（NW）', '偏度', '论文：均值 / 波动 / 夏普'], rows))
    p.append(table(['腿宽', 'C1', 'C2', 'C3', 'C4', 'C5', 'M1', 'M2', 'M3', 'M4', 'M5'],
                   [['夏普'] + [f"{D[c]['夏普']:.2f}" for c in ('C1', 'C2', 'C3', 'C4', 'CARRY', 'M1', 'M2', 'M3', 'M4', 'CMOM')],
                    ['论文'] + ['0.19', '0.29', '0.44', '0.59', '0.73', '0.18', '0.40', '0.40', '0.50', '0.61']]))
    pc = {r['因子']: r for r in pca.iter_rows(named=True)}
    p.append(f"<p>与论文一样，腿越宽（做多空的品种越多）夏普越高。carry 收益主要来自多头腿（做多贴水 {D['CARRY_long']['年化均值%']:.1f}%，做空升水 {D['CARRY_short']['年化均值%']:.1f}%），"
             f"与论文一致；动量多头腿 {D['CMOM_long']['年化均值%']:.1f}%、空头腿 {D['CMOM_short']['年化均值%']:.1f}%。carry 与动量相关 {corr:.2f}（论文 0.27），"
             f"{(np.sign(d['CARRY']) != np.sign(d['CMOM'])).mean():.0%} 的月份符号相反（论文 41%）。"
             f"主成分：11 个组合的前三个主成分解释 {pc['AVG']['PC1解释']+pc['AVG']['PC2解释']+pc['AVG']['PC3解释']:.0%}，第一主成分几乎就是 AVG（相关 {pc['AVG']['PC1']:.2f}），"
             f"动量主要对应第二主成分（{pc['CMOM']['PC2']:.2f}），carry 也主要对应第二主成分（{pc['CARRY']['PC2']:.2f}），与第三主成分几乎无关（{pc['CARRY']['PC3']:.2f}）。"
             "论文中动量对应第二、三主成分，carry 单独对应第三主成分；中国的 carry 与动量在主成分上没有那么分得开。</p>")

    p.append('<h2>二、测试组合</h2>')
    p.append(table(['组合', '含义', '年化均值', '年化波动', '夏普'],
                   [[c, m, f"{D[c]['年化均值%']:.1f}%", f"{D[c]['年化波动%']:.1f}%", f"{D[c]['夏普']:.2f}"] for c, m in (
                       ('P1', '贴水最深的一半'), ('P2', '贴水较浅的一半'), ('P3', '升水较浅的一半'), ('P4', '升水最深的一半'),
                       ('Q1', '过去 6 个月最差 20%'), ('Q2', ''), ('Q3', ''), ('Q4', ''), ('Q5', '过去 6 个月最好 20%'),
                       ('农产品', ''), ('金属', '有色、贵金属、黑色'))]))
    p.append(f"<p>贴水组合（P1、P2，年化 {D['P1']['年化均值%']:.0f}%~{D['P2']['年化均值%']:.0f}%）明显高于升水组合（P3、P4，约 0）；"
             f"动量五分位从 Q1 的 {D['Q1']['年化均值%']:.1f}% 单调升到 Q5 的 {D['Q5']['年化均值%']:.1f}%。与论文相同，两组都有清晰的截面差异；"
             '不同的是 P1 并不高于 P2，最深贴水的一半并没有赚得更多。</p>')

    p.append('<h2>三、截面定价：carry 与动量缺一不可</h2>')
    rows = []
    for name in PR.MODELS:
        r = F[name]
        lam = '；'.join(f"{f} {r[f'λ_{f}']:.2f}%（p={r[f'p_{f}(NW)']:.2f}）" for f in PR.MODELS[name])
        pp = PAPER_FM[name]
        rows.append([name, lam, f"{r['R2_GLS']:.0%}", f"{r['p_χ2_NW']:.2f} / {r['p_χ2_Shanken']:.2f}", f"{r['HJ']:.3f}（p={r['p_HJ']:.2f}）",
                     f"{pp[0]} / {pp[1]} / {pp[2]}"])
    p.append(table(['模型', '风险溢价 λ（月度，NW p 值）', 'R²（GLS）', '定价误差 χ² p（NW / Shanken）', 'HJ 距离', '论文：R²(GLS) / χ² p / HJ p'], rows))
    p.append(f'<figure><div class="title">三因子模型下各组合基本落在预测线附近；只用平均因子时，动量与 carry 组合的收益差异完全无法解释</div>{scatter_svg(pe, ["三因子", "只有 AVG"])}'
             '<figcaption>来源：pricing.py · Fama-MacBeth，截面回归不含常数（同论文）</figcaption></figure>')
    grs = []
    for name in PR.MODELS:
        q = ts.filter(pl.col('模型') == name)
        grs.append([name, f"{q['α%'].abs().mean():.1f}%", f"{q['α%'].abs().max():.1f}%", f"{int((q['t(α)'].abs() > 1.96).sum())}/{q.height}", f"{q['p_GRS'][0]:.2f}"])
    p.append(table(['模型（时间序列回归）', '平均 |α|（年化）', '最大 |α|', 'α 显著的组合', 'GRS 检验 p'], grs))
    s1, s2 = sub.row(0, named=True), sub.row(1, named=True)
    p.append(f"""<ul><li><b>与论文一致</b>：carry 与动量的风险溢价都显著（每月 {F['三因子']['λ_CARRY']:.2f}% 和 {F['三因子']['λ_CMOM']:.2f}%）；
             去掉动量（p={F['去掉动量']['p_χ2_NW']:.2f}）或去掉 carry（p={F['去掉 carry']['p_χ2_NW']:.2f}）后定价误差检验都被拒绝，R²(GLS) 从 {F['三因子']['R2_GLS']:.0%}
             降到 {F['去掉动量']['R2_GLS']:.0%} 和 {F['去掉 carry']['R2_GLS']:.0%}；只有平均因子时 α 显著的组合从 1 个增至 5 个。</li>
             <li><b>拟合不如论文</b>：三因子模型的 R²(GLS) 只有 {F['三因子']['R2_GLS']:.0%}（论文 96%），定价误差 χ² 检验边缘拒绝（p={F['三因子']['p_χ2_NW']:.2f}），
             不过 HJ 距离（p={F['三因子']['p_HJ']:.2f}）与 GRS 检验（p={ts.filter(pl.col('模型') == '三因子')['p_GRS'][0]:.2f}）都不拒绝。</li>
             <li><b>分段</b>：{s1['时段']} 年 carry 溢价显著（p={s1['p_CARRY']:.2f}）、动量不显著（p={s1['p_CMOM']:.2f}）；{s2['时段']} 年反过来，动量显著（p={s2['p_CMOM']:.2f}）、
             carry 不显著（p={s2['p_CARRY']:.2f}）。两个因子的重要性随时间变化。</li></ul>""")

    p.append('<h2>四、经济解释：论文的两条解释在中国都找不到支持</h2>')
    sel = lambda v, f: ets.filter((pl.col('变量').str.starts_with(v)) & (pl.col('因子') == f)).row(0, named=True)
    rows = []
    for v, lab in (('Δ股市', 'Δ股市波动率'), ('Δ商品', 'Δ商品波动率'), ('Δ投机', 'Δ投机度')):
        rows.append([lab] + [f"{sel(v, f)['系数（每 1 标准差，月%）']:+.2f}（t={sel(v, f)['t']:.1f}）" for f in ('CARRY', 'CARRY_long', 'CMOM', 'CMOM_long', 'CMOM_short')])
    p.append(table(['经济变量（每 1 个标准差）', 'CARRY', 'carry 多头腿', 'CMOM', '动量多头腿', '动量空头腿'], rows))
    ef = lambda v, t: efm.filter((pl.col('经济变量') == v) & (pl.col('测试资产') == t)).row(0, named=True)
    p.append(table(['AVG + 经济变量', '定价 carry 组合（χ² p）', '定价动量组合（χ² p）', '论文'], [
        [v, f"{ef(v, 'carry 组合')['p_χ2(NW)']:.2f}", f"{ef(v, '动量组合')['p_χ2(NW)']:.2f}", note] for v, note in (
            ('Δ股市波动率', 'carry 不拒绝（0.13），动量拒绝'), ('Δ商品波动率', '两者都拒绝'), ('Δ投机度', '动量不拒绝（0.28），carry 拒绝'))]))
    p.append('<ul><li><b>carry 与股市波动率</b>：论文发现股市波动上升时 carry 下跌，并能用它定价 carry 组合。中国数据中两者没有关系（t≈0），定价也被拒绝。'
             '注意这里只有中国股市（2010 年起），不是论文的全球股市。</li>'
             '<li><b>动量与投机活动</b>：论文发现投机持仓增加时动量上涨。中国的投机度代理（成交量 ÷ 持仓量）上升时动量反而下跌（t=−2.3），也不能定价动量组合。'
             '成交/持仓比与 CFTC 的非商业持仓含义不同，只是粗糙的代理。</li>'
             '<li>商品波动率不能定价 carry 组合，与论文一致。</li></ul>')

    p.append('<h2>五、多空腿成员与市场状态</h2>')
    p.append(f"<p>论文强调 carry 与动量持有的品种很不一样（每月同向入选品种数的 75% 分位只有 1 个）。中国每月同向入选平均 {ov.mean():.1f} 个（满分 10），75% 分位 {np.percentile(ov, 75):.0f} 个："
             "贴水的品种往往也是近期涨得多的品种，两个策略重叠明显更多。</p>")
    top = lambda leg: '、'.join(f'{a}（{n}）' for _, a, n in cnt.filter(pl.col('leg') == leg).head(5).iter_rows())
    p.append(table(['腿', '入选次数最多的品种（月数）'], [['carry 多头', top('carry_long')], ['carry 空头', top('carry_short')],
                                                     ['动量多头', top('mom_long')], ['动量空头', top('mom_short')]]))
    p.append(table(['商品市场状态（AVG 五分位）', 'AVG', 'CARRY', 'carry 多头', 'carry 空头', 'CMOM', '动量多头', '动量空头'],
                   [[r[0]] + [f'{x:+.2f}%' for x in r[1:]] for r in bys.iter_rows()]))
    p.append('<p>论文发现 carry 与动量在衰退期表现好，能对冲股市下跌。中国没有官方衰退划分，这里按商品市场整体涨跌分组：carry 与动量都在商品最差的月份亏损、最好的月份赚得最多，'
             '是顺商品周期的，与论文相反。</p>')

    p.append('<div class="part">第二部分：换成中国市场习惯的主力合约</div>')
    p.append(f"<p>论文方法持有的次近月，在中国只有 {diag['持有合约即主力的比例']:.0%} 的情况下就是主力合约，成交量中位数只有主力的 {diag['持有合约成交量 ÷ 主力成交量（中位数）']:.0%}；"
             f"很多品种（铁矿石、螺纹钢、豆粕、PTA 等）的交易集中在 1、5、9 月合约上。所以我们再用流动性最好的主力合约算一遍：收益用主力连续合约，"
             f"期限结构用主力与更远的活跃合约之间的年化价差（两种期限结构的截面排序相关中位数 {diag['两种期限结构的截面排序相关（中位数）']:.2f}，贴水/升水判断一致的比例 {diag['贴水 / 升水判断一致的比例']:.0%}），"
             f"动量用主力合约过去 6 个月涨幅；其余完全相同。</p>")
    p.append('<h2>六、主力合约口径的结果</h2>')
    p.append(table(['', '论文方法（次近月）', '主力合约', '论文（美国）'], [
        ['Carry 因子 年化 / 夏普', f"{D['CARRY']['年化均值%']:.1f}% / {D['CARRY']['夏普']:.2f}", f"{DM['CARRY']['年化均值%']:.1f}% / {DM['CARRY']['夏普']:.2f}", '16.3% / 0.73'],
        ['动量因子 年化 / 夏普', f"{D['CMOM']['年化均值%']:.1f}% / {D['CMOM']['夏普']:.2f}", f"<b>{DM['CMOM']['年化均值%']:.1f}% / {DM['CMOM']['夏普']:.2f}</b>（t={DM['CMOM']['t(NW)']:.2f}）", '16.1% / 0.61'],
        ['动量五分位 Q1 → Q5', f"{D['Q1']['年化均值%']:.1f}% → {D['Q5']['年化均值%']:.1f}%", f"{DM['Q1']['年化均值%']:.1f}% → {DM['Q5']['年化均值%']:.1f}%", '−1.7% → 14.4%'],
        ['动量风险溢价 p', f"{F['三因子']['p_CMOM(NW)']:.2f}", f"{FM['三因子']['p_CMOM(NW)']:.2f}", '0.00'],
        ['三因子：χ² p / HJ p', f"{F['三因子']['p_χ2_NW']:.2f} / {F['三因子']['p_HJ']:.2f}", f"{FM['三因子']['p_χ2_NW']:.2f} / {FM['三因子']['p_HJ']:.2f}", '0.22 / 0.22'],
        ['去掉动量：χ² p', f"{F['去掉动量']['p_χ2_NW']:.2f}（拒绝）", f"<b>{FM['去掉动量']['p_χ2_NW']:.2f}（不拒绝）</b>", '0.01（拒绝）'],
        ['去掉 carry：χ² p', f"{F['去掉 carry']['p_χ2_NW']:.2f}（拒绝）", f"{FM['去掉 carry']['p_χ2_NW']:.2f}（拒绝）", '0.00（拒绝）'],
        ['carry 与动量同向重叠（每月均值）', f'{ov.mean():.1f}', f'{ovm.mean():.1f}', '75% 分位 1']]))
    p.append('<p>主力合约上 carry 依然显著、依然必不可少；但按论文方式构造的动量因子不再显著，去掉它模型也不被拒绝。下面拆解原因。</p>')

    p.append('<h2>七、差距从哪里来</h2>')
    a = x2.filter(pl.col('样本') == '全部')
    g = lambda s, c, k: a.filter((pl.col('信号') == s) & (pl.col('持有合约') == c))[k][0]
    p.append('<h3>1. 主要取决于持有哪个合约</h3><p>在两种口径都有数据的品种-月上，把“用哪种方法排序”和“持有哪个合约”分开组合：</p>')
    p.append(table(['排序信号', '持有合约', 'Carry 年化（夏普）', '动量 年化（夏普）'],
                   [[s, c, f"{g(s, c, 'CARRY%'):.1f}%（{g(s, c, 'CARRY夏普'):.2f}）", f"{g(s, c, 'CMOM%'):.1f}%（{g(s, c, 'CMOM夏普'):.2f}）"]
                    for s in ('论文方法', '主力合约') for c in ('次近月', '主力')]))
    rob = x2.filter(pl.col('样本') != '全部')
    gr = lambda s, c, k: rob.filter((pl.col('样本') == s) & (pl.col('信号') == '主力合约') & (pl.col('持有合约') == c))[k][0]
    p.append(table(['稳健性（主力合约的排序信号）', 'Carry：持有主力 → 次近月', '动量：持有主力 → 次近月'],
                   [[s, f"{gr(s, '主力', 'CARRY%'):.1f}% → {gr(s, '次近月', 'CARRY%'):.1f}%", f"{gr(s, '主力', 'CMOM%'):.1f}% → {gr(s, '次近月', 'CMOM%'):.1f}%"]
                    for s in ('收益截尾 1%', '去掉煤焦', '2019 年前', '2020 年起')]))
    p.append(f"<p>排序方法不变，只把持有的合约从主力换成次近月，carry 从 {g('主力合约', '主力', 'CARRY%'):.0f}% 升到 {g('主力合约', '次近月', 'CARRY%'):.0f}%，"
             f"动量从 {g('主力合约', '主力', 'CMOM%'):.0f}% 升到 {g('主力合约', '次近月', 'CMOM%'):.0f}%；截尾极端收益、去掉煤焦、前后分段都如此。</p>")
    nr = ne.filter((pl.col('因变量') == '下月近月超额') & (pl.col('回归') == '三者一起'))
    p.append('<h3>2. 近月多赚的部分由过去的涨势预测</h3><p>把同一品种下个月“次近月收益 − 主力收益”（近月超额）对月末已知的变量做截面回归（三者一起）：</p>')
    p.append(table(['预测变量', '每 1 个标准差的效果（月）', 't 值'], [[r['变量'], f"{r['系数%']:+.2f}%", f"{r['t']:.1f}"] for r in nr.iter_rows(named=True)]))
    p.append('<p>能预测近月超额的是<b>过去的价格涨势</b>，而不是当前的贴水程度：过去涨得多的品种，下个月它的近月合约会比主力多涨一截。'
             '一个可能的解释（未直接验证，我们没有现货与库存数据）：近月合约马上交割，必须跟着现货走，反映“眼下”的供需；主力合约在几个月后，反映市场对“到时候”的预期。'
             '商品因供应紧张持续上涨时，市场预期紧张会缓解，主力合约涨得有所保留；但眼下的紧张往往比预期更持久，近月继续被往上拉。'
             '所以中国商品的价格趋势主要体现在曲线前端。carry 用次近月也更强，主要因为深度贴水的品种往往也是近期涨得多的品种。</p>')
    sp = lambda lab, f, k: spn.filter((pl.col('口径') == lab) & (pl.col('因子') == f))[k][0]
    p.append('<h3>3. carry 与动量是不是一回事</h3><p>对平均因子和另一个因子回归，看剩下的收益（α，年化）：</p>')
    p.append(table(['口径', '被解释的因子', '原始年化', '控制另一因子后的 α', 'R²'],
                   [[lab, NAMES[f], f"{sp(lab, f, '原始年化%'):.1f}%", f"<b>{sp(lab, f, 'α年化%'):.1f}%</b>（t={sp(lab, f, 't(α)'):.2f}）", f"{sp(lab, f, 'R2'):.2f}"]
                    for lab in ('论文方法（次近月）', '主力合约') for f in ('CMOM', 'CARRY')]))
    p.append('<p>两种口径下互相回归的 R² 都只有 0.15~0.27，两者相关但不是一回事。论文方法下动量控制 carry 后仍有显著的独立收益；'
             '主力合约上则是动量几乎全被 carry 吸收、carry 保持独立。</p>')
    con = rd('cmp_construction', None)
    dsp = rd('cmp_dispersion', None)
    cc = lambda k, n, c: con.filter((pl.col('回看月数') == k) & (pl.col('构造') == n))[c][0]
    p.append('<h3>4. 主力合约上的截面动量：关键在回看期</h3>')
    p.append('<p>同一批主力合约、同一个品种池，只改变组合构造方式和回看期（多空两边各 1 元名义，可直接比较）：</p>')
    p.append(table(['构造方式', '回看 6 个月（论文）', '回看 12 个月'],
                   [[n] + [f"夏普 {cc(k, n, '夏普'):.2f}（t={cc(k, n, 't'):.2f}）" for k in (6, 12)]
                    for n in ('两端各 5 个（论文 M5）', '两端各 20%', '两端各 1/3', '全部按排名加权')]))
    p.append('<p>回看 6 个月时，无论怎么构造都不显著；回看 12 个月时，各种构造都显著，夏普 0.55~0.65。但 12 个月的结果需要再拆开看。</p>')
    rs = rd('cmp_roll_split', None)
    rf = rd('cmp_roll_split_fm', None)
    p.append('<p><b>12 个月“动量”里装的是什么？</b>主力连续合约的后复权涨幅包含两部分：合约价格本身的变化（未复权价格的变化），'
             '以及每次换月时新旧合约价差带来的展期收益（贴水品种换到更便宜的远月，每次换月都“赚”一点，即 carry）。分别用两部分排序：</p>')
    p.append(table(['过去 12 个月的信号', '时间序列（等权 ±1）', '截面（全部按排名加权）'],
                   [[r['信号'], f"{r['时间序列%']:.1f}%（t={r['t(时间序列)']:.2f}）", f"{r['截面%']:.1f}%（t={r['t(截面)']:.2f}）"] for r in rs.iter_rows(named=True)]))
    p.append('<p>三者一起做截面回归（每 1 个标准差，月度）：' + '；'.join(f"{r['变量']} {r['系数%']:+.2f}%（t={r['t']:.2f}）" for r in rf.iter_rows(named=True)) + '。</p>')
    mc = rd('cmp_mom_vs_carry12', None).row(0, named=True)
    p.append(f"<p>换一种完全独立的检验：直接用每个月末的合约价差算出过去 12 个月的平均贴水（carry1-12，不经过价格涨跌）。12 个月动量与它的截面相关 {mc['截面相关']:.2f}；"
             f"动量策略年化 {mc['动量%']:.1f}%（t={mc['t动量']:.2f}），对 carry1-12 策略回归后只剩 {mc['α%']:.1f}%（t={mc['tα']:.2f}），不再显著。</p>")
    p.append('<p><b>主力合约上，12 个月“动量”的收益大部分与过去一年的 carry 重叠；扣掉 carry 后剩下的价格动量很弱，统计上不显著（但不能断定为零）。</b>'
             '直觉是：供应持续紧张的品种，价格在涨，期货也一直贴水，贴水又通过换月让连续价格继续上涨，所以“过去涨得多”与“过去一直贴水”往往是同一批品种。</p>')
    tr = lambda k, w, c: tsd.filter((pl.col('回看月数') == k) & (pl.col('仓位') == w))[c][0]
    dd = lambda k, lab, c: dsp.filter((pl.col('回看月数') == k) & (pl.col('月份') == lab))[c][0]
    p.append('<p>与 TSMOM 的关系：我们 TSMOM 的复现也用主力合约。把它（只看商品）的收益拆成两块：每个品种的仓位减去当月平均仓位后的<b>截面部分</b>'
             '（多空相抵，只赚“涨势品种跑赢跌势品种”），和平均仓位乘以商品整体收益的<b>整体方向部分</b>（赚“整个商品市场涨还是跌”）。两块相加严格等于 TSMOM 收益。</p>')
    p.append(table(['回看期', '仓位', 'TSMOM', '截面部分', '整体方向部分'],
                   [[f'{k} 个月', w, f"{tr(k, w, 'TSMOM%'):.1f}%（t={tr(k, w, 't(TSMOM)'):.1f}）", f"<b>{tr(k, w, '截面部分%'):.1f}%（t={tr(k, w, 't(截面)'):.1f}）</b>",
                     f"{tr(k, w, '整体方向%'):.1f}%（t={tr(k, w, 't(方向)'):.1f}）"] for k in (12, 6) for w in ('40%/σ（同 TSMOM 论文）', '等权 ±1')]))
    p.append(f"<p>TSMOM 在商品上的收益约八成来自截面部分，判断整个商品市场方向的部分不显著。6 个月回看时截面部分也显著，与上表不矛盾：它按涨跌符号下注且不做归一，"
             f"品种走势分化（一半涨、一半跌）的月份仓位大，同涨同跌的月份仓位几乎为零。6 个月的截面动量正好只在分化的月份存在："
             f"分化大的月份年化 {dd(6, '多空分歧大的月份', '年化%'):.1f}%（t={dd(6, '多空分歧大的月份', 't'):.1f}），分化小的月份 {dd(6, '多空分歧小的月份', '年化%'):.1f}%（t={dd(6, '多空分歧小的月份', 't'):.1f}）。"
             "论文的 M5 每个月押同样的规模，效果被稀释了。TSMOM 用的同样是后复权涨幅，所以它在商品上的截面部分也有很大一部分是 carry。</p>")

    p.append('<h2>结论</h2><ul>'
             f"<li><b>按论文方法，结论基本成立</b>：carry（夏普 {D['CARRY']['夏普']:.2f}）与动量（{D['CMOM']['夏普']:.2f}）都显著，去掉任何一个模型都被拒绝，与论文一致；三因子模型的拟合不如论文（R²(GLS) {F['三因子']['R2_GLS']:.0%}，χ² 检验边缘拒绝）。</li>"
             '<li><b>经济解释不成立</b>：股市波动率解释不了 carry，投机度解释不了动量；中国的 carry 与动量都顺商品周期，而非像论文那样在衰退期表现好。</li>'
             f"<li><b>换成主力合约</b>：carry 依然成立（夏普 {DM['CARRY']['夏普']:.2f}）；按论文方式（6 个月回看）构造的动量不再显著，无论组合怎么构造都一样。"
             '原因是中国商品 6 个月的价格趋势主要体现在临近交割的近月合约上。回看 12 个月时主力合约上的“动量”显著，但收益大部分与过去一年的 carry 重叠，扣掉后剩下的价格动量不显著；TSMOM 在商品上的截面收益也有很大一部分是这个。</li>'
             '<li><b>论文方法下的动量也含有展期收益</b>：它用滚动持有次近月的收益计算，同样包含换月的展期收益，论文方法下动量的独立性有多少来自价格趋势、多少来自展期收益，还需要单独拆分（未做）。</li>'
             '<li><b>可得性</b>：次近月在交割前一个月，个人客户须在交割月前平仓，流动性与容量有限，交割相关的挤仓也会放大波动；论文方法下更高的收益，普通投资者很难拿到。</li></ul>')
    p.append('<h2>与论文的差异</h2><ul><li>次近月须满足最低流动性（持仓 ≥ 500 手且有成交），早期常有月份合格品种不足 10 个，定价检验可用 '
             f'{T} 个月（论文 501 个月），检验功效更低。</li><li>测试组合 11 个：没有畜牧板块，能源化工因次近月不活跃被排除。</li>'
             '<li>每月品种数从 10 个增至 60 多个，固定 5 多 5 空在后期只覆盖两端很小一部分。</li>'
             '<li>经济变量都是代理：沪深 300 代替全球股市、成交/持仓比代替 CFTC 持仓、商品市场状态代替衰退期。</li>'
             '<li>未做两步 GMM 的排除检验与 J 检验、管理组合、随机分组、个品种检验、价值因子与下行风险因子等稳健性检验。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [
        ['<code>factors.py</code>', '品种池；论文方法（次近月）与主力合约两个版本的月度面板、因子、测试组合、多空腿成员'],
        ['<code>pricing.py</code>', '描述统计、主成分、Fama-MacBeth、HJ 距离、时间序列回归与 GRS 检验、分段（两个版本）'],
        ['<code>economics.py</code>', '股市波动率、商品波动率、投机度与因子的关系；多空腿成员重叠；按市场状态分组（两个版本）'],
        ['<code>compare.py</code>', '两种口径对比：2×2 拆解、近月超额、互相解释、TSMOM 拆解、主力合约上的构造方式 × 回看期、价格变化与展期收益的拆分'],
        ['<code>build_report.py</code>', '生成本报告'], ['<code>output/*_paper.parquet、*_main.parquet、cmp_*.parquet</code>', '各步结果；*_log.txt 为运行日志']]))
    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>商品三因子复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
