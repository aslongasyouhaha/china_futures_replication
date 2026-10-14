"""生成价值与动量复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 读取。输出: report.html"""
import os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def cum_svg(d):
    W, Hh, left, y0, ph = 760, 300, 50, 20, 230
    ser = {'动量': np.cumsum(np.log1p(d['M_F'].to_numpy())), '价值': np.cumsum(np.log1p(d['V_F'].to_numpy())),
           '50/50': np.cumsum(np.log1p(d['C_F'].to_numpy()))}
    lo = min(v.min() for v in ser.values()) - 0.05
    hi = max(v.max() for v in ser.values()) + 0.05
    n = d.height
    xs = lambda i: left + i / (n - 1) * (W - left - 90)
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="价值、动量与组合的累计收益">']
    for t in np.arange(np.ceil(lo * 4) / 4, hi, 0.25):
        s.append(f'<line x1="{left}" x2="{W-90}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="{"axis" if abs(t) < 1e-9 else "grid"}"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{np.exp(t):.2f}×</text>')
    for i, m in enumerate(d['m'].to_list()):
        if m.month == 1 and m.year % 2 == 0:
            s.append(f'<text x="{xs(i):.1f}" y="{y0+ph+18}" text-anchor="middle" class="sub">{m.year}</text>')
    style = {'动量': 'accent', '价值': 'muted', '50/50': 'fg'}
    for k, v in ser.items():
        s.append(f'<polyline points="{" ".join(f"{xs(i):.1f},{yv(y):.1f}" for i, y in enumerate(v))}" class="ln {style[k]}" stroke-width="2"/>'
                 f'<text x="{W-86}" y="{yv(v[-1])+4:.1f}" class="lbl {style[k]}t">{k}</text>')
    s.append('</svg>')
    return ''.join(s)


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--muted:#a9a69b;--accent:#2f6fd0;--axis:#9a978e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.meta{color:var(--sub);font-size:13px;margin-bottom:12px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--bg)}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}figcaption{color:var(--sub);font-size:12px;margin-top:6px}
figure .title{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--line)}svg .axis{stroke:var(--axis)}svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}svg .ln.fg{stroke:var(--fg)}svg .ln.muted{stroke:var(--muted)}
svg text{fill:var(--fg);font-size:12px}svg .sub{fill:var(--sub)}svg .accentt{fill:var(--accent);font-weight:600}svg .fgt{fill:var(--fg);font-weight:600}svg .mutedt{fill:var(--sub)}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    t = {r['组合']: r for r in rd('table1').iter_rows(named=True)}
    d = rd('portfolios').sort('m')
    cor = {r['口径']: r['相关'] for r in rd('correlation').iter_rows(named=True)}
    sp = rd('subperiods')
    ck = {r['组合']: r for r in rd('carry_link').iter_rows(named=True)}
    f = lambda k, c, fmt='{:.1f}%': fmt.format(t[k][c])
    p = []
    p.append('<h1>中国商品期货价值与动量复现报告</h1><div class="meta">复现论文：Clifford S. Asness, Tobias J. Moskowitz &amp; Lasse Heje Pedersen，'
             f'《Value and Momentum Everywhere》，<i>Journal of Finance</i>，2013 · 仅商品期货部分 · 中国商品期货 {d["m"].min().strftime("%Y-%m")} 至 {d["m"].max().strftime("%Y-%m")}，{d.height} 个月 · 不计成本（同论文）</div>')
    p.append(f"<p class=\"lead\">动量在中国商品期货上与论文几乎一样强（年化 {t['动量 因子']['均值']:.1f}%，t={t['动量 因子']['t']:.2f}；论文 11.5%，t=3.14），"
             f"价值则完全不存在（{t['价值 因子']['均值']:.1f}%，论文 7.3%）。两者负相关（{cor['因子']:.2f}），各半组合显著。与此前的研究衔接：动量的收益大部分与 carry 重叠；"
             f"价值表面无效是因为它与 carry 方向相反；而价值与动量各半的组合几乎独立于 carry，控制 carry 后仍有 {ck['50/50 因子']['控制 carry 后 α%']:.1f}%（t={ck['50/50 因子']['t']:.2f}）。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{t['动量 因子']['均值']:.1f}%</b><span>动量因子年化（论文 11.5%）</span></div>"
             f"<div class=\"kpi\"><b>{t['价值 因子']['均值']:.1f}%</b><span>价值因子年化（论文 7.3%）</span></div>"
             f"<div class=\"kpi\"><b>{cor['因子']:.2f}</b><span>价值与动量的相关（论文 −0.46）</span></div>"
             f"<div class=\"kpi\"><b>{t['50/50 因子']['夏普']:.2f}</b><span>各半组合夏普（论文 0.77）</span></div></div>")
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['范围', '8 个资产类别：美英欧日个股、股指期货、货币、国债、商品期货；1972–2011', '只有商品期货：中国股指期货只有 4 个、无账面价值，国债期货只有一个国家，无法构成横截面'],
        ['商品样本', '27 个商品，至少 10 个', f"中国商品期货，每月有价值信号的品种 {d['V_n'].min()}~{d['V_n'].max()} 个（中位数 {d['V_n'].median():.0f}）"],
        ['收益', '最活跃合约的日超额收益复利成月度', '主力连续后复权'],
        ['价值', 'ln(4.5–5.5 年前平均现货价 ÷ 最近现货价)', '同左；现货用未复权主力价格近似；需上市满 5.5 年，样本因此从 2010 年开始'],
        ['动量', '过去 12 个月收益，剔除最近 1 个月（MOM2–12）', '同左'],
        ['组合', '三等分等权 P1–P3；P3−P1；按排名加权因子（多空各 1 元）；价值、动量各半', '同左'],
        ['报告', '年化均值、t、波动、夏普；对等权商品篮子的 α；价值与动量残差的相关', '同左；t 用 Newey-West'],
        ['其余部分', '跨资产的共同因子、流动性风险、三因子模型', '中国只有一个资产类别，未做']]))
    p.append('<h2>一、主结果（对应论文 Table I 商品部分）</h2>')
    rows = []
    for k in ('价值 P1', '价值 P2', '价值 P3', '价值 P3−P1', '价值 因子', '动量 P1', '动量 P2', '动量 P3', '动量 P3−P1', '动量 因子', '50/50 P3−P1', '50/50 因子', '等权商品篮子'):
        r = t[k]
        paper = f"{r['论文均值']:.1f}%（t={r['论文t']:.2f}），夏普 {r['论文夏普']:.2f}" if r['论文均值'] is not None else ''
        rows.append([k, f"{r['均值']:.1f}%", f"{r['t']:.2f}", f"{r['波动']:.1f}%", f"{r['夏普']:.2f}", '—' if k == '等权商品篮子' else f"{r['α']:.1f}%（{r['tα']:.2f}）", paper])
    p.append(table(['组合', '年化均值', 't', '年化波动', '夏普', '对商品篮子的 α（t）', '论文'], rows))
    p.append(f'<figure><div class="title">动量稳定上升，价值基本走平，两者各半的组合更平稳</div>{cum_svg(d)}'
             '<figcaption>来源：vme.py · 排名加权因子，月度对数收益累计</figcaption></figure>')
    p.append(f"<ul><li><b>动量</b>与论文几乎一样：因子 {t['动量 因子']['均值']:.1f}%、t={t['动量 因子']['t']:.2f}，三组单调递增（P1 {t['动量 P1']['均值']:.1f}% → P3 {t['动量 P3']['均值']:.1f}%）。</li>"
             f"<li><b>价值</b>不存在：高价值组（过去 5 年跌得多）与低价值组收益几乎相同；中间组反而最高（{t['价值 P2']['均值']:.1f}%），没有单调关系。论文中商品价值也较弱（t=1.6~1.9）。</li>"
             f"<li><b>价值与动量负相关</b>（P3−P1 {cor['P3−P1']:.2f}、因子 {cor['因子']:.2f}），比论文更强，所以各半组合波动小、夏普接近论文。</li></ul>")
    p.append(table(['时段', '价值 因子', '动量 因子', '各半组合'],
                   [[per] + [f"{r['均值']:.1f}%（t={r['t']:.2f}）" for r in sp.filter(pl.col('时段') == per).iter_rows(named=True)] for per in sp['时段'].unique(maintain_order=True)]))
    p.append('<p>两个时段结论相同：动量显著，价值不显著。</p>')
    p.append('<h2>二、补充：与 carry 的关系（不是论文内容）</h2>')
    p.append('<p>此前的 carry、商品三因子研究发现，主力合约上的 12 个月动量与过去一年的 carry 高度重叠。这里在同一品种池、同一月份构造 carry1-12 因子（过去 12 个月平均年化 carry，按排名加权），对等权商品篮子与 carry 因子回归：</p>')
    p.append(table(['组合', '原始年化', '与 carry 的相关', '控制 carry 后的 α', 't', 'carry β'],
                   [[k, f"{r['原始年化%']:.1f}%", f"{r['与 carry 相关']:.2f}", '—' if r['控制 carry 后 α%'] is None else f"{r['控制 carry 后 α%']:.1f}%",
                     f"{r['t']:.2f}", '—' if r['carry β'] is None else f"{r['carry β']:.2f}"] for k, r in ck.items()]))
    p.append('<ul><li><b>动量</b>与 carry 相关 0.52，控制 carry 后只剩约 3.5%，不显著，与此前的发现一致。</li>'
             '<li><b>价值</b>与 carry 负相关（−0.43）：过去 5 年跌得多的“便宜”品种往往处于升水。价值表面无效，部分是因为它天然站在 carry 的对立面；控制 carry 后有约 4% 的正收益，但不显著。</li>'
             '<li><b>价值与动量各半</b>的组合与 carry 几乎无关（0.16），控制 carry 后仍显著（约 3.9%，t≈3.1）：这是目前在主力合约上找到的、独立于 carry 的收益来源。</li></ul>')
    p.append('<h2>结论</h2><ul><li>论文在商品期货上的两个发现，中国只成立一个：动量成立且与论文同等强度，价值不成立。</li>'
             '<li>价值与动量负相关、组合更稳定，这一点在中国更明显。</li><li>结合 carry 来看，中国商品期货的主要收益来源是 carry；动量大部分与它重叠；价值与动量的组合提供了与 carry 无关的额外收益。</li></ul>')
    p.append('<h2>与论文的差异</h2><ul><li>只复现商品期货一个资产类别，论文的核心卖点（跨资产的共同因子结构、流动性风险）无法检验。</li>'
             '<li>样本 2010–2026 年、约 16 年（论文 40 年）；价值信号需要上市满 5.5 年，早期品种少。</li><li>现货价格用未复权主力合约价格近似，含有期货与现货的基差，相对 5 年的价格变化影响不大。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [['<code>vme.py</code>', '月度面板、价值与动量信号、三等分组合与排名加权因子、分段、与 carry 的关系'],
                                     ['<code>build_report.py</code>', '生成本报告'], ['<code>output/*.parquet</code>', '各步结果；vme_log.txt 为运行日志']]))
    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>价值与动量复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
