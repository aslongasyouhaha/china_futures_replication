"""生成期货持仓量与收益预测复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 读取。输出: report.html"""
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
    b = ''.join('<tr>' + ''.join(f'<td>{"" if c is None else c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def ts_svg(a):
    d = a.drop_nulls(['OI12', 'R12'])
    W, Hh, left, y0, ph = 760, 280, 50, 20, 220
    s1, s2 = d['OI12'].to_numpy() * 100, d['R12'].to_numpy() * 100
    lo, hi = min(s1.min(), s2.min()) - 0.5, max(s1.max(), s2.max()) + 0.5
    n = d.height
    xs = lambda i: left + i / (n - 1) * (W - left - 120)
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="商品持仓增长与过去收益">']
    for t in np.arange(np.ceil(lo), hi, 2):
        s.append(f'<line x1="{left}" x2="{W-120}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="{"axis" if abs(t) < 1e-9 else "grid"}"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t:.0f}%</text>')
    for i, m in enumerate(d['m'].to_list()):
        if m.month == 1 and m.year % 2 == 0:
            s.append(f'<text x="{xs(i):.1f}" y="{y0+ph+18}" text-anchor="middle" class="sub">{m.year}</text>')
    for v, cls, lab in ((s1, 'accent', '持仓金额增长'), (s2, 'muted', '商品收益')):
        s.append(f'<polyline points="{" ".join(f"{xs(i):.1f},{yv(y):.1f}" for i, y in enumerate(v))}" class="ln {cls}" stroke-width="2"/>'
                 f'<text x="{W-114}" y="{yv(v[-1])+4:.1f}" class="lbl {cls}t">{lab}</text>')
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
svg .grid{stroke:var(--line)}svg .axis{stroke:var(--axis)}svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}svg .ln.muted{stroke:var(--muted)}
svg text{fill:var(--fg);font-size:12px}svg .sub{fill:var(--sub)}svg .accentt{fill:var(--accent);font-weight:600}svg .mutedt{fill:var(--sub)}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    a = rd('aggregate').sort('m')
    t6 = rd('table6')
    sp = rd('subperiods')
    fin = rd('financial')
    d = a.drop_nulls(['OI12', 'R12', 'basis', 'ret'])
    r = {x['回归'][:3]: x for x in t6.iter_rows(named=True)}
    p = []
    p.append('<h1>中国期货持仓量与收益预测复现报告</h1><div class="meta">复现论文：Harrison Hong &amp; Motohiro Yogo，《What Does Futures Market Interest Tell Us about the Macroeconomy and Asset Prices?》，'
             f'<i>Journal of Financial Economics</i>，2012（全文依据 NBER 工作论文 16712 修订版）· 中国商品期货 {d["m"].min().strftime("%Y-%m")} 至 {d["m"].max().strftime("%Y-%m")}，{d.height} 个月</div>')
    p.append(f"<p class=\"lead\">论文的两个主要发现在中国方向一致、但统计上较弱：商品持仓金额增长预测更高的商品收益（每 1 个标准差 {r['(2)']['OI12']}，论文 +0.72%（t=2.47）），"
             f"基差越高（远月越贵）后续收益越低（{r['(1)']['basis']}，论文 −0.57%（t=−2.08））。持仓的预测力集中在 2016 年以后；只用合约张数时消失；"
             "过去收益没有预测力。对国债的预测方向与论文一致但不显著，对股指无预测力。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{r['(2)']['OI12'].split('（')[0]}%</b><span>持仓增长每 1 个标准差的月收益（论文 +0.72%）</span></div>"
             f"<div class=\"kpi\"><b>{r['(1)']['basis'].split('（')[0]}%</b><span>基差每 1 个标准差的月收益（论文 −0.57%）</span></div>"
             f"<div class=\"kpi\"><b>{r['(1)']['R2%']:.1f}% → {r['(2)']['R2%']:.1f}%</b><span>R²：加入持仓增长（论文 2.5% → 4.8%）</span></div></div>")
    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>传统观点认为期货价格已经包含了对未来经济和资产价格的全部信息，持仓量无关紧要。论文认为：经济景气时套保需求上升（生产商卖期货、消费者买期货），所以持仓量随景气上升；'
             '但套保方向不定，价格可能涨也可能跌。加上信息扩散缓慢、套利有限，<b>持仓量是比价格更可靠的景气信号</b>，能预测商品收益，也能预测债券收益（景气→通胀压力→债券跌）。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['商品收益', '每个商品取剩余期限 1–12 个月的所有合约等权；商品在板块内等权；4 个板块（农产品、能源、畜牧、金属）等权', '同左；4 个板块为农产品（含鸡蛋、生猪，畜牧太少）、有色贵金属、黑色、能源化工；合约须月初持仓 ≥ 100 手'],
        ['持仓增长', '持仓金额（现货价 × 持仓量）板块内加总的月增长率，4 个板块等权，再取 12 个月几何平均', '同左，持仓金额 = 结算价 × 持仓量 × 合约乘数；板块增长只用前后两月都存在的品种（避免新品种上市造成跳升）'],
        ['基差', '(F/S)^(1/(T−t)) − 1，商品内取中位数，板块内取中位数，板块平均', '同左，以最近月合约代替现货'],
        ['其他预测变量', '短期利率、期限利差、经济活动指数、套保者多空失衡（CFTC）、过去 12 个月收益', '只有过去 12 个月收益；利率、经济活动指数、CFTC 数据没有，未纳入'],
        ['回归', '下月收益对标准化预测变量回归，系数 = 每 1 个标准差的月收益（%）', '同左，t 值用 Newey-West（滞后 12）'],
        ['其他市场', '货币、债券、股票期货的持仓增长预测各自收益；商品持仓预测债券收益', '国债期货（T）、沪深 300 股指期货（IF）']]))
    p.append('<h2>一、持仓增长与商品收益</h2>')
    p.append(f'<figure><div class="title">持仓金额增长与商品收益高度同步（相关 {np.corrcoef(d["OI12"], d["R12"])[0,1]:.2f}）</div>{ts_svg(a)}'
             '<figcaption>来源：oi.py · 两条线均为 12 个月几何平均的月度增长率</figcaption></figure>')
    p.append(f"<p>商品组合月均收益 {d['ret'].mean()*100:.2f}%、月波动 {d['ret'].std()*100:.2f}%；持仓增长的一阶自相关 {np.corrcoef(d['OI12'][1:], d['OI12'][:-1])[0,1]:.2f}，"
             f"与基差相关 {np.corrcoef(d['OI12'], d['basis'])[0,1]:.2f}。论文中持仓与基差几乎不相关，与收益正相关，中国也是如此。</p>")
    paper = {'(1)': '基差 −0.57（−2.08），R² 2.48%（另含利率、利差）', '(2)': '持仓 +0.72（2.47），R² 4.81%', '(3)': '过去收益 +0.30（1.27）',
             '(4)': '持仓 +0.77（1.85），过去收益 −0.10', '(5)': '脚注：结果相似', '(6)': ''}
    rows = [[x['回归'], x.get('basis') or '', x.get('OI12') or x.get('OI12_n') or '', x.get('R12') or '', f"{x['R2%']:.1f}%", paper[x['回归'][:3]]] for x in t6.iter_rows(named=True)]
    p.append(table(['回归', '基差', '持仓增长', '过去 12 个月收益', 'R²', '论文'], rows))
    p.append(table(['时段', '回归', '基差', '持仓增长', '过去收益', 'R²'],
                   [[x['资产'], x['回归'], x['basis'], x['OI12'], x.get('R12') or '', f"{x['R2%']:.1f}%"] for x in sp.iter_rows(named=True)]))
    p.append('<ul><li><b>基差</b>：基差越高（远月比近月贵，即升水），后续商品收益越低，显著，与论文和此前的 carry 研究一致。</li>'
             '<li><b>持仓增长</b>：方向与论文一致，加入基差后接近显著；2016 年后显著，控制过去收益后作用更强。</li>'
             '<li><b>过去收益</b>没有预测力（与论文相同），在后半段甚至为负。</li>'
             '<li><b>只用合约张数</b>时持仓增长没有预测力，论文说两种口径结果相似。中国的预测力似乎依赖于持仓金额中的价格成分，这一点削弱了“持仓量本身含有信息”的解释。</li></ul>')
    p.append('<h2>二、其他市场</h2>')
    p.append(table(['回归', '系数（t）', 'R²', '月数'], [[x['回归'], x['系数（t）'], f"{x['R2%']:.1f}%", x['月数']] for x in fin.iter_rows(named=True)]))
    p.append('<p>论文发现商品持仓增长显著预测更低的国债收益（每 1 个标准差 −0.32%），债券持仓增长也预测更低的债券收益。中国方向相同，但国债期货样本只有约 11 年，不显著。股指期货的持仓增长没有预测力，论文中股市也是证据最弱的。</p>')
    p.append('<h2>结论</h2><ul><li>论文的主要结论在中国方向一致：持仓上升预示商品收益更高、基差高预示收益更低。</li>'
             '<li>但证据较弱：全样本只接近显著，只在 2016 年后显著；只用合约张数时消失，说明信息可能主要来自价格而非持仓本身。</li>'
             '<li>论文的宏观解释（持仓随景气上升、预示通胀与债券下跌）需要利率与经济活动数据，本次未能检验。</li></ul>')
    p.append('<h2>与论文的差异</h2><ul><li>全文依据 NBER 工作论文修订版（2011 年 8 月），正式发表版本可能有改动。</li>'
             '<li>缺少短期利率、期限利差、经济活动指数与 CFTC 套保者持仓，相应的对照回归未做。</li>'
             '<li>样本 2010–2026 年（约 16 年），论文 1965–2008 年。</li><li>用最近月合约代替现货计算基差。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [['<code>oi.py</code>', '合约月度收益、持仓金额、基差；板块与总组合；预测回归；国债、股指期货'],
                                     ['<code>build_report.py</code>', '生成本报告'], ['<code>output/*.parquet</code>', '各步结果；oi_log.txt 为运行日志']]))
    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>持仓量与收益预测复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
