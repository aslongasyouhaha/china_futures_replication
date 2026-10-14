"""生成挤仓论文复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 读取。输出: report.html"""
import json
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import china_events as CE  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def bars_svg(t):
    """压力月 vs 对照月：偏离超过各阈值的到期次数。"""
    th = (0.05, 0.10, 0.20)
    vals = [((t['压力月_max_abs'] > x).sum(), (t['对照月_max_abs'] > x).sum()) for x in th]
    W, Hh, left, y0, ph = 760, 280, 60, 20, 200
    mx = max(max(v) for v in vals)
    band = (W - left - 20) / len(th)
    yv = lambda v: y0 + ph - v / mx * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="压力月与对照月的大偏离次数">',
         f'<line x1="{left}" x2="{W-20}" y1="{y0+ph}" y2="{y0+ph}" class="axis"/>']
    for i, (x, (a, b)) in enumerate(zip(th, vals)):
        x0 = left + i * band + band * 0.2
        bw = band * 0.28
        for k, (v, cls, lab) in enumerate(((a, 'c1', '压力月'), (b, 'c2', '对照月'))):
            xx = x0 + k * (bw + 6)
            s.append(f'<rect x="{xx:.1f}" y="{yv(v):.1f}" width="{bw:.1f}" height="{y0+ph-yv(v):.1f}" class="{cls}"><title>{lab}：{v} 次</title></rect>'
                     f'<text x="{xx+bw/2:.1f}" y="{yv(v)-5:.1f}" text-anchor="middle" class="lbl">{v}</text>')
        s.append(f'<text x="{x0+bw+3:.1f}" y="{y0+ph+18}" text-anchor="middle" class="sub">最大偏离超过 {x:.0%}</text>')
    s.append(f'<rect x="{left}" y="{Hh-20}" width="12" height="12" class="c1"/><text x="{left+18}" y="{Hh-10}" class="lbl">压力月（个人须在月底前平仓）</text>'
             f'<rect x="{left+250}" y="{Hh-20}" width="12" height="12" class="c2"/><text x="{left+268}" y="{Hh-10}" class="lbl">对照月（再往前一个月）</text></svg>')
    return ''.join(s)


def intraday_svg(it):
    ev = [('LU2410', '低硫燃料油 LU2410（2024-09）'), ('A1801', '豆一 A1801（2017-12）'), ('SA2005', '纯碱 SA2005（2020-04）')]
    W, Hh = 760, 250
    pw, ph, gap, y0 = 220, 170, 30, 30
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="最后几个交易日近月相对基准的偏离">']
    for i, (sym, lab) in enumerate(ev):
        r = it.filter((pl.col('事件') == sym) & (pl.col('版本').str.starts_with('A'))).sort('time')
        d = r['distortion'].to_numpy() * 100
        lo, hi = min(d.min(), 0) - 2, max(d.max(), 0) + 2
        x0 = 30 + i * (pw + gap)
        n = len(d)
        xs = lambda k: x0 + k / (n - 1) * pw
        yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
        s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" class="frame"/><text x="{x0}" y="{y0-10}" class="lbl">{lab}</text>'
                 f'<line x1="{x0}" x2="{x0+pw}" y1="{yv(0):.1f}" y2="{yv(0):.1f}" class="axis"/>')
        for t in np.arange(np.ceil(lo / 10) * 10, hi, 10):
            s.append(f'<text x="{x0-4}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t:.0f}%</text>')
        days = r['time'].dt.date().to_list()
        for k in range(1, n):
            if days[k] != days[k - 1]:
                s.append(f'<line x1="{xs(k):.1f}" x2="{xs(k):.1f}" y1="{y0}" y2="{y0+ph}" class="grid"/>')
        s.append(f'<polyline points="{" ".join(f"{xs(k):.1f},{yv(v):.1f}" for k, v in enumerate(d))}" class="ln accent" stroke-width="1.6"/>'
                 f'<text x="{x0+pw}" y="{y0+ph+16}" text-anchor="end" class="sub">最后 5 个交易日（竖线 = 换日）</text>')
    s.append('</svg>')
    return ''.join(s)


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--muted:#a9a69b;--accent:#2f6fd0;--axis:#9a978e;--c1:#2f6fd0;--c2:#b9b6ab}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c;--c1:#5b93e6;--c2:#6a675f}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c;--c1:#5b93e6;--c2:#6a675f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.meta{color:var(--sub);font-size:13px;margin-bottom:12px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--bg)}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}figcaption{color:var(--sub);font-size:12px;margin-top:6px}
figure .title{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
svg .axis{stroke:var(--axis)}svg .grid{stroke:var(--line)}svg .frame{fill:none;stroke:var(--line)}svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}
svg .c1{fill:var(--c1)}svg .c2{fill:var(--c2)}svg text{fill:var(--fg);font-size:12px}svg .sub{fill:var(--sub)}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    mr = json.load(open(os.path.join(O, 'model_results.json'), encoding='utf-8'))
    t = rd('exit_vs_placebo')
    bp = rd('exit_by_product')
    it = rd('intraday')
    dl = rd('china_daily')
    evs = rd('china_events').select('symbol', 'lambda')
    pk = (dl.join(evs, on='symbol').with_columns(pl.col('distortion').abs().alias('a')).sort('a', descending=True)
            .group_by('symbol', maintain_order=True).first()
            .with_columns(((pl.col('case') == 1) & (pl.col('lambda') < 1) | (pl.col('case') == 2) & (pl.col('lambda') > 1)).alias('同向')))
    e = t['压力月_end'].to_numpy()
    from exit_test import nw_t as nw
    n10 = ((t['压力月_max_abs'] > 0.10).sum(), (t['对照月_max_abs'] > 0.10).sum())
    p = []
    p.append('<h1>中国商品期货交割前挤仓复现报告</h1><div class="meta">复现论文：Iosif Zimbidis &amp; Ronnie Sircar，《Negative Oil &amp; Nickel Squeeze: A Feedback Model for Extreme Commodity '
             'Futures Prices》，arXiv:2610.00951，2026 · 中国商品期货 2005–2026 所有合约到期 · 日线与 5 分钟数据</div>')
    p.append(f"<p class=\"lead\">论文的反馈模型在数学上完整复现（期权价值与 WTI 的隐含失衡 0.87 一致）。论文描述的机制在中国确实存在：个人客户须在交割月前平仓，"
             f"这个月里近月相对次月的偏离明显大于前一个月（偏离超过 10% 的到期 {n10[0]} 次对 {n10[1]} 次），以被压低为主，且集中在平仓截止前最后两三天。"
             f"但中国的偏离幅度远小于原油跌成负数、镍价翻倍这样的极端事件；论文用来反推“挤压程度”的方法依赖对正常价格的假设，只适合解释个别事件，不适合系统比较。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{mr['wti']:.3f}</b><span>WTI 隐含失衡（论文 0.87）</span></div>"
             f"<div class=\"kpi\"><b>{n10[0]} : {n10[1]}</b><span>偏离 &gt;10% 的到期：压力月 vs 对照月</span></div>"
             f"<div class=\"kpi\"><b>{(e < 0).mean():.0%}</b><span>压力月末近月被压低的比例</span></div>"
             f"<div class=\"kpi\"><b>{t['压力月_oi_change'].median():.0%}</b><span>压力月近月持仓变化（中位数）</span></div></div>")
    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>2020 年 4 月 20 日，WTI 原油 5 月合约在到期前一天跌到 −37.63 美元；2022 年 3 月 8 日，LME 镍价一天内涨到 10 万美元以上。论文认为两者有共同的机制：'
             '市场上有一部分人（金融机构、投机者）不能进行实物交割，到期前必须平仓。如果这部分人净多头，就必须集中卖出（原油，近月被压低，Case 1）；'
             '如果净空头，就必须集中买入（镍，近月被抬高，Case 2）。对手只有少数能交割的人，价格被扭曲。</p>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['模型', '观测近月价 = 正常价 ∓ ε × 展期期权价值；期权收益依赖观测价，形成反馈；显式解为 Margrabe 交换期权公式 + 一元方程', '同左，数值实现并逐项核对'],
        ['ε', '不能交割一方的净头寸 ÷ 总持仓', '同左'],
        ['实证', '给定观测价、次月价，假设正常价 = 初始比值 × 次月价，反推所需的 ε（式 57）；WTI 日内 5 分钟逐点反推', '同左，用于中国每一次合约到期，并对三个事件做 5 分钟日内反推'],
        ['中国对应', '—', '个人客户不能进入交割月 ⇒ 交割月前一个月是“压力月”；再往前一个月作为“对照月”'],
        ['补充检验', '论文结论中提出的方向：比较有 / 无平仓压力时的偏离', '压力月 vs 对照月，不依赖模型']]))

    p.append('<h2>一、论文数值的复现</h2>')
    f1 = mr['fig1']
    nk = list(mr['nickel'].values())
    mc = mr['mc']
    p.append(table(['项目', '论文', '复现'], [
        ['Figure 1：无反馈时的期权价值', '约 3.10', f"{f1['0'][0]:.2f}"],
        ['Figure 1：ε = 0.95 时 V+ / V−', '11.25 / 8.06', f"{f1['0.95'][0]:.2f} / {f1['0.95'][1]:.2f}"],
        ['WTI 2020-04-20 隐含失衡', '0.87', f"{mr['wti']:.3f}"],
        ['LME 镍 2022-03-08 隐含失衡', '0.9346', f"{nk[0]:.4f}（按表中剩余期限 1/4 年）；{nk[1]:.4f}（按剩余 1 天）"],
        ['Monte Carlo 跌破零比例（μ = 0）', '61.0%', f"{mc['2|0.0']:.1%}（每天 2 步）；{mc['12|0.0']:.1%}（每天 12 步）"]]))
    p.append('<ul><li>期权价值与 WTI 完全一致。</li><li>镍：按论文 Table 2 写的剩余期限 1/4 年算出 0.836；按剩余 1 天算出 0.934，与论文的 0.9346 一致。论文表中的期限与实际计算不一致。</li>'
             '<li>Monte Carlo：跌破零的比例随时间步长变细而上升（离散时点才能观测到穿越），论文的数字对应较细的步长。</li></ul>')

    p.append('<h2>二、照搬反推方法：逐个事件看并不可靠</h2>')
    same, opp = pk.filter('同向'), pk.filter(~pl.col('同向'))
    q = lambda s, x: f"{s['eps'].quantile(x):.2f}"
    p.append(f"<p>对 {pk.height} 次合约到期，在压力月中偏离最大的那天按式 (57) 反推 ε：</p>")
    p.append(table(['情形', '到期次数', 'ε 的 25% / 中位数 / 75% 分位'], [
        ['挤压方向与原有价差同向（原油、镍都属于这种）', same.height, f"{q(same, .25)} / {q(same, .5)} / {q(same, .75)}"],
        ['方向相反', opp.height, f"{q(opp, .25)} / {q(opp, .5)} / {q(opp, .75)}"]]))
    p.append('<p>反推公式是 ε = 偏离 ÷（偏离 + 次月价 × 期权价值）。方向相反时期权是虚值，临近到期价值趋近于零，一点点偏离就让 ε 接近 1，没有区分度；'
             '即使方向相同，方法也把价差的全部变化都归因于被迫平仓。所以这个方法适合回答“要多大的平仓压力才能解释某个极端价格”，不适合在大量事件之间比较。</p>')

    p.append('<h2>三、系统检验：有平仓压力的月份偏离更大</h2>')
    p.append(f'<figure><div class="title">个人强制平仓的月份，大偏离明显更多</div>{bars_svg(t)}'
             '<figcaption>来源：exit_test.py · 偏离 = 近月相对“窗口开始前近月/次月比值不变”基准的最大绝对偏离；同一合约在两个月份各算一次</figcaption></figure>')
    p.append(table(['', '压力月', '对照月'], [
        ['最大偏离（中位数）', f"{t['压力月_max_abs'].median()*100:.2f}%", f"{t['对照月_max_abs'].median()*100:.2f}%"],
        ['最大偏离（99% 分位）', f"{t['压力月_max_abs'].quantile(.99)*100:.1f}%", f"{t['对照月_max_abs'].quantile(.99)*100:.1f}%"],
        ['近月持仓变化（中位数）', f"{t['压力月_oi_change'].median():.0%}", f"{t['对照月_oi_change'].median():.0%}"]]))
    p.append(f"<ul><li>{t.height} 次合约到期中，{((t['压力月_max_abs'] - t['对照月_max_abs']) > 0).mean():.0%} 的压力月偏离大于对照月。</li>"
             f"<li><b>方向</b>：压力月末近月被压低的占 {(e < 0).mean():.0%}，平均 {e.mean()*100:+.2f}%（t={nw(e):.1f}），对应论文 Case 1：不能交割的一方净多、被迫卖出，与中国散户偏多头一致。</li>"
             '<li><b>品种</b>：偏离最明显的是交割门槛高、交割品标准严或难储存的品种；仓单体系成熟的金属几乎没有，与“交割受限才会被挤压”的机制吻合。</li></ul>')
    p.append(table(['品种', '到期次数', '压力月比对照月多出的偏离（中位数）', '压力月偏离（中位数）'],
                   [[r['product'], r['n'], f"{r['压力月多出的偏离%（中位数）']:.2f}%", f"{r['压力月偏离%']:.2f}%"] for r in pl.concat([bp.head(8), bp.tail(4)]).iter_rows(named=True)]))

    p.append('<h2>四、日内分析：挤压集中在最后两三天</h2>')
    p.append(f'<figure><div class="title">近月相对基准的偏离，在个人平仓截止前最后两天跳得最明显</div>{intraday_svg(it)}'
             '<figcaption>来源：intraday.py · 5 分钟数据；基准 = 压力月开始前近月/次月比值 × 次月价</figcaption></figure>')
    rows = []
    for sym, lab in (('LU2410', '低硫燃料油 2024-09（被抬高，像镍）'), ('A1801', '豆一 2017-12（被压低，像原油）'), ('SA2005', '纯碱 2020-04（被压低，像原油）')):
        a = it.filter((pl.col('事件') == sym) & pl.col('版本').str.starts_with('A')).sort('time').row(-1, named=True)
        b = it.filter((pl.col('事件') == sym) & pl.col('版本').str.starts_with('B')).sort('time').row(-1, named=True)
        rows.append([lab, f"{a['distortion']*100:+.1f}% / {b['distortion']*100:+.1f}%", f"{a['eps']:.2f} / {b['eps']:.2f}"])
    rows += [['论文：WTI 2020-04-20', '跌至 −37.63 美元', '0.87'], ['论文：LME 镍 2022-03-08', '涨至 10.1 万美元', '0.93']]
    p.append(table(['事件', '最后时刻偏离（基准 A / B）', '隐含失衡 ε（A / B）'], rows))
    p.append('<ul><li>基准 A：压力月开始前的比值；基准 B：最后一周首日开盘的比值（论文的做法）。同一事件 ε 随基准变化很大，正是论文自己承认的局限。</li>'
             '<li>方向与原有价差一致时，低硫燃料油的 ε（0.90）与原油、镍同一量级；豆一、纯碱较小。</li>'
             '<li>纯碱与 WTI 同处 2020 年 4 月，都是需求骤降、库存积压，纯碱近月只被压低约 12%，没有出现负价格。同月上海原油 SC2005 相对 SC2006 的比值也只从 0.977 降到 0.94。</li>'
             '<li>反例：焦煤 JM2201（2021-12）原有价差为贴水而近月被压低，期权虚值，ε 一直接近 1；且偏离在月中已形成，更像煤价大跌带动的正常收敛。</li></ul>')

    p.append('<h2>结论</h2><ul><li><b>模型</b>：数学上完整复现。它能解释“要多大的平仓压力才产生这样的价格”，但反推的 ε 依赖对正常价格的假设，不能当作真实持仓失衡的测量。</li>'
             '<li><b>机制在中国存在</b>：个人强制平仓的月份近月偏离明显更大、以被压低为主、集中在截止前最后两三天，且在交割困难的品种上更明显。</li>'
             '<li><b>程度远小于海外极端事件</b>：中国最大的偏离不到 30%，没有负价格。可能因为个人须提前一个月整体退出，压力分散；交易所对临近交割合约提高保证金、限制持仓（推测，未验证）。</li>'
             '<li><b>与三因子研究的联系</b>：按论文方法持有次近月时动量与 carry 特别强，其中可能部分来自交割前被迫平仓造成的价格扭曲，而这部分收益个人投资者按规则拿不到。</li></ul>')
    p.append('<h2>局限</h2><ul><li>交割前近月本来就要向现货收敛；没有现货数据，无法把被迫平仓造成的偏离与正常的基差收敛完全分开。'
             '上面的证据（时间集中在截止前、方向以压低为主、与交割难度相关）支持挤仓机制，但不能排除基差收敛的作用。</li>'
             '<li>没有“谁能交割”的持仓数据，ε 无法独立测量。</li><li>持仓降得越少的合约偏离反而越大，原因未明，可能与交易清淡有关。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [['<code>model.py</code>', '模型公式（Margrabe、反馈解、反推 ε、Monte Carlo）与论文数值复现'],
                                     ['<code>china_events.py</code>', '中国每一次合约到期的逐日反推'], ['<code>exit_test.py</code>', '压力月 vs 对照月；2020 年 4 月上海原油'],
                                     ['<code>intraday.py</code>', '三个事件与一个反例的 5 分钟日内反推'], ['<code>build_report.py</code>', '生成本报告'],
                                     ['<code>output/</code>', '各步结果；*_log.txt 为运行日志']]))
    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>交割前挤仓复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
