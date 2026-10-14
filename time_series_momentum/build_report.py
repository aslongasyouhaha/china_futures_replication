"""生成 TSMOM 复现报告（离线 HTML，图表为内嵌 SVG）。输出: report.html"""
import html
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report.html')

# 逐年收益（%）：论文口径（无摩擦、月末成交）与实盘版（1 亿、次日 VWAP、扣成本）
YEARS = list(range(2006, 2027))
PAPER = [5.92, 12.01, 32.96, -19.63, 7.88, -4.67, -3.92, 27.62, 48.15, 22.88, -6.27, 1.55, 2.83, -0.43, 24.19, 18.95,
         11.24, 1.81, 12.78, 8.67, -8.80]
REAL = [1.7, 5.9, 10.3, -19.0, -1.1, -9.6, -2.8, 19.6, 35.1, 20.5, -10.9, 0.4, -0.4, -3.7, 17.8, 15.0, 4.9, -4.0, 7.9,
        14.0, -12.8]
# 从论文口径到实盘的夏普
BRIDGE = [('论文口径', 0.61, True), ('次日 VWAP', 0.53, False), ('成交额门槛', 0.48, False), ('整手与上限', 0.45, False),
          ('调仓成本', 0.43, False), ('换月成本', 0.38, False), ('冲击成本', 0.31, False), ('实盘版', 0.31, True)]


def yearly_svg():
    W, H, left, y0, ph = 760, 380, 56, 70, 250
    lo, hi = min(-20, min(PAPER + REAL)), max(PAPER + REAL)
    k = ph / (hi - lo)
    zy = y0 + hi * k
    yv = lambda v: zy - v * k
    band = (W - left - 24) / len(YEARS)
    bw = band * 0.36
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="逐年收益：论文口径与实盘版">']
    s.append(f'<rect x="{left}" y="40" width="12" height="12" class="muted"/><text x="{left+18}" y="50" class="lbl">论文口径（无摩擦、月末成交）</text>')
    s.append(f'<rect x="{left+260}" y="40" width="12" height="12" class="accent"/><text x="{left+278}" y="50" class="lbl">实盘版（1 亿、次日 VWAP、扣成本）</text>')
    for t in (-20, 0, 20, 40):
        cls = 'axis' if t == 0 else 'grid'
        s.append(f'<line x1="{left}" x2="{W-24}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="{cls}"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t}</text>')
    for i, yr in enumerate(YEARS):
        for j, (val, cls) in enumerate(((PAPER[i], 'muted'), (REAL[i], 'accent'))):
            x = left + i * band + (band * 0.5 if j else band * 0.12)
            y = yv(val) if val >= 0 else zy
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{abs(val)*k:.1f}" class="{cls}">'
                     f'<title>{yr} {"实盘版" if j else "论文口径"}: {val:+.1f}%</title></rect>')
            if j and yr in (2008, 2009, 2014, 2016):
                ty = yv(val) - 5 if val >= 0 else yv(val) + 13
                s.append(f'<text x="{x+bw/2:.1f}" y="{ty:.1f}" text-anchor="middle" class="lbl">{val:+.0f}</text>')
        s.append(f'<text x="{left+i*band+band/2:.1f}" y="{y0+ph+22}" text-anchor="middle" class="sub small">{yr}</text>')
    s.append('</svg>')
    return ''.join(s)


def bridge_svg():
    W, H, left, y0, ph, top_v = 760, 340, 40, 40, 240, 0.7
    base = y0 + ph
    yv = lambda v: base - v / top_v * ph
    band = (W - left - 40) / len(BRIDGE)
    bw = band * 0.6
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="夏普从 0.61 降到 0.31 的拆解">',
         f'<line x1="{left}" x2="{W-40}" y1="{base}" y2="{base}" class="axis"/>']
    prev = 0
    for i, (name, v, total) in enumerate(BRIDGE):
        x = left + i * band + (band - bw) / 2
        if total:
            y, h, cls, lab, ly = yv(v), base - yv(v), 'muted', f'{v:.2f}', yv(v) - 8
        else:
            y, h, cls, lab, ly = yv(prev), yv(v) - yv(prev), 'bad', f'−{prev - v:.2f}', yv(v) + 15
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,1):.1f}" class="{cls}">'
                 f'<title>{name}：夏普 {v:.2f}</title></rect>')
        if i < len(BRIDGE) - 1:
            s.append(f'<line x1="{x+bw:.1f}" x2="{x+band:.1f}" y1="{yv(v):.1f}" y2="{yv(v):.1f}" class="axis dash"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{ly:.1f}" text-anchor="middle" class="lbl">{lab}</text>'
                 f'<text x="{x+bw/2:.1f}" y="{base+22}" text-anchor="middle" class="sub">{html.escape(name)}</text>')
        prev = v
    s.append('</svg>')
    return ''.join(s)


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--muted:#c5c3b8;--accent:#2f6fd0;--bad:#d1605e;--axis:#9a978e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#5d5b55;--accent:#5b93e6;--bad:#d77a77;--axis:#77746c}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#5d5b55;--accent:#5b93e6;--bad:#d77a77;--axis:#77746c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.meta{color:var(--sub);font-size:13px;margin-bottom:20px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--bg)}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}figcaption{color:var(--sub);font-size:12px;margin-top:6px}
figure .title{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
svg .muted{fill:var(--muted)}svg .accent{fill:var(--accent)}svg .bad{fill:var(--bad);fill-opacity:.65}svg .grid{stroke:var(--line)}svg .axis{stroke:var(--axis)}svg .dash{stroke-dasharray:3 3}
svg text{fill:var(--fg);font-size:12px}svg .sub{fill:var(--sub)}svg .small{font-size:10px}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    p = []
    p.append('<h1>中国期货时间序列动量复现报告</h1><div class="meta">复现论文：Moskowitz, Ooi, Pedersen,《Time Series Momentum》，'
             '<i>Journal of Financial Economics</i>，2012 · 样本 2006-01 至 2026-09-11</div>')
    p.append('<p class="lead">中国期货存在时间序列动量，但按实盘条件执行后，夏普从论文口径的 <b>0.61</b> 降到 <b>0.31</b>（t 值 1.40），'
             '统计上不再显著；2016 年以后只有 0.24。</p>')
    p.append('<div class="kpis"><div class="kpi"><b>0.61</b><span>论文口径夏普</span></div><div class="kpi"><b>0.31</b><span>实盘版夏普（资金 1 亿）</span></div>'
             '<div class="kpi"><b>2.12%</b><span>实盘版年化成本</span></div><div class="kpi"><b>−43%</b><span>实盘版最大回撤</span></div></div>')

    p.append('<h2>复现对象与数据</h2>')
    p.append(table(['项目', '论文', '本次复现'], [
        ['品种', '58 个全球期货（24 商品、9 股指、13 国债、12 外汇）', '中国 6 家交易所 93 个品种（含股指、国债期货）'],
        ['样本', '1985–2009（另有 1965 起的检验）', '2006-01 ~ 2026-09（数据 2005 年起，留一年计算信号）'],
        ['价格', '最活跃合约拼接的收益', '主力连续合约的后复权收盘价'],
        ['成本', '不扣除', '第一部分不扣除；第二部分按实盘条件扣除']]))

    p.append('<h2>第一部分：按论文口径复现</h2><p>按论文方法、不扣成本、月末收盘成交，2006 年以来夏普 0.61、t 值 2.76，动量效应显著存在，但弱于论文的大于 1。</p>')
    p.append('<ol><li>信号：每月末看过去 12 个月累计收益，为正做多、为负做空。</li><li>波动率：日收益的指数加权方差，中心质量 60 天，年化系数 243（论文用 261）。</li>'
             '<li>仓位：40% ÷ 年化波动率，使每个品种的风险贡献相近。</li><li>组合：当月可交易品种等权，持有 1 个月；品种需过去 60 日平均成交 ≥ 1000 手。</li></ol>')
    p.append(table(['区间', '夏普', '年化收益', '最大回撤', 't 值'], [
        ['2006 年起', '0.61', '9.2%', '−40.9%', '2.76'], ['2010 年起', '0.67', '9.6%', '−32.0%', '2.74'], ['2016 年起', '0.53', '6.3%', '−27.6%', '1.74']]))
    p.append(table(['方面', '论文', '中国'], [
        ['最优回看期', '12 个月', '1~9 个月（夏普 0.74~0.77）；12 个月 0.61，24 个月 0.32'],
        ['微笑形（大涨大跌时都赚钱）', '成立，t = 3.88', '成立，t = 3.30'],
        ['与“始终做多”的关系', '不相关', '不相关；始终做多的夏普只有 0.17'],
        ['分资产类别夏普', '各类都有效', '商品 0.57，股指 0.48，国债 0.18']]))
    p.append('<p>中国的趋势比海外短：1~9 个月的回看期明显好于论文用的 12 个月。</p>')

    p.append('<h2>逐年表现</h2>')
    p.append(f'<figure><div class="title">大跌年份实盘版仍赚钱：2008 年 +10%，2014 年 +35%</div><div class="meta">年度收益（%），2026 年截至 9 月 11 日</div>'
             f'{yearly_svg()}<figcaption>来源：tsmom.py 与 tsmom_real.py 回测输出 · k=12</figcaption></figure>')
    p.append('<p>危机保护在实盘版中依然存在；亏损集中在行情急速反转的年份（如 2009 年 V 形反弹、2016 年供给侧改革）。作为对照，始终做多组合 2008 年亏 49%、2014 年亏 30%。</p>')

    p.append('<h2>第二部分：实盘化设定</h2><p>论文不扣成本、不考虑资金规模和执行细节；在信号和仓位规则不变的前提下加入以下条件。只有冲击系数和容量上限仍是经验参数。</p>')
    p.append(table(['环节', '设定', '依据'], [
        ['资金规模', '1 亿元，净值逐日滚动', '按实际投入规模设定'],
        ['整手', '目标名义金额 ÷ 每手价值，四舍五入；不足半手不开仓', '合约乘数由“成交额 ÷（成交量 × 价格）”估计，与交易所规定一致'],
        ['流动性筛选', '60 日成交额中位数 ≥ 1 亿元/日', '按金额而非手数；用中位数避免偶发放量'],
        ['容量上限', '单品种持仓 ≤ 日均成交额的 2%，超出则缩小', '业内常见的保守值'],
        ['成交时点', '信号日的下一个交易日，按全天 VWAP 成交', '最接近机构实际做法'],
        ['手续费', '1bp', '主流品种交易所手续费约 0.1~0.5bp，取保守值'],
        ['价差成本', '每个品种每年的半个买卖价差，下限半跳', '用分钟数据的 Roll 方法估计；活跃品种价差中位数约 1.2 跳'],
        ['冲击成本', '0.5 × 日波动率 × √（成交手数 ÷ 日均成交手数）', '平方根冲击模型；系数 0.5 为经验值'],
        ['换月成本', '两条腿各计手续费、价差和冲击；冲击分别对比旧、新合约当天的成交量', '实盘必须换月；论文未计'],
        ['保证金', '商品和股指 12%，国债按期限 0.5%~3.5%', '检查占用，确认不会爆仓']]))

    p.append('<h2>实盘版结果</h2><p>资金 1 亿、次日 VWAP 成交、扣除全部成本后，2006 年以来夏普 0.31、年化收益 4.54%，年化成本 2.12%，其中换月占一半以上。</p>')
    p.append(table(['指标', '结果'], [
        ['年化收益', '4.54%'], ['年化波动', '14.7%'], ['夏普', '0.31（2010 年起 0.39，2016 年起 0.24）'], ['最大回撤', '−43.2%'], ['t 值', '1.40（不显著）'],
        ['年化成本：调仓手续费和价差', '0.33%'], ['年化成本：调仓冲击', '0.31%'], ['年化成本：换月', '1.48%（其中冲击约 0.81%）'],
        ['平均持仓', '36 个品种；每次调仓约 2.5 个品种被容量上限缩小'], ['保证金占净值', '中位数 25%，最高 40%'], ['名义杠杆', '中位数 2.6 倍']]))

    p.append('<h2>从论文口径到实盘</h2>')
    p.append(f'<figure><div class="title">实盘条件把夏普从 0.61 压到 0.31，成交时点、冲击和换月影响最大</div>'
             f'<div class="meta">灰色：夏普水平；红色：每一步带来的下降。k=12，2006 年起</div>{bridge_svg()}'
             f'<figcaption>来源：tsmom_real.py 回测输出 · 资金 1 亿</figcaption></figure>')

    p.append('<h2>敏感性</h2><p>结果对冲击系数和资金规模最敏感，对价差和容量上限不敏感；现实的夏普大约在 0.24~0.38 之间。</p>')
    p.append(table(['改变的设定', '夏普', '年化收益', '年化成本'], [
        ['定稿方案（资金 1 亿，k=12）', '0.31', '4.54%', '2.12%'], ['资金 1000 万', '0.37', '5.72%', '1.65%'], ['资金 10 亿', '0.20', '2.50%', '2.30%'],
        ['冲击系数 Y = 0', '0.38', '5.61%', '1.00%'], ['冲击系数 Y = 1', '0.24', '3.52%', '3.17%'], ['价差成本 × 0.5', '0.33', '4.82%', '1.81%'],
        ['价差成本 × 2', '0.27', '3.91%', '2.75%'], ['容量上限 1%', '0.32', '4.57%', '1.86%'], ['容量上限 5%', '0.30', '4.56%', '2.37%'],
        ['去掉成交额门槛', '0.30', '3.76%', '1.74%'], ['回看期 k = 3', '0.28', '4.02%', '2.65%'], ['回看期 k = 6', '0.35', '4.97%', '2.31%']]))

    p.append('<h2>结论</h2><ol>'
             '<li><b>效应存在，但实盘收益有限。</b>约一半的夏普被执行条件和成本吃掉，实盘夏普约 0.3，2016 年以后更弱。</li>'
             '<li><b>影响最大的三个环节</b>：成交时点（−0.08）、冲击成本（−0.07）、换月成本（−0.05）。</li>'
             '<li><b>换月是最大的成本来源</b>，年化 1.48%，主要来自每月换月的有色金属。</li>'
             '<li><b>短回看期的优势被成本抵消</b>：无摩擦时 1~9 个月最好，扣成本后 k = 6 最好。</li>'
             '<li><b>容量有限</b>：资金从 1 亿增至 10 亿，夏普从 0.31 降到 0.20。</li>'
             '<li><b>危机保护仍在</b>：2008、2014 年的单边大跌中，实盘版也分别赚了 10% 和 35%。</li></ol>')
    p.append('<h2>局限与改进方向</h2><ul><li>冲击系数 0.5 未经实盘数据校准。</li><li>价差由 Roll 方法从分钟成交价估计，只保证数量级；原始数据没有买一、卖一报价。</li>'
             '<li>未考虑涨跌停无法成交、交易所持仓限额、保证金临时上调。</li><li>手续费统一按 1bp。</li></ul>'
             '<ol><li>优化换月：有色金属持有远月合约、减少换月次数；用跨期价差单换月。</li><li>减少无效交易：仓位变化不大时不调整；换月和调仓放在同一天。</li>'
             '<li>多回看期组合：将 3、6、12 个月的信号平均。</li><li>改进风险分配；用 HAR 模型预测波动率（见 <code>research/har_rv</code>）。</li><li>叠加 Carry 信号。</li></ol>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [
        ['<code>tsmom.py</code>', '论文口径复现'], ['<code>tsmom_real.py</code>', '实盘版回测：定稿方案、拆解、敏感性'],
        ['<code>build_report.py</code>', '生成本报告'],
        ['<code>../common/</code>', '合约乘数、最小变动价位、买卖价差、每日 VWAP'], ['<code>G:\\CTA\\raw_data\\</code>', '期货数据']]))

    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>TSMOM 复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
