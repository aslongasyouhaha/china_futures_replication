"""生成波动率管理组合复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 读取。输出: report.html"""
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vmp as V  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = V.OUT


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def cum_panel(name, x0, y0, pw, ph):
    x = rd(f'monthly_{name}').sort('m')
    f = x['f'].to_numpy()
    fs, _ = V.managed(f, 1 / x['rv2_lag'].to_numpy())
    a, b = np.cumsum(np.log1p(f)), np.cumsum(np.log1p(fs))
    lo, hi = min(a.min(), b.min(), 0) - 0.05, max(a.max(), b.max()) + 0.05
    n = len(f)
    xs = lambda i: x0 + i / (n - 1) * pw
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    s = [f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" class="frame"/><text x="{x0+6}" y="{y0+16}" class="lbl">{name}</text>',
         f'<line x1="{x0}" x2="{x0+pw}" y1="{yv(0):.1f}" y2="{yv(0):.1f}" class="axis"/>']
    for t in np.arange(np.ceil(lo * 2) / 2, hi, 0.5):
        s.append(f'<text x="{x0-5}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{np.exp(t):.1f}×</text>')
    ms = x['m'].to_list()
    for i, d in enumerate(ms):
        if d.month == 1 and d.year % 4 == 0:
            s.append(f'<text x="{xs(i):.1f}" y="{y0+ph+15}" text-anchor="middle" class="sub">{d.year}</text>')
    for v, cls in ((a, 'muted'), (b, 'accent')):
        s.append(f'<polyline points="{" ".join(f"{xs(i):.1f},{yv(y):.1f}" for i, y in enumerate(v))}" class="ln {cls}" stroke-width="2"/>')
    return ''.join(s)


def cum_svg():
    W, Hh = 760, 300
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="买入持有与波动率管理的累计收益">',
         cum_panel('沪深300', 50, 20, 300, 230), cum_panel('商品平均', 430, 20, 300, 230),
         f'<line x1="250" x2="272" y1="{Hh-10}" y2="{Hh-10}" class="ln muted" stroke-width="2"/><text x="278" y="{Hh-6}" class="lbl">买入持有</text>'
         f'<line x1="360" x2="382" y1="{Hh-10}" y2="{Hh-10}" class="ln accent" stroke-width="2"/><text x="388" y="{Hh-6}" class="lbl">波动率管理（同样波动）</text>', '</svg>']
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
svg .axis{stroke:var(--axis)}svg .frame{fill:none;stroke:var(--line)}svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}svg .ln.muted{stroke:var(--muted)}
svg text{fill:var(--fg);font-size:12px}svg .sub{fill:var(--sub)}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    t1 = {r['因子']: r for r in rd('table1').iter_rows(named=True)}
    f1 = rd('figure1')
    t2 = rd('table2')
    sp = rd('subperiods')
    vs = rd('variants')
    rs = rd('regime_split')
    ry = {r['因子']: r for r in rd('annual_vol_return').iter_rows(named=True)}
    cy = rd('commodity_vol_years')
    R = lambda f, k: rs.filter((pl.col('因子') == f) & (pl.col('做法') == k)).row(0, named=True)
    p = []
    p.append('<h1>中国期货波动率管理组合复现报告</h1><div class="meta">复现论文：Alan Moreira &amp; Tyler Muir，《Volatility-Managed Portfolios》，'
             '<i>Journal of Finance</i>，2017 · 中国股指、国债、商品期货及 TSMOM、carry 因子，样本至 2026-08 · 论文口径（不计成本）</div>')
    p.append(f"<p class=\"lead\">在中国股指上，论文的结论成立：按上月方差反向调整仓位，沪深 300 的 α 为每年 {t1['沪深300']['α年化']:.1f}%"
             f"（t={t1['沪深300']['t_α']:.2f}），夏普从 {t1['沪深300']['原夏普']:.2f} 升到 {t1['沪深300']['管理后夏普']:.2f}，与论文的美国股市相当。"
             f"在商品上则无效甚至有害：商品平均因子的 α 为 {t1['商品平均']['α年化']:.1f}%。拆开看，论文的机制（短期波动突然放大时减仓）在股指上很有效，"
             f"中证 500 只用这部分时 α 达 {R('中证500', '只用短期偏离')['α年化']:.0f}%；商品亏在另一部分：中国商品的高波动时期多是牛市，按波动减仓等于在牛市里减仓。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{t1['沪深300']['α年化']:.1f}%</b><span>沪深 300 的 α（论文美国股市 4.9%）</span></div>"
             f"<div class=\"kpi\"><b>{t1['沪深300']['原夏普']:.2f} → {t1['沪深300']['管理后夏普']:.2f}</b><span>沪深 300 夏普</span></div>"
             f"<div class=\"kpi\"><b>{t1['商品平均']['α年化']:.1f}%</b><span>商品平均因子的 α（无效）</span></div>"
             f"<div class=\"kpi\"><b>{R('中证500', '只用短期偏离')['α年化']:.0f}%</b><span>中证 500 只按短期波动偏离择时的 α（t={R('中证500', '只用短期偏离')['t_α']:.1f}）</span></div></div>")
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['因子', '美国股市、市值、价值、动量、盈利、投资、ROE、BAB 等股票因子，外汇 carry；1926–2015', '沪深 300、中证 500、10 年国债期货（主力连续），商品平均（等权做多商品主力），TSMOM、商品 carry（前两篇复现的论文口径日收益）'],
        ['波动率管理', 'f^σ = c ÷ RV²（上月）× f（下月）；RV² 为上月日收益去均值后的平方和；c 使管理后与原组合波动相同', '同左；每月至少 15 个交易日'],
        ['检验', '管理后组合对原因子回归：α、评估比率、新夏普', '同左；α 的 t 值用 Newey-West'],
        ['其他', 'Figure 1 按上月波动分组；多因子均值-方差组合；30 年分段；NBER 衰退期；降低换手的变体与交易成本；杠杆约束', '同左（分段改为前后两半）；中国没有官方衰退划分，未做衰退期检验'],
        ['补充', '—', '把上月方差拆成长期水平与短期偏离，分别择时，解释商品上为何无效']]))

    p.append('<h2>一、主结果：股指有效，商品无效</h2>')
    p.append(table(['因子', '起始', 'α（年化）', 't', 'β', '评估比率', '原夏普', '管理后夏普', '月数'],
                   [[n, r['起始'][:7], f"<b>{r['α年化']:.1f}%</b>", f"{r['t_α']:.2f}", f"{r['β']:.2f}", f"{r['评估比率']:.2f}", f"{r['原夏普']:.2f}",
                     f"{r['管理后夏普']:.2f}", r['月数']] for n, r in t1.items()]))
    p.append(f'<figure><div class="title">同样的波动下，沪深 300 做波动率管理后累计收益明显更高；商品平均因子则更差</div>{cum_svg()}'
             '<figcaption>来源：vmp.py · 月度对数收益累计；管理后组合按全样本缩放到与原组合相同的波动</figcaption></figure>')
    p.append('<p>论文中美国股市的 α 为 4.9%，夏普提高约 25%；沪深 300 的 α 为 6.4%，夏普提高约一半，结论一致。TSMOM 的 α 为 4.6%（t=1.85），接近显著；'
             '中证 500、国债与商品 carry 的 α 不显著；商品平均因子为负。</p>')

    p.append('<h2>二、论文的机制：波动大时，收益没有跟着变高</h2>')
    rows = []
    for n in ('沪深300', '商品平均', 'TSMOM'):
        g = f1.filter(pl.col('因子') == n)
        rows += [[n if i == 0 else '', r['上月波动分组'], f"{r['平均收益']*100:.1f}%", f"{r['收益波动']*100:.1f}%", f"{r['当月已实现波动']*100:.1f}%", f"{r['收益÷方差']:.1f}"]
                 for i, r in enumerate(g.iter_rows(named=True))]
    p.append(table(['因子', '上月波动分组', '下月平均收益（年化）', '下月收益波动', '下月已实现波动', '收益 ÷ 方差'], rows))
    p.append('<p>波动率很好预测：上月波动越高，下月波动也越高。论文的关键在于，波动高的时候平均收益没有相应变高，所以“每承担一份风险的收益”变差，应少持有。'
             '沪深 300 符合这一点：上月波动最高的一组，下月平均收益为负。商品平均因子不符合：上月波动最低的一组，下月平均收益反而是负的，策略恰好在这时加仓。</p>')

    p.append('<h2>三、为什么商品上无效</h2>')
    p.append('<p>把上月方差拆成两部分：<b>长期水平</b>（过去 12 个月的平均方差，反映“处在高波动还是低波动的时期”）和<b>短期偏离</b>（上月方差 ÷ 长期水平，反映“最近是否突然比平时更动荡”），分别只用一部分调整仓位：</p>')
    p.append(table(['因子', '论文做法', '只用长期水平', '只用短期偏离', '年度波动与当年收益的相关'],
                   [[n] + [f"{R(n, k)['α年化']:+.1f}%（t={R(n, k)['t_α']:.1f}）" for k in ('论文做法（1/上月方差）', '只用长期水平', '只用短期偏离')]
                    + [f"{ry[n]['年度波动与当年收益相关']:+.2f}（{ry[n]['年数']} 年）"] for n in V.FACTORS]))
    p.append('<p>商品平均因子各年份落入“上月波动最低 / 最高 20%”的月数：</p>')
    p.append(table(['年份', '最低 20% 的月数', '最高 20% 的月数', '年份', '最低 20% 的月数', '最高 20% 的月数'],
                   [[a[0], a[1], a[2]] + ([b[0], b[1], b[2]] if b else ['', '', ''])
                    for a, b in zip(cy.select('年', '最低', '最高').head((cy.height + 1) // 2).rows(),
                                    list(cy.select('年', '最低', '最高').tail(cy.height // 2).rows()) + [None])]))
    p.append(f"<ul><li><b>股指的有效部分是短期偏离</b>：股市突然变得比平时动荡（如 2015 年股灾）时，接下来往往还会差，及时减仓就能避开。中证 500 只用这部分时 α 达 "
             f"{R('中证500', '只用短期偏离')['α年化']:.0f}%（t={R('中证500', '只用短期偏离')['t_α']:.1f}），但长期水平部分为 {R('中证500', '只用长期水平')['α年化']:.0f}%，两者抵消，所以论文做法在中证 500 上看起来无效。</li>"
             f"<li><b>商品的亏损来自长期水平</b>：中国商品的低波动时期集中在 2013–2014、2018–2019、2025 年，多是缓慢阴跌或横盘；高波动时期集中在 2008–2009、2016–2017、2021–2022 年，"
             f"除 2008 年外多是大牛市。按波动减仓，等于在慢熊市里加仓、在牛市里减仓。只用短期偏离时，商品既不赚也不亏（{R('商品平均', '只用短期偏离')['α年化']:+.1f}%）。</li>"
             '<li>年度上，商品“波动高、当年收益也高”的相关只有 +0.15（20 年），统计上并不强，以上是符合数据的解释，而非强证据。'
             '另外检查过节假日月份交易日少导致方差偏低、少数极端月份等原因，都不是主因。</li></ul>')

    p.append('<h2>四、多因子组合</h2>')
    p.append(table(['组合', '起始', '样本内最优权重', 'α（年化）', 't', '原夏普', '管理后夏普'],
                   [[r['组合'], r['起始'][:7], r['权重'], f"{r['α年化']:.2f}%", f"{r['t_α']:.2f}", f"{r['原夏普']:.2f}", f"{r['管理后夏普']:.2f}"] for r in t2.iter_rows(named=True)]))
    p.append('<p>论文中，各种因子组合做波动率管理后都有正的 α。中国只有包含国债和中证 500 的组合（2015 年起）α 显著；以商品为主的组合几乎没有改善，因为商品因子本身不适合按波动择时。</p>')

    p.append('<h2>五、分段</h2>')
    p.append(table(['因子', '时段', 'α（年化）', 't', '原夏普', '管理后夏普'],
                   [[r['因子'], r['时段'], f"{r['α年化']:.1f}%", f"{r['t_α']:.2f}", f"{r['原夏普']:.2f}", f"{r['管理后夏普']:.2f}"] for r in sp.iter_rows(named=True)]))
    p.append('<p>沪深 300 的 α 主要来自前半段（含 2015 年股灾），后半段为 2.5%，不显著；TSMOM 也是前半段显著。论文同样发现，波动变化小的时期效果弱，波动不变时策略就等于买入持有。</p>')

    p.append('<h2>六、降低换手与交易成本</h2>')
    p.append(table(['因子', '做法', '权重中位数', '99% 分位', '月均换手', 'α（年化）', 't', '管理后夏普', '盈亏平衡成本'],
                   [[r['因子'], r['做法'], f"{r['权重中位数']:.2f}", f"{r['权重 99%分位']:.1f}", f"{r['月均换手 |Δw|']:.2f}", f"{r['α年化']:.1f}%", f"{r['t_α']:.2f}",
                     f"{r['管理后夏普']:.2f}", f"{r['盈亏平衡成本bp']:.0f}bp" if r['α年化'] > 0 else '—'] for r in vs.filter(pl.col('因子').is_in(['沪深300', 'TSMOM'])).iter_rows(named=True)]))
    p.append('<p>与论文一样，用波动代替方差、用 AR(1) 预测方差，都能明显降低极端杠杆（99% 分位从约 5 倍降到 2.5~3 倍），α 仍然为正；加杠杆上限 1 或 1.5 后 α 变小但夏普基本不变。'
             '沪深 300 的盈亏平衡成本约每单位换手 100bp，远高于股指期货的实际交易成本（约 1~2bp），成本不是问题。</p>')

    p.append('<h2>结论</h2><ul>'
             f"<li><b>股指上成立</b>：沪深 300 做波动率管理后 α {t1['沪深300']['α年化']:.1f}%、夏普提高约一半，与论文一致；起作用的是短期波动突然放大时减仓。</li>"
             '<li><b>商品上不成立</b>：商品平均因子与商品 carry 做波动率管理后没有改善甚至变差。原因是中国商品的高波动时期多是牛市，按波动水平减仓等于在牛市里减仓；只按短期偏离择时则既不赚也不亏。</li>'
             '<li><b>效果集中在少数时期</b>：股指的 α 主要来自 2015 年前后的大波动时期，近年不显著。</li></ul>')
    p.append('<h2>与论文的差异</h2><ul><li>因子不同：中国没有长期、公开的股票因子日收益，这里用期货上的股指、国债、商品及前两篇复现的 TSMOM、carry。</li>'
             '<li>样本短：股指 2010 年起、中证 500 与国债 2015 年起、商品 2005 年起；论文从 1926 年开始。</li><li>未做衰退期检验、期权实现、VAR 与均衡模型部分。</li>'
             '<li>补充的“长期水平 / 短期偏离”拆分是本次新增的分析，不是论文的内容。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [['<code>vmp.py</code>', '各因子日收益与月度已实现方差；Table I、Figure 1、Table II、分段、降低换手的变体、长期水平 / 短期偏离拆分'],
                                     ['<code>build_report.py</code>', '生成本报告'], ['<code>output/*.parquet</code>', '各步结果；output_log.txt 为运行日志']]))
    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>波动率管理组合复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
