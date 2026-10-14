"""生成 Kang-Rouwenhorst-Tang 复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harq'))
from build_report import CSS, table  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def f(x, k=2):
    return '' if x is None or x != x else f'{x:.{k}f}'


def bt(r, c, k=3):
    return '' if r.get(f'b_{c}') is None else f"{f(r[f'b_{c}'], k)}（{f(r[f't_{c}'])}）"


def build():
    w = rd('weekly')
    t2, t3, t5, t6, t8, ts = rd('t2'), rd('t3'), rd('t5'), rd('t6'), rd('t8'), rd('subperiods')
    a = t2.filter(pl.col('设定') == 'Q_t ~ R_t').row(0, named=True)
    q1 = t3.filter((pl.col('因变量') == '第 t+1 周') & (pl.col('控制变量') == '无')).row(0, named=True)
    q1c = t3.filter((pl.col('因变量') == '第 t+1 周') & (pl.col('控制变量') != '无')).row(0, named=True)
    s14 = t5.filter((pl.col('窗口') == '第 1~4 日') & (pl.col('组') == '5 − 1')).row(0, named=True)
    s510 = t5.filter((pl.col('窗口') == '第 5~10 日') & (pl.col('组') == '5 − 1')).row(0, named=True)
    pre = t5.filter((pl.col('窗口') == '形成前两周') & (pl.col('组') == '5 − 1')).row(0, named=True)
    hpb2 = t6.filter((pl.col('设定') == '平滑 HP̄') & (pl.col('因变量') == '第 t+2 周')).row(0, named=True)

    p = []
    p.append('<h1>套保溢价与流动性溢价复现报告</h1><div class="meta">复现论文：Wenjin Kang, K. Geert Rouwenhorst &amp; Ke Tang，'
             '《A Tale of Two Premiums: The Role of Hedgers and Speculators in Commodity Futures Markets》，<i>Journal of Finance</i>，2020 · '
             f"中国商品期货，{w['we'].min()} 至 {w['we'].max()}，{w['product'].n_unique()} 个品种，周度 · 用交易所“前 20 名会员持仓”代表大户（方案 B）</div>")
    p.append(f"<p class=\"lead\">中国的结果与论文的故事<b>正好相反</b>。论文发现：投机者顺势交易、急于成交，买入后价格回落，他们向提供流动性的套保者付出溢价。"
             f"中国的前 20 名大户也顺势交易（净买入与当周收益的系数 {a['b_R']:.2f}，t = {a['t_R']:.1f}），但他们买入后价格<b>继续</b>上涨："
             f"按净买入五分组，最高组比最低组在之后第 1–4 天多涨 {s14['平均%']:.2f}%（t = {s14['t']:.2f}）、第 5–10 天多涨 {s510['平均%']:.2f}%（t = {s510['t']:.2f}），之后没有反转。"
             f"大户长期净空头越多，两周后收益越低（t = {hpb2['t_HPbar']:.2f}），也与论文中“套保者付保险费”的方向相反。也就是说，中国的大户不是在付流动性成本，"
             "更像是信息或趋势判断更好的一方，付钱的是前 20 名以外、以散户为主的交易者。这一效应集中在 2016 年以前，之后明显减弱。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{a['b_R']:.2f}（t {a['t_R']:.1f}）</b><span>大户净买入对当周收益：顺势（论文投机者同为正、套保者为负）</span></div>"
             f"<div class=\"kpi\"><b>{q1['b_Q']:.2f}（t {q1['t_Q']:.2f}）</b><span>大户净买入预测下周收益（论文投机者为负）</span></div>"
             f"<div class=\"kpi\"><b>{s510['平均%']:.2f}%（t {s510['t']:.2f}）</b><span>Q 最高组 − 最低组，形成后第 5–10 天</span></div>"
             f"<div class=\"kpi\"><b>{hpb2['b_HPbar']:.2f}（t {hpb2['t_HPbar']:.2f}）</b><span>平滑净空头预测第 t+2 周收益（论文套保者为正）</span></div></div>")

    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>用 CFTC 周度持仓把交易者分成商业（套保者）和非商业（投机者）。论文发现两种溢价：'
             '① 短期流动性溢价：投机者顺势交易、急于成交，他们买入的品种下周收益更低、卖出的更高；套保者逆势交易、提供流动性，赚取这部分收益（周度 t ≈ 5~6.6）。'
             '② 长期套保溢价：把套保压力做 52 周平滑后，套保者净空头越多，未来收益越高，符合 Keynes 的正常贴水理论。两者方向相反，所以不加区分时套保压力预测不出收益。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['持仓', 'CFTC 周报：商业 / 非商业 / 非报告', '交易所每日公布的前 20 名会员（期货公司）多空持仓；同一品种所有合约加总，剔除“期货公司会员”等汇总行；不区分交易者类型（用户选定方案 B）'],
        ['频率', '每周二', '每周最后一个交易日'],
        ['Q', '净多头周变化 / 上周总持仓', '同左（大户整体）；要求上周总持仓 ≥ 1000 手'],
        ['HP / 平滑 HP̄', '商业净空头 / 总持仓；52 周均值 / 当周总持仓', '大户净空头，同样的平滑'],
        ['收益', '近月期货（取曲线最活跃部分）', '主力合约后复权收益（中国最活跃的是主力而非近月）'],
        ['控制变量', '对数基差、S × 对标普 500 的特质波动、当周收益', '对数基差（主力与更远月最大持仓合约）、S × 对商品等权指数的特质波动、当周收益'],
        ['未做', '第四节风险环境（期权隐含波动、VIX）、第五节利润分解', '本次只做第二、三节核心检验']]))
    p.append('<p><b>解读上的限制</b>：这里检验的是“前 20 名大户 vs 其余交易者”，不是论文的“套保 vs 投机”；前 20 名每周成员会变，加总值带有成员更替的噪声。</p>')

    p.append('<h2>一、大户怎么交易（对应论文 Table II）</h2>')
    rows = [[r['设定'], r['周数'], bt(r, 'R'), bt(r, 'Rlag'), bt(r, 'Qlag'), f(r['R2%'], 1)] for r in t2.iter_rows(named=True)]
    p.append(table(['回归', '周数', '当周收益 R_t', '上周收益 R_{t−1}', '上周 Q_{t−1}', '平均 R²%'], rows))
    p.append('<p>大户净买入与当周、上周收益都显著正相关：涨了就加多，即顺势交易。论文中非商业（投机者）是这种模式，商业（套保者）相反。</p>')

    p.append('<h2>二、大户买卖之后的收益（对应论文 Table III）</h2>')
    rows = [[r['因变量'], r['控制变量'], r['周数'], bt(r, 'Q'), bt(r, 'B'), bt(r, 'SIV'), bt(r, 'R'), f(r['R2%'], 1)] for r in t3.iter_rows(named=True)]
    p.append(table(['因变量（%）', '控制变量', '周数', 'Q', '对数基差', 'S × 特质波动', 'R_t', '平均 R²%'], rows))
    p.append(f"<p>论文：商业交易者 Q 的系数显著为正（t = 4.95，加控制变量后 6.60），非商业显著为负。中国大户的 Q 系数为正（t = {q1['t_Q']:.2f}），"
             f"控制基差和当周收益后降到 t = {q1c['t_Q']:.2f}：大户买入的品种下周继续上涨，但其中一部分是动量本身。对数基差系数为负（升水越深收益越低），与 carry 一致。</p>")

    p.append('<h2>三、按净买入五分组（对应论文 Table V）</h2>')
    rows = []
    for win in t5['窗口'].unique(maintain_order=True):
        q = t5.filter(pl.col('窗口') == win)
        rows.append([win] + [f(q.filter(pl.col('组') == str(k))['平均%'][0], 3) for k in range(1, 6)] +
                    [f"{f(q.filter(pl.col('组') == '5 − 1')['平均%'][0], 3)}（{f(q.filter(pl.col('组') == '5 − 1')['t'][0])}）"])
    p.append(table(['窗口（累计收益 %）', '组 1（大户卖最多）', '2', '3', '4', '组 5（大户买最多）', '5 − 1（t）'], rows))
    p.append(f"<p>论文（商业交易者）：形成前两周组 5 跌 3.19%、组 1 涨 3.72%（逆势）；之后 5 − 1 为第 1–4 天 0.24%（t 3.65）、第 5–10 天 0.29%（t 3.73）。"
             f"中国大户：形成前两周组 5 比组 1 多涨 {pre['平均%']:.2f}%（顺势）；之后第 1–10 天继续同向，第 11–40 天没有反转。</p>")

    p.append('<h2>四、套保压力再检验（对应论文 Table VI、VIII）</h2>')
    rows = [[r['设定'], r['因变量'], r['周数'], bt(r, 'HP'), bt(r, 'HPbar'), bt(r, 'Q'), bt(r, 'B'), bt(r, 'R')] for r in t6.iter_rows(named=True)]
    p.append(table(['设定', '因变量（%）', '周数', 'HP', '平滑 HP̄', 'Q', '对数基差', 'R_t'], rows))
    p.append('<p>论文：HP 本身预测不了收益，平滑后显著为正（套保溢价）。中国：大户当周净空头 HP 不显著；平滑后为负，第 t+2 周显著（大户长期净空的品种之后更弱）。'
             '方向与论文相反，说明大户的长期净头寸不是在“买保险”，而是方向判断。</p>')
    rows = [[r['窗口'], r['组'], f(r['平均%'], 3), f(r['t'])] for r in t8.iter_rows(named=True)]
    p.append('<h3>平滑 HP̄ × Q 双重排序</h3>' + table(['窗口', '组合', '平均累计收益%', 't'], rows))

    p.append('<h2>五、分时期</h2>')
    rows = [[r['时期'], r['周数'], bt(r, 'Q'), bt(r, 'HPbar'), bt(r, 'B'), bt(r, 'R')] for r in ts.iter_rows(named=True)]
    p.append(table(['时期（因变量：第 t+1 周收益）', '周数', 'Q', '平滑 HP̄', '对数基差', 'R_t'], rows))
    p.append('<p>大户净买入的预测力集中在 2006–2015 年（控制平滑 HP̄ 后 t ≈ 2.4），2016 年以后消失。</p>')

    p.append('<h2>结论</h2>')
    p.append('<ul><li><b>中国大户的交易方式像论文里的投机者</b>（顺势），<b>但结果像论文里的套保者</b>（赚钱）：他们买入后价格继续同向，没有反转，说明他们不是在付流动性成本。</li>'
             '<li><b>论文的“两种溢价”框架在中国无法用这类数据复现</b>：既看不到大户付出的流动性溢价，也看不到长期净空头带来的套保溢价，两者方向都反了。'
             '一种解释（未检验）是：中国“前 20 名会员”里聚集的是信息更多、资金更大的机构和产业客户，而真正“急于成交、追涨杀跌”的是前 20 名以外的散户。</li>'
             '<li><b>效应在减弱</b>：2016 年以后，大户净买入不再预测收益，与本库其他复现中“2016 年后很多效应减弱”一致。</li>'
             '<li><b>局限</b>：期货公司名下混有各类客户；前 20 名成员每周变化；没有区分交易者类型。这些都会削弱信号，结论应视为“大户整体”的行为。</li></ul>')
    p.append('<h2>代码与数据</h2>')
    p.append(table(['文件', '作用'], [['krt.py', '构造周度大户持仓变量、Fama-MacBeth 回归、分组与双重排序'], ['build_report.py', '生成本报告'],
                                    ['raw_data/{交易所}/{品种}/holding_rank.parquet', '交易所前 20 名会员成交持仓排名（Tushare fut_holding）']]))
    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>套保与流动性溢价</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
