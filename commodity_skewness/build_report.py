"""生成偏度复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harq'))
from build_report import CSS, table  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')
VN = {'paper': '论文规则（近月，到期前一个月换月）', 'liquid': '主力合约'}


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def f(x, k=2):
    return '' if x is None or x != x else f'{x:.{k}f}'


def build():
    R = {v: {k: rd(f'{k}_{v}') for k in ('t1', 't2', 't3', 't4', 't5', 't6', 'panel')} for v in ('paper', 'liquid')}
    t3t = {v: rd(f't3t_{v}').row(0, named=True) for v in ('paper', 'liquid')}
    P = R['paper']
    ls = P['t4'].filter(pl.col('组合') == '多空 (Q1−Q5)/2').row(0, named=True)
    q1 = P['t4'].filter(pl.col('组合') == 'Q1').row(0, named=True)
    q5 = P['t4'].filter(pl.col('组合') == 'Q5').row(0, named=True)
    lsl = R['liquid']['t4'].filter(pl.col('组合') == '多空 (Q1−Q5)/2').row(0, named=True)
    fil = P['t5'].filter(pl.col('检验').str.contains('三因子回归的残差')).row(0, named=True)
    t3p = P['t3']
    x = P['panel']

    p = []
    p.append('<h1>商品期货偏度溢价复现报告</h1><div class="meta">复现论文：Adrian Fernandez-Perez, Bart Frijns, Ana-Maria Fuertes &amp; Joëlle Miffre，'
             '《The Skewness of Commodity Futures Returns》，<i>Journal of Banking &amp; Finance</i>，2018 · 中国商品期货，'
             f"{x['m'].min().strftime('%Y-%m')} 至 {x['m'].max().strftime('%Y-%m')}，{x['m'].n_unique()} 个月，{x['product'].n_unique()} 个品种</div>")
    p.append(f"<p class=\"lead\">中国商品期货<b>没有偏度溢价</b>。按论文方法做空正偏度、做多负偏度的品种，多空收益年化 {ls['年化%']:.2f}%（t = {ls['t(NW)']:.2f}），"
             f"论文为 8.01%（t = 3.83）；对基准模型的 α 为 {ls['α 年化%']:.2f}%（t = {ls['t(α)']:.2f}），论文为 6.58%（t = 3.58）。"
             f"原因是中国的偏度和动量方向相反：正偏度的品种恰恰是过去的赢家（Q5 过去 12 个月日均收益折年 {t3p.filter(pl.col('Q') == 'Q5')['MOM'][0]*100:.1f}%，Q1 为 {t3p.filter(pl.col('Q') == 'Q1')['MOM'][0]*100:.1f}%），"
             f"而论文中负偏度的才是赢家、贴水品种。把动量等因子的影响剔除后再算偏度，α 变为 {fil['α 年化%']:.2f}%（t = {fil['t(α)']:.2f}），说明偏度溢价被动量掩盖了，但它本身并不强。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{ls['年化%']:.2f}%（t {ls['t(NW)']:.2f}）</b><span>多空 (Q1−Q5)/2 年化（论文 8.01%，t 3.83）</span></div>"
             f"<div class=\"kpi\"><b>{ls['α 年化%']:.2f}%（t {ls['t(α)']:.2f}）</b><span>对 EW+TS+Mom 的 α（论文 6.58%，t 3.58）</span></div>"
             f"<div class=\"kpi\"><b>{q5['年化%']:.2f}%</b><span>正偏度组 Q5 年化（论文 −10.89%，最差）</span></div>"
             f"<div class=\"kpi\"><b>{lsl['年化%']:.2f}%（t {lsl['t(NW)']:.2f}）</b><span>主力合约口径多空年化</span></div></div>")

    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>按过去 12 个月日收益的偏度给商品排序，做多偏度最低（左偏）的五分之一、做空偏度最高（右偏、像彩票）的五分之一，'
             '美国 27 个商品 1987–2014 年年化 8.01%，夏普 0.78，控制期限结构、动量、套保压力后 α 仍有 6.58%。'
             '解释是投资者（包括做“选择性套保”的产业客户）偏好右偏的“彩票型”资产，使其被高估、未来收益偏低。截面检验中偏度因子的风险价格显著为正。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['合约', '近月合约，到期前一个月换到第二近月', '每月选交割月 ≥ 当前月 + 2 的第一个合约（持有月内不进入个人须退出的月份）；当天持仓 ≥ 100 手且有成交；另做主力合约口径'],
        ['偏度', '过去 12 个月日对数收益的 Pearson 偏度', '同左（至少 150 个交易日）'],
        ['组合', '五分组，多空 = (Q1 − Q5)/2（全额保证金口径），持有 1 个月', '同左；每月 10~60 个品种'],
        ['基准', 'EW + TS + Mom + HP（CFTC 套保压力）；增强模型另加流动性、持仓变化、通胀 beta、美元 beta、价值、波动率', 'EW + TS + Mom；增强模型加流动性（Amihud）、持仓变化、价值、波动率（变异系数）。HP 无中国对应数据；通胀、美元 beta 未做'],
        ['检验', 'Table III 组合特征；IV 表现与 α；V 稳健性；VI 截面 Fama-MacBeth', '同左']]))

    p.append('<h2>一、偏度的分布（对应论文 Table I）</h2>')
    p.append(f"<p>各品种偏度为正的月份平均占 {P['t1']['为正比例'].mean()*100:.0f}%（论文 43%），偏度随时间变化大、会变号，与论文一致。</p>")

    p.append('<h2>二、各组特征（对应论文 Table III）</h2>')
    rows = [[r['Q'], f(r['SK']), f(r['RY'] * 100, 3), f(r['MOM'] * 100)] for r in t3p.iter_rows(named=True)]
    p.append(table(['组合', '形成期偏度', '平均展期收益 ln(F1/F2)（%，日）', '过去 12 个月日均收益折年 %'],
                   rows + [['Q1 − Q5 的 t', f(t3t['paper']['SK']), f(t3t['paper']['RY']), f(t3t['paper']['MOM'])]]))
    p.append('<p><b>与论文最大的不同</b>：论文中左偏的 Q1 是贴水、过去表现好的品种，右偏的 Q5 是升水、过去表现差的品种，偏度与动量、期限结构同向。'
             '中国正好相反：右偏的 Q5 是过去的赢家，左偏的 Q1 是过去的输家（差异 t 值约 −6），展期收益在各组之间几乎没有差别。'
             '在中国，涨得多的品种往往是被少数几天暴涨拉起来的，所以右偏；这让“做空右偏”在中国等于“做空赢家”。</p>')

    p.append('<h2>三、组合表现与 α（对应论文 Table IV）</h2>')
    for v in ('paper', 'liquid'):
        rows = []
        for r in R[v]['t4'].iter_rows(named=True):
            rows.append([r['组合'], f(r['年化%']), f(r['t(NW)']), f(r['年化波动%']), f(r['夏普']), f(r['最大回撤%'], 1), f(r['α 年化%']), f(r['t(α)']),
                         f(r['β_TS']), f(r['β_MOM']), f(r['t_MOM'])])
        p.append(f'<h3>{VN[v]}</h3>' + table(['组合', '年化%', 't', '年化波动%', '夏普', '最大回撤%', 'α 年化%', 't(α)', 'β_TS', 'β_MOM', 't(β_MOM)'], rows))
    p.append(table(['论文（美国）', 'Q1', 'Q5', '多空', '多空 α'], [['年化%（t）', '5.12（1.42）', '−10.89（−3.35）', '8.01（3.83）', '6.58（3.58）']]))
    p.append(f"<p>论文规则下，偏度从低到高的五组收益没有单调关系，正偏度的 Q5 反而最好。多空组合对动量因子的暴露显著为负（β = {ls['β_MOM']:.2f}，t = {ls['t_MOM']:.2f}），"
             f"控制后 α 也只有 {ls['α 年化%']:.2f}%，不显著。主力合约口径下方向和论文一致但同样不显著。</p>")

    p.append('<h2>四、稳健性（对应论文 Table V）</h2>')
    for v in ('paper', 'liquid'):
        rows = [[r['检验'], f(r['α 年化%']), f(r['t(α)']), f(r.get('多空年化%'))] for r in R[v]['t5'].iter_rows(named=True)]
        p.append(f'<h3>{VN[v]}</h3>' + table(['检验', 'α 年化%（交易成本行为盈亏平衡成本 %）', 't(α)', '多空年化%'], rows))
    p.append('<ul><li><b>过滤后偏度</b>：用过去 12 个月日收益对 EW、TS、Mom 三个因子回归，取残差算偏度再排序，论文规则下 α 升到 4.7%（t = 2.5）。'
             '也就是说，去掉“偏度里的动量成分”后，偏度溢价是存在的。但它的原始多空收益只有约 0.8%，α 主要来自对动量的负暴露被扣回。主力合约口径下不显著。</li>'
             '<li>只控制日历月（季节性）或剔除流动性最差的 20% 后，都不显著。</li>'
             '<li>换增强模型（加流动性、持仓变化、价值、波动率）不改变结论。</li>'
             '<li>分时期：2016 年以后偏度溢价方向变为正（论文规则 α 4.3%，t = 1.85），2016 年以前为负。</li>'
             '<li>盈亏平衡成本：论文为 0.93%；中国的多空收益本身为负或很小，谈不上盈亏平衡。</li></ul>')

    p.append('<h2>五、截面定价（对应论文 Table VI）</h2>')
    rows = []
    for v in ('paper', 'liquid'):
        for r in R[v]['t6'].iter_rows(named=True):
            rows.append([VN[v], r['模型'], f(r['平均 R2%']), f(r.get('λ_TS')), f(r.get('t_TS')), f(r.get('λ_MOM')), f(r.get('t_MOM')), f(r.get('λ_SKEW')), f(r.get('t_SKEW'))])
    p.append(table(['口径', '模型', '平均 R²%', 'λ_TS 年化%', 't', 'λ_Mom 年化%', 't', 'λ_偏度 年化%', 't'], rows + [['论文', '加偏度后', '31.77 → 35.26', '', '', '', '', '约 5.04', '显著']]))
    p.append('<p>按论文的做法：每个月用当月日收益估计各品种对因子的 beta，再对当月收益做截面回归。中国的期限结构、动量因子都有显著的正风险价格；偏度因子的风险价格不显著。</p>')

    lc = rd('lsz_check')
    p.append('<h2>六、与 Li, Sun &amp; Zhou (2021) 的对照</h2>')
    p.append('<p>Li, Sun &amp; Zhou（<i>Journal of Management Science and Engineering</i>，2021）用 2003-01 ~ 2018-06 的中国商品期货，'
             '以最近月合约（持有到到期再换月）、全部 47 个品种、四分组多空，得到偏度策略（做多左偏、做空右偏）年化 −9.78%（t = −2.55），'
             '即右偏品种跑赢左偏品种，方向与本复现一致；他们所说的“偏度最重要”是指在多因子组合预测中剔除偏度后收益下降最多，并不是美国式的偏度溢价成立。'
             '下面用同一个偏度信号、同样的四分组满额多空（多头 − 空头），逐项替换合约构造和品种筛选。</p>')
    rows = [[r['设定'], r['时期'], r['月数'], f"{r['每月品种中位数']:.0f}", f(r['多空年化%']), f(r['t']), f(r['左偏组年化%']), f(r['右偏组年化%'])] for r in lc.iter_rows(named=True)]
    p.append(table(['设定', '时期', '月数', '每月品种', '多空年化%（左偏 − 右偏）', 't', '左偏组年化%', '右偏组年化%'], rows))
    p.append('<ul><li><b>差异来自样本期，不是合约构造</b>：在 Li, Sun &amp; Zhou 的样本期内，三种口径都显著为负（t 约 −2.1 ~ −2.9），用他们的口径得到 −9.2%（t = −2.85），与原文几乎相同。</li>'
             '<li><b>2018 年以后方向翻转</b>：三种口径都转为正（+3% ~ +6%），与美国的方向一致了，因此全样本不显著。这与第四部分“2016 年后偏度溢价转正”、Fan &amp; Zhang 复现中“样本外偏度转正”是同一件事。</li>'
             '<li>“最近月 + 持仓门槛”（D）偏弱，是因为中国最近月合约多不活跃，加门槛后每月只剩约 14 个品种。</li>'
             '<li>翻转的原因本研究没有检验；可能与商品期权上市、机构参与度提高、投资者结构变化有关，仅为推测。</li></ul>')
    fs = rd('flip_summary')
    fy = rd('flip_yearly')
    p.append('<h2>七、为什么 2018 年前后翻转</h2>')
    p.append('<p>同样的四分组满额多空（左偏 − 右偏，负值 = 右偏跑赢）。检验 1：偏度里是否混进了动量——(a) 偏度与过去 12 个月收益的截面秩相关；'
             '(b) 先按动量分成两组，在组内按偏度做多空；(c) 每月把偏度对动量做截面回归，用残差排序；(d) Fama-MacBeth：下月收益同时对偏度、动量（截面标准化）回归，'
             '系数为“偏度高 1 个标准差，下月收益高多少”（年化 %）。检验 2：逐年看。</p>')
    rows = []
    for r in fs.iter_rows(named=True):
        rows.append([r['口径'], r['时期'], f(r['偏度-动量秩相关']), f"{f(r['原始偏度多空%'])}（{f(r['t_raw'])}）", f"{f(r['动量组内偏度多空%'])}（{f(r['t_within'])}）",
                     f"{f(r['残差偏度多空%'])}（{f(r['t_resid'])}）", f"{f(r['FM 偏度（单独）'])}（{f(r['t_fm_sk_alone'])}）", f"{f(r['FM 偏度（控制动量）'])}（{f(r['t_fm_sk'])}）",
                     f"{f(r['动量多空%（高−低）'])}（{f(r['t_mom'])}）"])
    p.append(table(['口径', '时期', '偏度-动量秩相关', '原始偏度多空%（t）', '动量组内（t）', '残差偏度（t）', 'FM 偏度单独（t）', 'FM 偏度控制动量（t）', '动量多空 高−低（t）'], rows))
    rows = [[r['年'], f(r['原始'], 1), f(r['残差'], 1), f(r['动量'], 1), f(r['秩相关'])] for r in fy.filter(pl.col('口径').str.starts_with('C')).iter_rows(named=True)]
    p.append('<h3>逐年（持仓 ≥ 100 手口径，年化 %）</h3>' + table(['年', '原始偏度多空', '残差偏度多空', '动量多空', '偏度-动量秩相关'], rows))
    p.append('<ul><li><b>2018 年以前的“反向偏度效应”就是动量</b>：偏度与过去收益正相关（秩相关约 0.2~0.3），而这段时间动量很强。'
             '在动量组内排序后效应减半且不显著，用残差排序后接近 0，Fama-MacBeth 控制动量后偏度系数也变成 0（t ≈ 0.1）。也就是说，那时并不存在独立的偏度效应。</li>'
             '<li><b>2018 年以后出现了独立的、美国方向的偏度溢价</b>：原始排序看不出来（+3% ~ +4%，不显著），因为右偏的品种仍然偏向是赢家、动量依然很强；'
             '但控制动量后，右偏品种显著跑输——无门槛口径下残差排序 +11.3%（t = 2.27），Fama-MacBeth 系数 −4.9%（t = −3.14）；持仓门槛口径方向相同、显著性较弱（t ≈ −1.8）。</li>'
             '<li><b>逐年看并不是干净的一次性翻转</b>：年度数字噪声很大，残差偏度在 2016–2019、2021、2025 年为正，2020、2022、2024 年为负。2018 年后的结论应视为初步证据。</li>'
             '<li>为什么 2018 年后才出现独立的偏度溢价，本研究没有检验；可能与商品期权上市（2017 年起，2019 年后大量扩容）、机构参与增加有关，仅为推测。</li></ul>')
    p.append('<h2>结论</h2>')
    p.append('<ul><li><b>论文结论在中国全样本不成立</b>：做空右偏、做多左偏的品种在 2006–2026 年整体不赚钱，偏度因子在截面上也没有定价能力。'
             '但分时期看，2018 年以前是显著的“反向”（右偏跑赢，Li, Sun &amp; Zhou 2021 同样发现），这其实是动量的影子；2018 年以后，控制动量后出现了与美国同向的独立偏度溢价。</li>'
             '<li><b>原因是偏度在中国与动量反向</b>：中国的赢家品种往往由少数几天暴涨拉起（右偏），所以偏度信号变成了“反动量”信号；剔除动量成分后才隐约看到偏度溢价。'
             'Fan &amp; Zhang (2020, JFM) 也发现中国的偏度溢价被保证金和限仓“扭曲”。</li>'
             '<li><b>与我们之前的发现一致</b>：中国商品的截面收益主要由期限结构（carry）和动量驱动，偏度并不是独立于它们的新来源。</li></ul>')
    p.append('<h2>代码与数据</h2>')
    p.append(table(['文件', '作用'], [['skew.py', '全部检验；python skew.py paper|liquid'], ['lsz_check.py', '与 Li, Sun & Zhou (2021) 的口径对照'], ['skew_flip.py', '2018 年前后翻转的原因检验'], ['build_report.py', '生成本报告'],
                                    ['../basis_momentum/data.py', '按合约规则选合约，生成月末曲线和持有月内逐日收益（本研究共用）']]))
    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>商品偏度复现</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
