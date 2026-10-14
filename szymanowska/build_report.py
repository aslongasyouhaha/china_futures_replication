"""生成 Szymanowska et al. (2014) 复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harq'))
from build_report import CSS, table  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')
VN = {'BASIS': '基差', 'MOM': '动量', 'CV': '变异系数', 'INF': '通胀 beta', 'FX': '美元 beta', 'HP': '套保压力', 'OI': '持仓', 'LIQ': '流动性'}
# 论文 Table II / III：P4 − P1 年化%（t），Short Roll n = 1..4，Excess Holding n = 2..4
PAPER_SR = {'BASIS': ['−8.29（−2.40）', '−11.35（−4.33）', '−13.51（−5.22）', '−14.53（−4.45）'],
            'MOM': ['9.00（2.02）', '6.57（1.90）', '4.68（1.35）', '2.11（0.51）'],
            'CV': ['8.13（2.37）', '8.67（2.94）', '9.27（3.18）', '9.28（2.56）'],
            'INF': ['9.56（1.99）', '9.60（2.19）', '8.60（1.87）', '10.04（1.86）'],
            'FX': ['−1.86（−0.35）', '−1.41（−0.30）', '−0.91（−0.21）', '−1.81（−0.34）'],
            'HP': ['5.58（1.66）', '5.75（1.77）', '4.17（1.31）', '5.09（1.64）'],
            'OI': ['5.78（1.71）', '5.33（1.77）', '6.35（1.83）', '—'],
            'LIQ': ['−9.40（−2.22）', '−7.47（−2.05）', '−5.89（−1.85）', '−6.92（−1.82）']}
PAPER_EH = {'BASIS': ['0.61（1.72）', '1.44（2.91）', '1.84（3.00）'], 'MOM': ['−0.47（−1.12）', '−0.63（−1.01）', '−0.26（−0.40）'],
            'CV': ['−1.00（−3.09）', '−1.25（−2.69）', '−0.79（−1.16）'], 'INF': ['−0.60（−1.53）', '−1.15（−1.76）', '−1.46（−1.67）'],
            'FX': ['0.91（2.48）', '1.24（2.10）', '0.87（1.05）'], 'HP': ['−0.50（−1.30）', '−0.57（−0.89）', '−0.93（−1.34）'],
            'OI': ['−1.01（−2.49）', '−1.38（−2.05）', '0.69（0.52）'], 'LIQ': ['0.49（1.57）', '0.66（1.55）', '1.26（2.04）']}
# 论文 Table IV（平均 |α|，Wald p）：Short Roll n = 1..4；Excess Holding n = 2..4
PAPER_T4 = {'BASIS': (['0.60（0.993）', '0.63（0.988）', '2.27（0.390）', '1.24（0.745）'], ['0.48（0.102）', '0.82（0.031）', '0.71（0.007）']),
            'MOM': (['1.28（0.853）', '0.95（0.935）', '2.26（0.538）', '1.95（0.191）'], ['0.49（0.072）', '0.81（0.073）', '0.61（0.099）']),
            'CV': (['1.92（0.546）', '2.35（0.258）', '3.03（0.024）', '2.63（0.277）'], ['0.76（0.000）', '1.12（0.001）', '1.11（0.017）']),
            'INF': (['2.09（0.688）', '1.53（0.798）', '3.35（0.676）', '2.13（0.181）'], ['0.74（0.001）', '1.15（0.010）', '1.13（0.070）']),
            'HP': (['2.06（0.179）', '2.51（0.104）', '2.24（0.670）', '2.01（0.390）'], ['0.50（0.025）', '0.87（0.021）', '0.70（0.126）']),
            'LIQ': (['2.11（0.655）', '1.73（0.598）', '3.00（0.278）', '2.62（0.230）'], ['0.75（0.000）', '1.01（0.000）', '1.09（0.000）'])}
# 论文 Table V（Excess Holding，n = 2..4）：一个 Spreading 因子 / 两个
PAPER_T5 = {'BASIS': (['0.50（0.053）', '0.75（0.053）', '0.79（0.008）'], ['0.07（0.937）', '0.15（0.757）', '0.40（0.193）']),
            'MOM': (['0.49（0.059）', '0.75（0.048）', '0.74（0.073）'], ['0.09（0.910）', '0.18（0.814）', '0.25（0.536）']),
            'CV': (['0.75（0.000）', '1.08（0.004）', '1.24（0.003）'], ['0.21（0.434）', '0.24（0.356）', '0.38（0.490）']),
            'INF': (['0.73（0.004）', '1.09（0.019）', '1.35（0.013）'], ['0.08（0.964）', '0.14（0.900）', '0.38（0.579）']),
            'HP': (['0.52（0.015）', '0.81（0.037）', '0.85（0.036）'], ['0.18（0.601）', '0.20（0.531）', '0.25（0.477）']),
            'LIQ': (['0.73（0.000）', '1.01（0.000）', '1.28（0.000）'], ['0.30（0.097）', '0.75（0.000）', '0.89（0.000）'])}


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def f2(v, t):
    s = f'{v:.2f}（{t:.2f}）'
    return f'<b>{s}</b>' if abs(t) >= 1.96 else s


def fp(a, p):
    s = f'{a:.2f}（{p:.3f}）'
    return f'<b>{s}</b>' if p < 0.05 else s


def build():
    S, T, t1 = rd('sorts'), rd('tests'), rd('table1')
    P = rd('periods')

    def srt(var, a, n, tag='全样本'):
        q = S.filter((pl.col('样本') == tag) & (pl.col('变量') == var) & (pl.col('收益') == a) & (pl.col('n') == n))
        return q.row(0, named=True) if q.height else None

    def tst(var, a, n, model, tag='全样本'):
        q = T.filter((pl.col('样本') == tag) & (pl.col('排序') == var) & (pl.col('收益') == a) & (pl.col('n') == n) & (pl.col('模型') == model))
        return q.row(0, named=True) if q.height else None

    def hl(var, a, n, tag='全样本'):
        r = srt(var, a, n, tag)
        return f2(r['P4-P1'], r['P4-P1_t']) if r else '—'

    b1, b4 = srt('BASIS', 'SR', 1), srt('BASIS', 'SR', 4)
    e2, e4 = srt('BASIS', 'EH', 2), srt('BASIS', 'EH', 4)
    e2l = srt('BASIS', 'EH', 2, '持仓≥100')
    iv1 = tst('BASIS', 'SR', 1, 'IV 基差因子')
    ivm = tst('BASIS', 'SR', 1, '中国补充 基差因子+等权市场')
    v2 = [tst('BASIS', 'EH', n, 'V 高/低基差 Spreading 双因子') for n in (2, 3, 4)]
    n_med = P.filter(pl.col('SR1').is_not_null()).group_by('mi').len()['len'].median()
    y0, y1 = P['mi'].min() // 12, P['mi'].max() // 12

    p = []
    p.append('<h1>商品期货风险溢价的解剖：中国复现</h1><div class="meta">复现论文：Marta Szymanowska, Frans de Roon, Theo Nijman &amp; Rob van den Goorbergh，'
             '《An Anatomy of Commodity Futures Risk Premia》，<i>Journal of Finance</i> 69(1)，2014（使用 2013 年 SSRN 接受稿）· '
             f'中国商品期货 {P["product"].n_unique()} 个品种，双月频率，{y0}–{y1}</div>')
    p.append(f'<p class="lead">论文的核心结论在中国<b>成立，而且更强</b>：按基差分组，低基差（贴水）品种的现货溢价比高基差（升水）品种每年高 '
             f'{-b4["P4-P1"]:.1f}%~{-b1["P4-P1"]:.1f}%（t −5.7 到 −6.8；论文 −8% 到 −15%）；期限溢价方向与论文相同、幅度更大（{e2["P4-P1"]:.1f}%~{e4["P4-P1"]:.1f}%，论文 0.6%~1.8%），'
             f'但它主要来自不活跃的远月合约，只用持仓 ≥ 100 手的合约时降到 {e2l["P4-P1"]:.1f}%（t {e2l["P4-P1_t"]:.2f}）。'
             f'论文的定价结论只部分成立：单一基差因子<b>解释不了</b>中国按基差分组的现货溢价（平均 |α| {iv1["平均|α|"]:.1f}%，Wald p = {iv1["p"]:.3f}；论文 0.6%、p = 0.99）；'
             f'加入等权市场后 R² 从 {iv1["平均R2"]:.2f} 升到 {ivm["平均R2"]:.2f}，|α| 降到 {ivm["平均|α|"]:.1f}%，但仍被拒绝。高/低基差两个 Spreading 因子对期限溢价的解释力也弱于论文。</p>')
    p.append(f'<div class="kpis"><div class="kpi"><b>{b1["P4-P1"]:.1f}%（t {b1["P4-P1_t"]:.1f}）</b><span>基差 P4 − P1，Short Roll，n = 1（论文 −8.3%）</span></div>'
             f'<div class="kpi"><b>{e2["P4-P1"]:.1f}%（t {e2["P4-P1_t"]:.1f}）</b><span>基差 P4 − P1，Excess Holding，n = 2（论文 0.6%）</span></div>'
             f'<div class="kpi"><b>{iv1["p"]:.3f}</b><span>基差因子解释基差组合 Short Roll 的 Wald p（论文 0.993）</span></div>'
             f'<div class="kpi"><b>{n_med:.0f}</b><span>每期品种数中位数（论文 21 个）</span></div></div>')

    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>期货收益可以拆成两部分：<b>现货溢价</b>（持有最近月合约、到期前不断换月得到的收益，反映现货价格的风险补偿）和<b>期限溢价</b>（买远月、卖近月的收益，反映期限结构的风险补偿）。'
             '论文用四种策略把它们分开：Short Roll（连续滚动最近月 n 期，只含现货溢价）、Holding（买第 n 期合约持有到期，两者都有）、'
             'Excess Holding（Holding − Short Roll，只含期限溢价）、Spreading（每期买各期限的价差再滚动，也只含期限溢价）。</p>'
             '<p>主要发现：（1）按基差分组，现货溢价每年差 8%~15%，期限溢价只差 0.6%~1.8%，方向相反；（2）动量、波动率、通胀 beta、流动性分组也能产生溢价；'
             '（3）一个基差因子（Holding 收益）就能解释几乎所有分组的现货溢价，但解释不了期限溢价；（4）期限溢价需要另外两个因子——高基差和低基差品种的 Spreading 收益。'
             '合计三个因子刻画商品期货的风险溢价。</p>')

    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['频率与合约', '双月；最近月合约代替现货；第 n 期 = 第 n+1 近月；交割前一个月及交割月的价格剔除（在 T−1 月之前换月）',
         '双月（偶数月月末）；“第 n 期合约” = 交割月 ≥ 当前月 + 2 + 2n 的第一个合约（n = 0 为现货代理），保证持有到期时仍未进入交割前一月；官方结算价'],
        ['收益', '对数收益，等权；Short Roll、Holding、Excess Holding、Spreading，n = 1~4（2、4、6、8 个月）', '同左；年化 = 每期均值 × 6 / n'],
        ['样本', '21 个品种，1986-03 ~ 2010-12（CRB）', f'{P["product"].n_unique()} 个品种（不含金融期货；纤维板因报价单位变更剔除），{y0}–{y1}；每组至少 3 个品种（同论文）'],
        ['基差', '短期限对数基差 (f(1) − s)/期数', '[ln F(1) − ln F(0)] / 两合约相隔期数'],
        ['其他排序变量', '动量（t−12 ~ t−1 累计对数收益）；变异系数（36 个月 方差/均值）；通胀 beta（60 个月滚动，对 CPI 环比的变化）；美元 beta；CFTC 套保压力；总持仓；Amivest 流动性（2 个月）',
         '主力合约月收益计算动量、变异系数、beta；通胀用国家统计局 CPI 环比的变化（滞后一月）；美元用 Δln USDCNH（2012 年起，样本短）；'
         '<b>套保压力用前 20 名会员 (空 − 多)/(空 + 多)</b>——会员里混有投机者，不是 CFTC 的商业套保者；总持仓（手）；Amivest 用人民币成交额'],
        ['检验', 'NW t；时间序列回归 α 的 Wald 检验（NW）', '同左；n 期收益每期采样有重叠，NW 滞后 = n − 1 + 常规滞后；组合本身构成因子时协方差奇异，用广义逆、自由度取秩'],
        ['稳健性', '—', '（1）2005–2014 / 2015–2026 分段；（2）进场所用合约持仓 ≥ 100 手；（3）定价检验中加入等权市场因子']]))

    p.append('<h2>一、等权指数与板块（对应 Table I）</h2>')
    rows = []
    for s in ['全部', '农产品', '有色贵金属', '黑色', '能源化工']:
        r = [s]
        for a, ns in (('SR', (1, 2, 3, 4)), ('EH', (2, 3, 4))):
            for n in ns:
                q = t1.filter((pl.col('板块') == s) & (pl.col('收益') == a) & (pl.col('n') == n)).row(0, named=True)
                r.append(f2(q['年化%'], q['t']))
        rows.append(r)
    p.append(table(['板块', 'SR n=1', 'SR n=2', 'SR n=3', 'SR n=4', 'EH n=2', 'EH n=3', 'EH n=4'], rows))
    p.append('<p>年化%（NW t）。与论文一样，等权指数的现货溢价接近零；期限溢价为正但比现货溢价小一个数量级，等权层面不显著（论文显著）。</p>')

    p.append('<h2>二、按基差分组（对应 Table II）</h2>')
    rows = []
    for a, ns, lab in (('SR', (1, 2, 3, 4), 'Short Roll（现货溢价）'), ('EH', (2, 3, 4), 'Excess Holding（期限溢价）')):
        for n in ns:
            r = srt('BASIS', a, n)
            pap = PAPER_SR['BASIS'][n - 1] if a == 'SR' else PAPER_EH['BASIS'][n - 2]
            rows.append([lab, n] + [f2(r[q], r[q + '_t']) for q in ('P1', 'P2', 'P3', 'P4', 'P4-P1')] + [pap])
    p.append(table(['收益', 'n', 'P1 低基差', 'P2', 'P3', 'P4 高基差', 'P4 − P1', '论文 P4 − P1'], rows))
    p.append('<p>现货溢价随基差单调下降，期限溢价随基差单调上升，与论文完全一致；中国的幅度更大。</p>')
    rows = []
    for var, lab in (('BASIS_MEAN', 'B.1 品种平均基差'), ('BASIS_DEV', 'B.2 偏离平均的部分')):
        rows.append([lab, 'Short Roll'] + [hl(var, 'SR', n) for n in (1, 2, 3, 4)])
        rows.append([lab, 'Excess Holding', '—'] + [hl(var, 'EH', n) for n in (2, 3, 4)])
    rows.append(['论文 B.1', 'Short Roll', '−15.42（−4.37）', '−15.36（−4.40）', '−14.86（−4.17）', '−18.02（−3.91）'])
    rows.append(['论文 B.2', 'Short Roll', '−3.82（−1.12）', '−5.27（−2.01）', '−6.66（−2.74）', '−10.22（−3.27）'])
    p.append('<h3>基差的横截面部分与时变部分（Panel B）</h3>' + table(['分组', '收益', 'n=1', 'n=2', 'n=3', 'n=4'], rows))
    p.append('<p>论文中现货溢价主要来自“长期贴水的品种”（B.1），时变部分较弱。中国两部分都显著，时变部分（B.2）相对更重要——'
             '这与中国品种的期限结构经常在升水和贴水之间切换一致。注意 B.1 用了全样本平均基差，带前视（同论文）。</p>')

    p.append('<h2>三、其他排序变量（对应 Table III）</h2>')
    rows = []
    for v in ['MOM', 'CV', 'INF', 'FX', 'HP', 'OI', 'LIQ']:
        rows.append([VN[v], 'Short Roll'] + [hl(v, 'SR', n) for n in (1, 2, 3, 4)] + [PAPER_SR[v][0], PAPER_SR[v][3]])
        rows.append(['', 'Excess Holding', '—'] + [hl(v, 'EH', n) for n in (2, 3, 4)] + ['—', PAPER_EH[v][2]])
    p.append(table(['变量', '收益', 'n=1', 'n=2', 'n=3', 'n=4', '论文 n=1', '论文 n=4'], rows))
    p.append('<ul><li><b>动量</b>：现货溢价显著（n = 1 时 10%，论文 9%），且期限变长时衰减较慢；期限溢价不显著，同论文。</li>'
             '<li><b>变异系数</b>、<b>通胀 beta</b>：论文中都显著（8%~10%），中国不显著；通胀 beta 只在期限溢价上显著为负，方向与论文相同。'
             '通胀代理较粗（CPI 环比的变化，没有预期调整）。</li>'
             '<li><b>美元 beta</b>：不显著，同论文（且只有 2012 年起的 USDCNH）。</li>'
             '<li><b>套保压力方向相反</b>：论文中套保者净空越多、收益越高（+5.6%）；中国前 20 名会员净空越多，现货溢价越<b>低</b>（−5.5%~−6.7%，显著），期限溢价越高。'
             '前 20 名会员并非套保者，这个结果与本库 KRT 复现中“大会员仓位变动预测同向收益”一致——更像是知情交易，而不是套保需求的补偿。</li>'
             '<li><b>持仓</b>：总持仓大的品种现货溢价高 5%~6%（显著），论文同方向但不显著；期限溢价为负，与论文同向。</li>'
             '<li><b>流动性</b>：不显著，且符号与论文相反。</li></ul>')

    p.append('<h2>四、基差因子能否解释现货溢价（对应 Table IV）</h2>')
    rows = []
    for v in ['BASIS', 'MOM', 'CV', 'INF', 'HP', 'LIQ']:
        for n in (1, 2, 3, 4):
            a = tst(v, 'SR', n, 'IV 基差因子')
            e = tst(v, 'EH', n, 'IV 基差因子') if n > 1 else None
            m = tst(v, 'SR', n, '中国补充 基差因子+等权市场')
            rows.append([VN[v] if n == 1 else '', n, fp(a['平均|α|'], a['p']), PAPER_T4[v][0][n - 1],
                         fp(e['平均|α|'], e['p']) if e else '—', PAPER_T4[v][1][n - 2] if n > 1 else '—', fp(m['平均|α|'], m['p'])])
    p.append(table(['分组', 'n', 'Short Roll |α|（p）', '论文', 'Excess Holding |α|（p）', '论文', '中国补充：+等权市场 SR'], rows))
    p.append(f'<p>|α| 为四个组合年化 α 绝对值的平均（%），括号内为 α 联合为零的 Wald 检验 p 值，p &lt; 0.05 加粗。论文中基差因子把所有分组的现货溢价都解释掉了（p 普遍大于 0.1），'
             f'中国则不行：在基差自身分组上，n = 1 时 P4（高基差）组合的 α 达 {iv1["a4"]:.1f}%。原因是这个多空因子与高基差组合几乎不相关'
             f'（因子的波动主要来自低基差一侧），而中国商品整体同涨同跌、等权市场的波动很大。加入等权市场后，R² 从 {iv1["平均R2"]:.2f} 升到 {ivm["平均R2"]:.2f}，'
             f'α 大幅缩小，动量、变异系数、通胀、美元、套保压力分组在 n = 1 时不再被拒绝；但基差、持仓、流动性分组仍被拒绝，n ≥ 2 时多数分组也被拒绝。'
             '期限溢价方面，在基差、套保压力、通胀分组上，基差（Holding）因子同样解释不了（同论文）；其他分组本身的期限溢价就很小，谈不上需要解释。</p>')

    p.append('<h2>五、期限溢价需要两个 Spreading 因子吗（对应 Table V）</h2>')
    rows = []
    for v in ['BASIS', 'MOM', 'CV', 'INF', 'HP', 'LIQ', 'OI']:
        for n in (2, 3, 4):
            a = tst(v, 'EH', n, 'V 基差 Spreading 因子')
            b = tst(v, 'EH', n, 'V 高/低基差 Spreading 双因子')
            pv = PAPER_T5.get(v)
            rows.append([VN[v] if n == 2 else '', n, fp(a['平均|α|'], a['p']), pv[0][n - 2] if pv else '—', fp(b['平均|α|'], b['p']), pv[1][n - 2] if pv else '—',
                         f"{b['平均R2']:.2f}"])
    p.append(table(['分组', 'n', '一个因子 |α|（p）', '论文', '两个因子 |α|（p）', '论文', '两因子平均 R²'], rows))
    p.append(f'<p>和论文一样，两个因子在多数分组上明显好于一个（基差、动量、持仓分组的 |α| 减半以上），变异系数、通胀、流动性分组的期限溢价被解释；'
             f'但在基差自身分组上 n = 3、4 仍被拒绝（p = {v2[1]["p"]:.3f}、{v2[2]["p"]:.3f}），套保压力分组全部被拒绝，动量、持仓在 n = 4 被拒绝。论文中只有流动性分组不能被解释。</p>')

    p.append('<h2>六、其他变量的因子（对应 Table VI）</h2>')
    v = T.filter((pl.col('样本') == '全样本') & (pl.col('收益') == 'SR') & (pl.col('模型').str.starts_with('VI') | (pl.col('模型') == 'IV 基差因子')))
    g = v.group_by('模型').agg(pl.col('平均|α|').mean().alias('a'), (pl.col('p') < 0.05).sum().alias('rej'), pl.len().alias('k')).sort('a')
    rows = [[r['模型'].replace('VI ', '').replace('IV ', ''), f"{r['a']:.2f}", f"{r['rej']} / {r['k']}"] for r in g.iter_rows(named=True)]
    p.append(table(['单因子（Holding 收益，P3+P4 − P1−P2）', '8 种分组 × 4 个期限的平均 |α|%', 'Wald 拒绝（p &lt; 0.05）'], rows))
    p.append('<p>论文中基差因子在“赛马”里胜出。中国没有哪个单因子能解释全部分组：基差因子被拒绝的次数最少，但平均 |α| 并不最小——'
             '动量、通胀、持仓等因子的 α 更小，却更常被拒绝。这再次说明中国商品收益的横截面不是单一基差因子能概括的。</p>')

    p.append('<h2>七、分段与流动性稳健性</h2>')
    rows = []
    for v in ['BASIS', 'MOM', 'HP', 'OI']:
        for a, n in (('SR', 1), ('SR', 4), ('EH', 2), ('EH', 4)):
            rows.append([VN[v], f'{"Short Roll" if a == "SR" else "Excess Holding"} n={n}'] + [hl(v, a, n, tag) for tag in ('全样本', '2005–2014', '2015–2026', '持仓≥100')])
    p.append(table(['分组', '收益', '全样本', '2005–2014', '2015–2026', '合约持仓 ≥ 100 手'], rows))
    p.append('<p>基差的现货溢价在两段都显著，前一段更大。期限溢价在两段都显著，但只用持仓 ≥ 100 手的合约时明显变小——'
             '中国远月合约常常不活跃，结算价未必是可成交的价格，部分“期限溢价”可能是流动性溢价或价格噪声。'
             '动量在 2015 年后更稳定；套保压力（前 20 名净空）的效应主要在 2015 年后。</p>')

    p.append('<h2>与论文的比较</h2>')
    p.append('<ul><li><b>一致的</b>：现货溢价与期限溢价方向相反、量级相差一个数量级；基差分组同时产生显著的现货溢价和期限溢价；动量产生现货溢价但没有期限溢价；美元 beta 无效；'
             '基差（Holding）因子解释不了基差分组的期限溢价，高/低基差两个 Spreading 因子在多数分组上明显好于一个。</li>'
             '<li><b>不一致的</b>：基差因子在中国解释不了现货溢价（论文的核心定价结论）；变异系数、通胀 beta 分组没有现货溢价；'
             '套保压力方向相反（但代理变量不同）；期限溢价更大，且对远月流动性敏感。</li>'
             '<li><b>没有检验的</b>：论文网络附录中的 Holding、Spreading 分组结果（output/sorts.parquet 里已算出）、三因子的主成分分析、Table VII 的 Spreading 替代因子。</li></ul>')
    p.append('<h2>代码与数据</h2>')
    p.append(table(['文件', '作用'], [['szy.py', '双月合约面板、四种收益、排序变量、Table I~VI 及稳健性'], ['build_report.py', '生成本报告'],
                                    ['output/periods.parquet', '品种 × 双月：各期合约、价格、收益、排序变量'], ['output/sorts.parquet', '所有分组的 P1~P4、P4 − P1（含 Holding、Spreading）'],
                                    ['output/tests.parquet', '所有定价检验的 α、t、Wald p、R²'], ['output/table1.parquet', '板块等权指数']]))
    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>商品风险溢价解剖</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
