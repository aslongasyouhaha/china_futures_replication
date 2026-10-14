"""生成 Basis-Momentum 复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harq'))
from build_report import CSS, table  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')
NAMES = {'BM_nb': 'BM 近月因子', 'BM_spr': 'BM 价差因子', 'B_nb': '基差近月因子', 'B_H4_spr': '基差 High4 价差', 'B_L4_spr': '基差 Low4 价差',
         'AVG': '商品等权（AVG）', 'M_nb': '动量近月因子', 'B12_nb': '12 个月平均基差因子'}


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def f(x, k=2):
    return '' if x is None else f'{x:.{k}f}'


def ft(a, t, k=2):
    return '' if a is None else f'{a:.{k}f}（{t:.2f}）'


def build():
    t1 = rd('t1_paper')
    t1r = {v: rd(f't1r_{v}') for v in ('paper', 'paper500', 'liquid')}
    t2 = {v: rd(f't2_{v}') for v in ('paper', 'paper500', 'liquid')}
    sp = {v: rd(f'span_{v}') for v in ('paper', 'paper500', 'liquid')}
    fs, fc = rd('fstats_paper'), rd('fcorr_paper')
    t4a, t4b, t5, t8, ce = rd('t4a_paper'), rd('t4b_paper'), rd('t5_paper'), rd('t8_paper'), rd('curve_ext_paper')
    x = rd('panel_paper')

    def s1(var, leg, kind, d=t1):
        r = d.filter((pl.col('排序变量') == var) & (pl.col('组合') == leg) & (pl.col('收益') == kind)).row(0, named=True)
        return r

    def hl(v, var, kind, setting='High4−Low4，全样本'):
        return t1r[v].filter((pl.col('设定') == setting) & (pl.col('排序变量') == var) & (pl.col('收益') == kind)).row(0, named=True)

    def span(v, model, fac):
        return sp[v].filter((pl.col('基准模型') == model) & (pl.col('因子') == fac)).row(0, named=True)

    def reg(v, y, model):
        return t2[v].filter((pl.col('收益') == y) & (pl.col('模型') == model)).row(0, named=True)

    bmn = s1('Basis-momentum', 'High4−Low4', '近月')
    bms = s1('Basis-momentum', 'High4−Low4', '价差')
    sb = span('paper', 'Bakshi-Gao-Rossi', 'BM_nb')
    sl = span('liquid', 'Bakshi-Gao-Rossi', 'BM_nb')
    r4 = reg('paper', '近月', '(4)')
    corr_bm_m = fc.filter(pl.col('因子') == 'BM_nb')['M_nb'][0]
    nm = x['m'].n_unique()

    p = []
    p.append('<h1>Basis-Momentum 复现报告</h1><div class="meta">复现论文：Martijn Boons &amp; Melissa Porras Prado，《Basis-Momentum》，'
             '<i>Journal of Finance</i>，2019 · 中国商品期货，'
             f"{x['m'].min().strftime('%Y-%m')} 至 {x['m'].max().strftime('%Y-%m')}，{nm} 个月，{x['product'].n_unique()} 个品种 · 月末结算价</div>")
    p.append(f"<p class=\"lead\">按论文方法，basis-momentum 在中国商品期货上<b>能预测收益，但不是独立的因子</b>。"
             f"单变量排序的多空收益为年化 {bmn['年化%']:.1f}%（t = {bmn['t']:.2f}），论文为 18.4%（t = 6.73）；"
             f"但它和普通动量高度重叠：控制 Bakshi-Gao-Rossi 三因子（含动量）后 α 只剩 {sb['α 年化%']:.1f}%（t = {sb['t(α)']:.2f}），"
             f"控制品种和时间固定效应的混合回归也不显著（t = {r4['t_BM']:.2f}，论文 6.81）。论文的结论是 basis-momentum 吸收了动量，中国正好相反。"
             f"换成中国流动性最好的合约（主力与次主力）计算时，basis-momentum 更强，控制三因子后仍显著（α {sl['α 年化%']:.1f}%，t = {sl['t(α)']:.2f}），"
             f"价差收益尤其稳健。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{bmn['年化%']:.1f}%（t {bmn['t']:.2f}）</b><span>BM 多空近月收益（论文 18.4%，t 6.73）</span></div>"
             f"<div class=\"kpi\"><b>{sb['α 年化%']:.1f}%（t {sb['t(α)']:.2f}）</b><span>对 BGR 三因子的 α（论文 12.8%，t 5.20）</span></div>"
             f"<div class=\"kpi\"><b>{corr_bm_m:.2f}</b><span>BM 因子与动量因子的相关系数</span></div>"
             f"<div class=\"kpi\"><b>{sl['α 年化%']:.1f}%（t {sl['t(α)']:.2f}）</b><span>主力合约口径：对 BGR 三因子的 α</span></div></div>")

    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>Basis-momentum = 近月合约过去 12 个月的滚动收益 − 第二近月合约过去 12 个月的滚动收益。它衡量期限结构形状的变化：'
             '可以拆成平均曲率（近端比远端陡多少）和斜率变化两部分。论文用 1960–2014 年 21 个美国商品发现：'
             '按 basis-momentum 排序，最高 4 个减最低 4 个的近月收益年化 18.4%，价差收益（近月 − 第二近月）4.1%，夏普都约 0.9；'
             '它比基差（carry）和动量都强，联合回归中基差和动量不再显著；对已有的商品三因子模型有很大的 α；'
             '论文认为它补偿的是投机者和中介在波动大、流动性差时承担的风险，而不是库存或套保压力。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['合约', '第 n 近月 = 到期在 t+2 之后的合约中按到期顺序排第 n 个（沿用 Szymanowska et al.）', '交割月 ≥ 当前月 + 2 的合约按交割月排序，取第 1~4 个（持有期内不进入交割月，与三因子复现一致）'],
        ['收益', '同一合约月末到下月末；近月收益 R1、价差收益 R1 − R2', '同左，官方结算价'],
        ['特征', '基差 F2/F1 − 1；动量 = 近月 12 个月滚动收益；BM = 近月与第二近月 12 个月滚动收益之差', '同左'],
        ['排序', '21 个商品，High4 / Mid / Low4，等权', '同左（每月 10~70 个品种，中位数 15）；另做前后 20% 分组'],
        ['流动性', '美国合约流动性好，无额外要求', '第 1、2 个合约当天持仓 ≥ 100 手且有成交；品种池同三因子复现（主力 60 日平均成交 ≥ 1000 手），每月 ≥ 10 个品种'],
        ['检验', 'Table I 排序；II 混合回归；III spanning；IV 截面定价；V 双重排序；VIII 波动率；IA.XI 期限结构远端', '同左'],
        ['未做', '库存、套保压力（CFTC 持仓）、TED / VIX、中介资本、下行市场风险、其他资产类别', '缺数据或没有中国对应物']]))
    p.append('<p>关于流动性门槛：如果要求第 1、2 个合约持仓都 ≥ 500 手，2018 年以前的非主力月份大多不合格，样本只剩 125 个月（几乎全是 2018 年以后）。'
             '这一版本作为稳健性检验列在第七部分。</p>')

    p.append('<h2>一、单变量排序（对应论文 Table I）</h2>')
    rows = []
    for var in ('Basis-momentum', '基差', '动量'):
        for kind in ('近月', '价差'):
            legs = ['High4', 'Mid', 'Low4', 'High4−Low4'] if var == 'Basis-momentum' else ['High4', 'Low4', 'High4−Low4']
            for leg in legs:
                r = s1(var, leg, kind)
                rows.append([var, kind, leg, f(r['年化%']), f(r['t']), f(r['夏普'])])
    p.append(table(['排序变量', '收益', '组合', '年化%', 't', '夏普'], rows))
    p.append(table(['论文（1960–2014）', 'BM 近月', 'BM 价差', '基差近月', '动量近月', '基差价差', '动量价差'],
                   [['High4−Low4 年化%（t）', '18.38（6.73）', '4.08（6.43）', '−10.61（−3.88）', '15.02（4.61）', '−0.77（−1.13）', '0.53（0.82）']]))
    m_n = s1('动量', 'High4−Low4', '近月')
    m_s = s1('动量', 'High4−Low4', '价差')
    p.append(f"<ul><li>Basis-momentum 有效：近月多空 {bmn['年化%']:.1f}%（t {bmn['t']:.2f}），价差多空 {bms['年化%']:.1f}%（t {bms['t']:.2f}），方向与论文一致，力度约为论文的 70%。</li>"
             f"<li>与论文最大的不同：<b>动量比 basis-momentum 更强</b>（近月 {m_n['年化%']:.1f}%，t {m_n['t']:.2f}），而且动量也能预测价差收益（{m_s['年化%']:.1f}%，t {m_s['t']:.2f}）；"
             "论文中动量对价差收益没有预测力，这正是 basis-momentum 的独特之处。在中国，这个独特性不存在。</li></ul>")

    p.append('<h2>二、混合回归（对应论文 Table II）</h2>')
    p.append('<p>月收益（%）对上月末特征回归，按月聚类标准误。论文：BM 单独 10.45（7.45），加时间与品种固定效应 9.16（6.81），与基差、动量同时放入仍为 8.22（4.09），基差和动量不显著。</p>')
    rows = []
    for y in ('近月', '价差'):
        for mdl in ('(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(7a) 2007–2015', '(7b) 2016–2026'):
            r = reg('paper', y, mdl)
            rows.append([y, mdl, '是' if r['时间FE'] else '', '是' if r['品种FE'] else '', ft(r.get('λ_BM'), r.get('t_BM')),
                         ft(r.get('λ_B'), r.get('t_B')), ft(r.get('λ_M1'), r.get('t_M1')), f(r['R2']), r['观测数']])
    p.append(table(['收益', '模型', '时间FE', '品种FE', 'λ_BM（t）', 'λ_基差（t）', 'λ_动量（t）', 'R²', '观测数'], rows))
    p.append(f"<ul><li>不加品种固定效应时 BM 显著（t {reg('paper', '近月', '(2)')['t_BM']:.2f}），加了以后降到 {r4['t_BM']:.2f}：中国的 BM 效应主要来自<b>品种之间的长期差异</b>，"
             "而不是同一品种内的时间变化；论文则相反，以时间变化为主。</li>"
             "<li>三个变量同时放入时都不显著（近月），它们在中国高度重叠。</li></ul>")
    rows = []
    for y in ('近月', '价差'):
        for mdl, cols in (('C1 近月动量 vs 第二近月动量', ('M1', 'M2')), ('C2 曲率 vs 斜率变化', ('Curv', 'dSlope')),
                          ('C3 BM + 过去 12 个月平均基差', ('BM', 'B12')), ('C4 只用过去 12 个月平均基差', ('B12',))):
            r = reg('paper', y, mdl)
            rows.append([y, mdl] + [f"{c}：{ft(r.get('λ_' + c), r.get('t_' + c))}" for c in cols] + ([''] if len(cols) == 1 else []))
    p.append('<h3>Panel C：分解（均含时间、品种固定效应）</h3>')
    p.append(table(['收益', '设定', '变量 1', '变量 2'], rows))
    p.append('<p>论文：近月动量与第二近月动量系数大小相近、符号相反（9.06 与 −8.84），曲率贡献最大。中国两者方向相同但都不显著；曲率、斜率变化也都不显著。'
             '“过去 12 个月平均基差”（我们在三因子复现中发现的 carry 信息）同样不显著，三者在固定效应下都很难区分。</p>')

    p.append('<h2>三、因子与 spanning 回归（对应论文 Table III）</h2>')
    rows = [[NAMES[r['因子']], f(r['年化%']), f(r['t']), f(r['年化波动%']), f(r['偏度']), f(r['峰度']), f(r['AR(1)'])] for r in fs.iter_rows(named=True)]
    p.append(table(['因子', '年化%', 't', '年化波动%', '偏度', '峰度', 'AR(1)'], rows))
    rows = []
    for r in sp['paper'].iter_rows(named=True):
        rows.append([r['基准模型'], NAMES[r['因子']], f(r['α 年化%']), f(r['t(α)']), f(r['R2']), f(r['GRS'], 2), f(r['GRS p'], 3)])
    p.append(table(['基准模型', '因子', 'α 年化%', 't(α)', 'R²', 'GRS', 'GRS p'], rows + [
        ['论文：Szymanowska et al.', 'BM 近月 / 价差', '13.82 / 3.49', '5.56 / 6.10', '', '23.12', '2×10⁻¹⁰'],
        ['论文：Bakshi-Gao-Rossi', 'BM 近月 / 价差', '12.76 / 3.32', '5.20 / 5.39', '', '19.61', '5×10⁻⁹']]))
    sz = span('paper', 'Szymanowska et al. (2014)', 'BM_nb')
    p.append(f"<ul><li>只用基差因子（Szymanowska et al.）解释不了 BM：α {sz['α 年化%']:.1f}%（t {sz['t(α)']:.2f}）。</li>"
             f"<li>加入动量因子（Bakshi-Gao-Rossi）后，α 降到 {sb['α 年化%']:.1f}%（t {sb['t(α)']:.2f}），两个 BM 因子联合的 GRS p = {sb['GRS p']:.2f}，不再显著。"
             f"BM 与动量因子的相关系数为 {corr_bm_m:.2f}（论文 0.27）。</li>"
             "<li>再加入“12 个月平均基差”因子，α 进一步下降。</li></ul>")

    p.append('<h2>四、截面定价（对应论文 Table IV）</h2>')
    p.append('<p>Panel A：测试资产为 BM、基差、动量三种排序的 High4/Mid/Low4 与 4 个板块（论文为 7 个）的近月和价差收益，共 26 个；全样本 beta，Shanken t。'
             'Panel B：每月用过去一年日收益估计每个品种的 beta（≥ 125 个观测），Fama-MacBeth。γ 为年化风险价格（%）。</p>')
    for lab, d in (('Panel A：组合层面', t4a), ('Panel B：品种层面', t4b)):
        rows = []
        for r in d.iter_rows(named=True):
            rows.append([r['模型'], ft(r.get('γ_BM_nb'), r.get('t_BM_nb') or 0) if r.get('γ_BM_nb') is not None else '',
                         ft(r.get('γ_B_nb'), r.get('t_B_nb') or 0) if r.get('γ_B_nb') is not None else '',
                         ft(r.get('γ_M_nb'), r.get('t_M_nb') or 0) if r.get('γ_M_nb') is not None else '', f(r['R2']), f(r['MAPE%'])])
        p.append(f'<h3>{lab}</h3>' + table(['模型', 'γ BM 近月（t）', 'γ 基差近月（t）', 'γ 动量近月（t）', 'R²', 'MAPE%'], rows))
    p.append('<ul><li>组合层面：BM 因子有正的风险价格，加入两个基准模型后拟合度都有提高，与论文方向一致；但加入动量的模型中 BM 只是边际显著（t ≈ 2）。</li>'
             '<li>品种层面：BM 的风险价格在基准模型中都不显著；论文中约 15%、显著。动量和基差在中国品种层面同样难以定价（R² 只有 0.1~0.3）。</li></ul>')

    p.append('<h2>五、双重排序（对应论文 Table V）</h2>')
    p.append('<p>按 BM（中位数）与控制变量独立分组，看每个控制组内 BM 高减低的收益。论文中 BM 在所有控制组内都显著（近月 &gt; 12.7%）。库存与可储存性两项缺数据未做；'
             '“过去 12 个月平均基差”是论文 Table V 的最后一行，正好对应我们在三因子复现中的发现。</p>')
    rows = []
    for r in t5.iter_rows(named=True):
        rows.append([r['控制变量'], r['收益'], f"{f(r['单排序 H−L'])}（{f(r['t(H−L)'])}）",
                     '' if r.get('控制=H：BM H−L') is None else f"{f(r['控制=H：BM H−L'])}（{f(r['控制=H：t'])}）",
                     '' if r.get('控制=L：BM H−L') is None else f"{f(r['控制=L：BM H−L'])}（{f(r['控制=L：t'])}）"])
    p.append(table(['控制变量', '收益', '单排序高 − 低（t）', '控制变量高组内：BM 高 − 低（t）', '控制变量低组内：BM 高 − 低（t）'], rows))
    p.append('<p>控制基差、事后波动后，BM 效应仍然存在；<b>控制动量后，两个组内都不显著</b>（t 约 1.4）；控制过去 12 个月平均基差后也只是边际显著。</p>')

    p.append('<h2>六、商品市场波动与 BM 收益（对应论文 Table VIII）</h2>')
    rows = [[r['k（月）'], NAMES[r['因子']], '市场指数方差' if r['方差'] == 'var_mkt' else '品种平均方差', f(r['v1（%，每 1 个标准差）'], 3), f(r['t']), f(r['R2'], 3)]
            for r in t8.iter_rows(named=True)]
    p.append(table(['持有月数 k', '因子', '方差', '每 1 个标准差的收益变化（%）', 't', 'R²'], rows))
    p.append('<p>论文：滞后波动越高，BM 收益越高，并以此支持“流动性提供补偿”的解释。中国只有 BM 价差因子的下月收益随波动升高（t 约 3），近月因子和更长期限不显著，12 个月期限甚至为负。</p>')

    p.append('<h2>七、稳健性</h2>')
    rows = []
    for v, lab in (('paper', '论文规则，持仓 ≥ 100 手（主结果）'), ('paper500', '论文规则，持仓 ≥ 500 手'), ('liquid', '主力与次主力合约')):
        for setting in ('High4−Low4，全样本', '前后 20%，全样本'):
            a, b, c = hl(v, 'Basis-momentum', '近月', setting), hl(v, 'Basis-momentum', '价差', setting), hl(v, '动量', '近月', setting)
            rows.append([lab, setting.split('，')[0], a['月数'], ft(a['年化%'], a['t']), ft(b['年化%'], b['t']), ft(c['年化%'], c['t'])])
    for setting in ('High4−Low4，2007–2015', 'High4−Low4，2016–2026'):
        a, b, c = hl('paper', 'Basis-momentum', '近月', setting), hl('paper', 'Basis-momentum', '价差', setting), hl('paper', '动量', '近月', setting)
        rows.append(['论文规则，持仓 ≥ 100 手', setting.split('，')[1], a['月数'], ft(a['年化%'], a['t']), ft(b['年化%'], b['t']), ft(c['年化%'], c['t'])])
    p.append(table(['合约口径', '分组 / 时期', '月数', 'BM 近月 年化%（t）', 'BM 价差 年化%（t）', '动量近月 年化%（t）'], rows))
    rows = []
    for v, lab in (('paper', '论文规则 ≥ 100 手'), ('paper500', '论文规则 ≥ 500 手'), ('liquid', '主力与次主力')):
        a = span(v, 'Bakshi-Gao-Rossi', 'BM_nb')
        b = span(v, 'Bakshi-Gao-Rossi', 'BM_spr')
        c = span(v, 'BGR + 12 个月平均基差因子', 'BM_nb')
        r = reg(v, '近月', '(4)')
        rows.append([lab, ft(a['α 年化%'], a['t(α)']), ft(b['α 年化%'], b['t(α)']), f(a['GRS p'], 3), ft(c['α 年化%'], c['t(α)']), ft(r['λ_BM'], r['t_BM'])])
    p.append(table(['合约口径', '对 BGR：BM 近月 α（t）', '对 BGR：BM 价差 α（t）', 'GRS p', '再加 12 个月平均基差：BM 近月 α（t）', '混合回归（双固定效应）λ_BM（t）'], rows))
    p.append('<ul><li><b>主力与次主力口径下 BM 更强</b>：多空近月与价差收益都更高，对 BGR 三因子的 α 显著，价差因子尤其稳健（t 约 3.5）。'
             '可能的原因是：按论文规则选出的近月、第二近月在中国常常是不活跃的月份，结算价噪声大，削弱了期限结构信号；而主力与次主力才是中国期限结构真正被交易的地方。</li>'
             '<li>但无论哪种口径，加入品种固定效应的混合回归都不显著：BM 在中国主要区分“哪些品种长期更好”，而不是预测同一品种何时更好。</li></ul>')
    p.append('<h3>期限结构更远端（对应论文 Table IA.XI）</h3>')
    p.append(table(['按 BM 排序后持有', '月数', '年化%', 't', '夏普'], [[r['收益'], r['月数'], f(r['年化%']), f(r['t']), f(r['夏普'])] for r in ce.iter_rows(named=True)]))
    p.append('<p>与论文一致，BM 也能预测第二、第三近月的收益，越往远端越弱。</p>')

    p.append('<h2>结论</h2>')
    p.append('<ul><li><b>现象存在</b>：按 basis-momentum 排序的多空组合在中国显著赚钱，近月和价差收益方向都与论文一致。</li>'
             '<li><b>按论文方法，它不是新因子</b>：在中国它与普通动量高度重叠，控制动量后不再显著；论文的核心结论（BM 吸收动量和基差）不成立，方向反而相反。</li>'
             '<li><b>换成中国的主力合约口径，它更像独立因子</b>，尤其是价差收益。这说明期限结构信息在中国要从活跃合约上取。</li>'
             '<li><b>和之前发现的关系</b>：三因子复现中我们发现动量的预测力主要来自过去的基差。BM 正是“过去基差变化”的一种度量，但在中国它与动量、过去平均基差三者高度重叠，'
             '在品种固定效应下谁都不显著。也就是说，中国商品的这些信号主要是在区分品种，而不是在择时。</li></ul>')
    p.append('<h2>代码与数据</h2>')
    p.append(table(['文件', '作用'], [['data.py', '按两种口径选第 1~4 个合约，生成月末曲线与持有月内逐日收益'], ['bm.py', '全部检验（paper、paper500、liquid 三种口径）'],
                                    ['build_report.py', '生成本报告'], ['output/', '曲线、面板、因子与各表结果（parquet）']]))

    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Basis-Momentum 复现</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
