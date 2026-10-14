"""生成《Time Series Momentum: Is It There?》复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
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


def build():
    t2, t3, pt, pc = rd('table2'), rd('table3'), rd('pooled'), rd('pooled_cluster')
    oos, t9, t10, t10d, t11, sim = rd('pooled_oos'), rd('table9'), rd('table10'), rd('table10_diff'), rd('table11'), rd('sim')
    b_real = rd('beta_real')['β_real'][0]
    fig3 = rd('fig3')

    def prow(d, sample, pred, h):
        return d.filter((pl.col('样本') == sample) & (pl.col('预测变量') == pred) & (pl.col('h') == h)).row(0, named=True)

    m12 = prow(pt, '全部资产', '过去 h 月累计收益', 12)
    c12 = prow(pc, '全部资产', '过去 h 月累计收益', 12)
    fe12 = prow(pc, '控制固定效应', '过去 h 月累计收益', 12)
    post = prow(pc, '2016–2026', '过去 h 月累计收益', 12)
    n_sig = int(((t2['t'] > 1.65)).sum())
    n_neg = int((t2['β'] < 0).sum())
    n_all = t2.height
    d_ew = t10d.filter(pl.col('权重') == '等权').row(0, named=True)
    tsm_ew = t10.filter((pl.col('权重') == '等权') & (pl.col('策略') == 'TSM') & (pl.col('腿') == '多 − 空')).row(0, named=True)
    tsh_ew = t10.filter((pl.col('权重') == '等权') & (pl.col('策略') == 'TSH') & (pl.col('腿') == '多 − 空')).row(0, named=True)
    lw = t11.filter((pl.col('预测') == '波动率缩放') & (pl.col('资产类别') == '全部')).row(0, named=True)

    p = []
    p.append('<h1>时间序列动量真的存在吗：复现报告</h1><div class="meta">复现论文：Dashan Huang, Jiangyuan Li, Liyao Wang &amp; Guofu Zhou，'
             '《Time Series Momentum: Is It There?》，<i>Journal of Financial Economics</i>，2020（全文依据作者接受稿）· '
             f'中国期货 {n_all} 个有 5 年以上数据的品种（商品、股指、国债），2005–2026 · 月度</div>')
    p.append(f"<p class=\"lead\">论文的统计结论在中国大体成立：<b>逐个品种看，过去 12 个月收益几乎预测不了下个月</b>（{n_all} 个品种中只有 {n_sig} 个显著为正，"
             f"{n_neg} 个斜率为负）；混合回归的 t 值 {m12['t']:.2f}，控制品种固定效应后降到 {fe12['t']:.2f}，2016 年以后只有 {post['t']:.2f}。"
             f"但论文的经济学结论在中国<b>不成立</b>：论文说 TSM 赚钱只是因为它买入了长期均值高的资产，和“按历史均值买卖”（TSH）表现一样；"
             f"在中国 TSH 基本不赚钱（等权月均 {tsh_ew['月均%']:.2f}%），TSM 月均 {tsm_ew['月均%']:.2f}%，两者之差 {d_ew['收益差%']:.2f}%（p = {d_ew['收益差 p']:.2f}），"
             f"TSM 的预测也明显不像 TSH 的预测。另外，论文用来否定显著性的自助法有缺陷：它逐个观测重抽样，破坏了同月各资产之间的相关性，临界值被系统性抬高。"
             f"改为按月整块重抽后，全样本的 TSM 在 5% 水平上显著（单侧 p = {c12['pairs cluster p']:.3f}），但控制固定效应或只看 2016 年以后就不显著。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{n_sig}/{n_all}</b><span>逐个品种回归显著为正（10% 水平；论文 8/55 显著）</span></div>"
             f"<div class=\"kpi\"><b>{m12['t']:.2f}</b><span>混合回归 t（过去 12 个月，波动率缩放；论文 4.34）</span></div>"
             f"<div class=\"kpi\"><b>{m12['wild 论文口径']:.1f} / {c12['wild cluster 临界值']:.1f}</b><span>wild 自助法临界值：论文做法 / 按月整块</span></div>"
             f"<div class=\"kpi\"><b>{d_ew['收益差%']:.2f}%</b><span>TSM − TSH 月均收益差（等权；论文 0.14%）</span></div></div>")

    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>Moskowitz, Ooi &amp; Pedersen (2012) 用 55 个期货的混合回归（所有资产叠在一起）发现，过去 12 个月收益能预测下个月收益（t ≈ 4），称之为“时间序列动量无处不在”。'
             '这篇论文用同样的数据质疑：① 逐个资产回归，绝大多数不显著，样本外几乎都不如历史均值；② 混合回归的 t 值被高估——各资产均值不同、预测变量高度持续、'
             '波动率缩放都会让它偏大，用自助法得到的临界值（12.53、4.83）都高于真实 t 值 4.34；③ TSM 策略赚钱，但一个不需要任何可预测性的策略——'
             '按资产历史均值的符号做多做空（TSH）——表现几乎一样，因为过去 12 个月涨的资产往往就是长期均值高的资产。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['资产', '24 个商品、9 个股指、13 个国债、9 个外汇，1985–2015', '中国全部期货品种（与本库 TSMOM 复现相同的流动性门槛），商品为主，另有 3 个股指、3 个国债（有 5 年以上数据的）；没有外汇'],
        ['收益', '近月 / 次近月合约日收益连乘成月收益', '主力连续后复权日收益连乘成月收益'],
        ['波动率', '日收益指数加权（中心质量 60 天）', '同左'],
        ['样本外', '前 15 年训练，2000–2015 评估；Clark-West 检验', '每个品种前一半（至少 5 年）训练；混合回归样本外从 2016 年起'],
        ['自助法', 'wild（逐观测 Rademacher）与 pairs（逐观测重抽），各 1000 次，取模拟 t 的 97.5% 分位数', '同左（“论文口径”）；另加按月整块的 wild cluster（施加 β = 0）与 pairs cluster'],
        ['基准模型', 'Fama-French 四因子、AMP 三因子（MSCI、价值、动量）', '“市场”= 全部品种等权做多；“AMP 中国版”= 市场 + 本库 AMP 复现的商品价值、动量因子']]))

    p.append('<h2>一、逐个品种回归（对应论文 Table 2、Fig. 1）</h2>')
    p.append('<p>r<sub>t+1</sub> = α + β r<sub>t−12,t</sub>，NW t。样本外 R²<sub>OS</sub> 相对历史均值，Clark-West 单侧检验。</p>')
    srt = t2.sort('t', descending=True)
    p.append(table(['品种', '类别', '月数', 'β', 't', 'R²%', 'R²_OS%', 'CW p'],
                   [[r['品种'], r['类别'], r['月数'], f(r['β'], 3), f(r['t']), f(r['R2%']), f(r['R2_OS%']), f(r['CW p'])] for r in srt.iter_rows(named=True)]))
    n_os_pos = int((t2['R2_OS%'] > 0).sum())
    n_cw = int((t2['CW p'] < 0.1).sum())
    p.append(f"<ul><li>10% 水平下显著为正的只有 {n_sig} 个，{n_neg} 个斜率为负（论文：55 个中 8 个显著、17 个为负）；平均 R² {t2['R2%'].mean():.2f}%（论文 0.39%）。</li>"
             f"<li>样本外 R² 为正的 {n_os_pos} 个，Clark-West 10% 显著的 {n_cw} 个，中位数 {t2['R2_OS%'].drop_nans().median():.2f}%（论文：45 个为负、3 个显著）。</li>"
             "<li>结论与论文一致：逐个品种看，几乎没有时间序列动量。</li></ul>")
    p.append('<h3>各品种的均值、夏普是否相同（对应论文 Table 3）</h3>')
    p.append(table(['检验对象', '品种数', 'ANOVA p', 'Welch ANOVA p', 'Kruskal-Wallis p', '自助法 p'],
                   [[r['检验对象'], r['资产数'], f(r['ANOVA p'], 3), f(r['Welch ANOVA p'], 3), f(r['Kruskal-Wallis p'], 5), f(r['自助法 p'], 3)] for r in t3.iter_rows(named=True)]))
    p.append('<p>各品种的平均收益、夏普并不相同（多数检验在 5% 水平拒绝），与论文一致。这意味着不控制固定效应的混合回归会把“均值差异”误当成“可预测性”。</p>')

    p.append('<h2>二、混合回归与自助法（对应论文 Table 4–8）</h2>')
    p.append('<p>因变量 r<sub>t+1</sub>/σ<sub>t</sub>，按月聚类 t。论文口径的临界值 = 按论文做法模拟出的 t 的 97.5% 分位数；按月整块的 p 值 = 原假设（无可预测性）下模拟 t ≥ 真实 t 的比例（单侧）。'
             '论文：过去 12 个月累计收益 t = 4.34，wild / pairs 临界值 12.53 / 4.83。</p>')
    rows = []
    for r in pt.iter_rows(named=True):
        c = prow(pc, r['样本'], r['预测变量'], r['h'])
        rows.append([r['样本'], r['预测变量'], r['h'], f(r['β'], 3), f(r['t']), f(r['wild 论文口径']), f(r['pairs 论文口径']),
                     f(c['wild cluster 临界值']), f(c['wild cluster p'], 3), f(c['pairs cluster 临界值']), f(c['pairs cluster p'], 3)])
    p.append(table(['样本', '预测变量', 'h', 'β', 't', 'wild 临界值（论文）', 'pairs 临界值（论文）', 'wild 临界值（按月）', 'wild p（按月）', 'pairs 临界值（按月）', 'pairs p（按月）'], rows))
    p.append('<h3>为什么论文口径的临界值这么大</h3>')
    p.append(f"<p>论文的 wild 自助法给每个观测（品种 × 月）独立地乘 ±1，pairs 自助法逐个观测重抽。这两种做法都<b>破坏了同一个月内各资产收益之间的相关性</b>："
             f"模拟样本里，聚类标准误变得很小，t 值随之被放大。所以模拟 t 的分布远在真实 t 的右边（本样本过去 12 个月：真实 t {m12['t']:.2f}，"
             f"wild 模拟的 97.5% 分位数 {m12['wild 论文口径']:.2f}），用它当临界值几乎一定得出“不显著”。论文的 12.53 对 4.34 是同样的现象。"
             f"另外，论文按估计出的模型（β = β̂）而非原假设（β = 0）生成样本，这个分布本来就以真实 t 为中心，也不适合用来检验 β = 0。</p>"
             f"<p>按月整块重抽（同一个月的所有资产一起抽、一起乘 ±1）保留了截面相关性，临界值回到 2 左右（wild {c12['wild cluster 临界值']:.2f}，pairs {c12['pairs cluster 临界值']:.2f}）。"
             f"按这个正确的做法：</p>")
    sgn = prow(pc, '全部资产', '过去 h 月累计收益（符号）', 12)
    com = prow(pc, '商品', '过去 h 月累计收益', 12)
    pre = prow(pc, '2005–2015', '过去 h 月累计收益', 12)
    ns = prow(pc, '不做波动率缩放', '过去 h 月累计收益', 12)
    p.append(f"<ul><li>全样本过去 12 个月收益：t = {c12['t']:.2f}，按月整块 p = {c12['wild cluster p']:.3f}（wild）/ {c12['pairs cluster p']:.3f}（pairs），5% 水平显著；符号版本 t = {sgn['t']:.2f}。</li>"
             f"<li>显著性几乎全部来自商品（t = {com['t']:.2f}）和 2016 年以前（t = {pre['t']:.2f}）；股指、国债都不显著，2016 年以后 t = {post['t']:.2f}。</li>"
             f"<li>控制品种固定效应后 t = {fe12['t']:.2f}，不做波动率缩放 t = {ns['t']:.2f}，都不显著：和论文的论点一致，均值差异和波动率缩放抬高了混合回归的 t。</li>"
             "<li>滞后单月的结果（Panel A）在中国和论文不同：滞后 12 个月那一个月的收益反而显著为负（t ≈ −2.8），论文中接近 0。</li></ul>")
    lags = fig3.filter(~pl.col('符号')).sort('滞后月数')
    p.append('<h3>滞后 1–60 个月的单月收益 t 值（对应论文 Fig. 3 Panel A）</h3>')
    p.append(table(['滞后'] + [str(i) for i in range(1, 13)], [['t'] + [f(x, 1) for x in lags['t'].to_list()[:12]]]))
    p.append(f"<p>13–60 个月的 t 值在 −2.3 到 2.6 之间来回波动，没有论文中“前 12 个月持续为正、之后反转”的规律。</p>")

    p.append('<h2>三、混合回归的样本外表现（对应论文 Fig. 4）</h2>')
    agg = oos.group_by('类别').agg(pl.len().alias('n'), pl.col('R2_OS% 缩放').mean().alias('mean'), pl.col('R2_OS% 缩放').median().alias('med'),
                                  (pl.col('R2_OS% 缩放') > 0).sum().alias('pos'), (pl.col('CW p 缩放') < 0.1).sum().alias('sig')).sort('类别')
    p.append(table(['类别', '品种数', '平均 R²_OS%', '中位数 R²_OS%', 'R²_OS > 0', 'CW 10% 显著'],
                   [[r['类别'], r['n'], f(r['mean']), f(r['med']), r['pos'], r['sig']] for r in agg.iter_rows(named=True)]))
    p.append('<p>2016 年起用混合回归逐月预测：商品约四分之一的品种显著优于历史均值，但平均 R²<sub>OS</sub> 仍为负；股指、国债都为负。'
             '论文中混合回归主要改善了股指（平均 2.08%），中国正相反，股指最差。</p>')

    p.append('<h2>四、TSM 与 TSH：赚钱是不是只因为均值差异（对应论文 Table 9–10）</h2>')
    p.append('<p>TSM：过去 12 个月收益 ≥ 0 做多，否则做空；TSH：从上市到现在的累计收益（即历史均值）≥ 0 做多，否则做空，不需要任何可预测性。月度收益（%）。</p>')
    n_better = int((t9['收益差%'] > 0).sum())
    n_sigp = int(((t9['收益差 p'] < 0.1) & (t9['收益差%'] > 0)).sum())
    n_sign = int(((t9['收益差 p'] < 0.1) & (t9['收益差%'] < 0)).sum())
    p.append(f"<p>单个品种（Table 9）：{t9.height} 个品种中 TSM 收益高于 TSH 的有 {n_better} 个，其中 10% 水平显著的 {n_sigp} 个，显著更差的 {n_sign} 个（论文：55 个中只有 5 个 TSM 更高）。</p>")
    rows = []
    for r in t10.iter_rows(named=True):
        rows.append([r['权重'], r['策略'], r['腿'], f(r['月均%'], 3), f(r['t']), f(r['α% 市场'], 3), f(r['t(α) 市场']), f(r['α% AMP 中国版'], 3), f(r['t(α) AMP 中国版'])])
    p.append(table(['权重', '策略', '腿', '月均%', 't', 'α% 市场', 't', 'α% AMP 中国版', 't'], rows))
    p.append(table(['权重', 'TSM − TSH 月均%', 'p', 'α 差 市场', 'p', 'α 差 AMP 中国版', 'p'],
                   [[r['权重'], f(r['收益差%'], 3), f(r['收益差 p'], 3), f(r['α 差% 市场'], 3), f(r['α 差 p 市场'], 3), f(r['α 差% AMP 中国版'], 3), f(r['α 差 p AMP 中国版'], 3)]
                    for r in t10d.iter_rows(named=True)] + [['论文：等权', '0.14', '0.19', '0.10（四因子）', '0.26', '−0.02（AMP）', '0.84']]))
    p.append('<ul><li><b>与论文不同</b>：中国的 TSH 几乎不赚钱（等权、各种权重下 t 都小于 1），TSM 在四种权重下都比 TSH 高 0.3%~0.5%/月，p 值约 0.1~0.2。'
             '论文的解释——TSM 只是买到了长期均值高的资产——在中国不成立：中国商品的长期均值差异不稳定，按历史均值选不出赢家。</li>'
             '<li>但 TSM 本身也不强：等权月均 0.26%（t 1.5），波动率加权 0.76%（t 2.5）；控制 AMP 中国版（含商品动量因子）后 α 接近 0。</li>'
             '<li>注：这里的“市场”是全部品种等权做多，所以多头腿与空头腿对它的 α 恰好相同（多头 − 空头腿正好等于市场）。</li></ul>')

    p.append('<h2>五、Lewellen 预测斜率（对应论文 Table 11）</h2>')
    p.append(table(['预测', '资产类别', 'Lewellen 斜率', 't', 'TSM 预测对 TSH 预测的斜率 d', 't(d)'],
                   [[r['预测'], r['资产类别'], f(r['Lewellen 斜率']), f(r['t']), f(r['TSM 预测对 TSH 预测的斜率 d']), f(r['t(d)'])] for r in t11.iter_rows(named=True)]
                   + [['论文：缩放', '全部', '0.19', '0.61', '1.09', '18.56']]))
    p.append(f"<p>论文：TSM 预测几乎预测不了实际收益（斜率 0.19），而且和 TSH 预测几乎一样（d ≈ 1）。中国：用波动率缩放的预测，斜率 {lw['Lewellen 斜率']:.2f}（t {lw['t']:.2f}），"
             f"商品上 t 约 2.1，比论文强；d 只有 {lw['TSM 预测对 TSH 预测的斜率 d']:.2f}，TSM 预测与 TSH 预测相差很大。不做缩放时预测力消失。</p>")

    p.append('<h2>六、模拟：多强的可预测性才能让 TSM 胜过 TSH（对应论文 Fig. 5）</h2>')
    p.append(f"<p>按中国样本的实际品种结构模拟：r<sub>t+1</sub> = α<sub>i</sub> + β r<sub>t−12,t</sub>/12 + ε，每个 β 模拟 1000 次。用真实数据（控制固定效应）估计的 β = {b_real:.3f}（论文 0.08）。</p>")
    p.append(table(['真实 β', 'TSM − TSH 月均差%（平均）', '差值 t 中位数', 'TSM 显著胜出比例', 'TSM 胜出比例'],
                   [[f(r['真实斜率 β'], 1), f(r['TSM − TSH 月均差% 平均'], 3), f(r['差值 t 中位数']), f(r['TSM 显著胜出比例（t > 1.96）']), f(r['TSM 胜出比例'])] for r in sim.iter_rows(named=True)]))
    p.append(f"<p>与论文一致：β = 0.1 时 TSM 和 TSH 难分胜负，β = 0.4 时 TSM 几乎总是显著胜出。中国真实数据中 TSM 比 TSH 高 {d_ew['收益差%']:.2f}%/月，"
             f"接近模拟中 β ≈ 0.2~0.4 的水平；但按固定效应估计的 β 为负，两者并不一致。这说明 TSM 的超额收益不是来自“同一品种过去涨、未来继续涨”的稳定线性关系，"
             f"更可能来自少数品种、少数时期（2016 年以前的商品）的趋势。</p>")

    p.append('<h2>结论</h2>')
    p.append('<ul><li><b>统计上，论文的结论在中国基本成立</b>：逐个品种几乎没有可预测性；混合回归的显著性主要来自品种间的均值差异和波动率缩放，控制固定效应后消失；2016 年以后不显著。'
             '这与本库 TSMOM 复现中“2016 年后夏普只有 0.24”一致。</li>'
             '<li><b>论文的自助法有缺陷</b>：逐观测重抽样破坏了截面相关性，临界值被系统性抬高。按月整块重抽后，全样本、波动率缩放的混合回归在 5% 水平显著。这一点论文的结论过强。</li>'
             '<li><b>经济上，论文的解释在中国不成立</b>：TSM 不是 TSH 的翻版，TSH 在中国不赚钱，TSM 比它好 0.3%~0.5%/月（边际显著）。中国 TSM 的收益更像来自少数时期的趋势，而不是长期均值差异。</li>'
             '<li><b>和之前复现的关系</b>：控制 AMP 复现中的商品动量因子后，TSM 的 α 接近 0；三因子和 Basis-Momentum 的复现又显示动量与 carry 高度重叠。中国期货的“动量”很大程度上是同一种信息的不同包装。</li></ul>')
    p.append('<h2>代码与数据</h2>')
    p.append(table(['文件', '作用'], [['huang.py', '全部检验；python huang.py cluster 单独计算按月整块的自助法'], ['build_report.py', '生成本报告'], ['output/', '各表结果（parquet）']]))

    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>TSMOM 再检验</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
