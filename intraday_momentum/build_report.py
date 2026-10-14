"""生成日内动量复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'harq'))
from build_report import CSS, table  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')
IDX = ['沪深300', '上证50', '中证500', '中证1000']
BOND = ['10年国债', '5年国债', '2年国债']


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def f2(x, k=2):
    return '' if x is None else f'{x:.{k}f}'


def bt(r, k):
    return '' if r[f'β_{k}×100'] is None else f"{f2(r[f'β_{k}×100'])}（{f2(r[f't_{k}'])}）"


def reg_rows(df, label_col='品种'):
    return [[r[label_col], r['预测变量'], r['天数'], bt(r, 'r1'), bt(r, 'r12'), f2(r['R2%']), f"{f2(r['截距（年化%）'])}（{f2(r['t截距'])}）"]
            for r in df.iter_rows(named=True)]


def build():
    freg, foos, futil, ftim, fsub, fspl = (rd(f'fin_{n}') for n in ('reg', 'oos', 'utility', 'timing', 'subsample', 'splits'))
    creg, coos, ctim = rd('com_reg'), rd('com_oos'), rd('com_timing')
    cost = rd('costs')

    def g(df, prod, var, col):
        return df.filter((pl.col('品种') == prod) & (pl.col('预测变量') == var))[col][0]

    cb = creg.filter(pl.col('预测变量') == 'r1+r12')
    n_c = cb.height
    c_r1 = (cb['t_r1'] > 1.96).sum()
    c_r12 = (cb['t_r12'] > 1.96).sum()
    ct = ctim.filter(pl.col('策略') == 'η(r1,r12)')
    c_tim = (ct['t'] > 1.96).sum()
    cc = cost.filter((pl.col('组') == 'com') & (pl.col('策略') == 'η(r1,r12)'))
    c_net = (cc['扣一跳后bp'] > 0).sum()

    p = []
    p.append('<h1>日内动量复现报告</h1><div class="meta">复现论文：Lei Gao, Yufeng Han, Sophia Zhengzi Li &amp; Guofu Zhou，《Market Intraday Momentum》，'
             '<i>Journal of Financial Economics</i>，2018 · 主样本：沪深 300 股指期货（对应论文的 SPY）；补充：上证 50、中证 500、中证 1000、国债期货、23 个商品期货 · 1 分钟数据</div>')
    p.append(f"<p class=\"lead\">日内动量在中国期货上存在，但<b>驱动它的半小时与论文不同</b>。论文里最重要的是开盘后第一个半小时（含隔夜）的收益 r1；"
             f"在沪深 300 上，r1 只有边际作用（t = {g(freg, '沪深300', 'r1', 't_r1'):.2f}，样本外 R² 为负）。真正有预测力的是收盘前倒数第二个半小时的收益 r12"
             f"（t = {g(freg, '沪深300', 'r12', 't_r12'):.2f}，样本外 R² {g(foos, '沪深300', 'r12', 'R2_OS%'):.2f}%），与论文的 r12 结果几乎一样。四个股指期货都是如此；"
             f"国债期货则反过来，r1 显著、r12 不显著；商品期货 r1 与 r12 都普遍为正。波动大、成交量大、2015 年股灾期间可预测性更强，这点与论文一致。"
             f"按信号择时的毛收益在统计上显著，但每笔只有几个基点：扣掉一跳价差后股指仍有余，商品大多不够，而股指还要面对较高的日内平今手续费。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{g(freg, '沪深300', 'r12', 'β_r12×100'):.1f}（t {g(freg, '沪深300', 'r12', 't_r12'):.1f}）</b><span>沪深 300：r12 的 β×100（论文 SPY 11.8，t 2.62）</span></div>"
             f"<div class=\"kpi\"><b>{g(freg, '沪深300', 'r1', 'β_r1×100'):.1f}（t {g(freg, '沪深300', 'r1', 't_r1'):.1f}）</b><span>沪深 300：r1 的 β×100（论文 6.94，t 4.08）</span></div>"
             f"<div class=\"kpi\"><b>{g(foos, '沪深300', 'r1+r12', 'R2_OS%'):.2f}%</b><span>沪深 300 样本外 R²（r1+r12；论文 2.0%）</span></div>"
             f"<div class=\"kpi\"><b>{c_tim}/{n_c}</b><span>商品中 η(r1,r12) 择时毛收益显著的品种数</span></div></div>")

    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>把一个交易日切成 13 个半小时。论文发现：开盘后第一个半小时的收益（r1，从前一天收盘算起，包含隔夜信息）能正向预测当天最后半小时的收益（r13）；'
             '倒数第二个半小时的收益（r12）也有预测力，并且与 r1 互补。论文给出两个解释：一是知情交易者在收盘前完成调仓，方向与开盘时消化的信息一致；'
             '二是不常交易的投资者（如需要收盘价的机构）在尾盘集中下单。可预测性在高波动、高成交量和金融危机期间更强，而且能转化为可观的择时收益。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['资产', 'SPY（1993–2013），另有 10 只 ETF', '沪深 300 主力合约（2010-04 起）；其他股指期货、国债期货；23 个商品期货作补充'],
        ['半小时划分', '9:30 开盘到 16:00 收盘，共 13 段', '按每天实际交易时间：r1 = 前一交易日收盘 → 开盘后 30 分钟（有夜盘的商品从夜盘开盘起算）；r12 = 收盘前 60→30 分钟；r13 = 收盘前 30 分钟'],
        ['前收盘', '前一日收盘价', '同一合约前一交易日收盘价，避免换月日用错合约'],
        ['回归', 'r13 对 r1、r12 回归，Newey-West t', '同左'],
        ['样本外', '前 5 年起递推估计，R²_OS 相对历史均值', '同左（约 1215 个交易日起，每月重估）'],
        ['分组', '金融危机期（2007–2009）；首半小时波动、成交量三分位', '危机期改为 2015 年股灾（2015-06-15 至 2016-01-31）；分组同左（成交量按年内排名）'],
        ['经济价值', 'η(r1)、η(r12)、η(r1,r12) 择时；均值-方差投资者 γ=5、权重 [−0.5, 1.5] 的 CER；交易成本', '同左；交易成本用 1 跳价差估计（见第四部分）']]))

    p.append('<h2>一、样本内回归（对应论文 Table 2）</h2>')
    p.append('<p>括号内为 Newey-West t 值。论文 SPY：r1 的 β×100 = 6.94（4.08），R² 1.6%；r12 为 11.8（2.62），R² 1.1%；两者一起 R² 2.6%。</p>')
    hd = ['品种', '预测变量', '天数', 'β_r1×100（t）', 'β_r12×100（t）', 'R²%', '截距 年化%（t）']
    p.append('<h3>股指期货</h3>' + table(hd, reg_rows(freg.filter(pl.col('品种').is_in(IDX)))))
    p.append('<h3>国债期货</h3>' + table(hd, reg_rows(freg.filter(pl.col('品种').is_in(BOND)))))
    p.append(f"<ul><li><b>r12 在股指上稳定复现</b>：四个股指期货的 r12 系数在 9~22 之间，t 为 2.3~3.0，与论文 SPY 的 11.8（2.62）同一量级。</li>"
             f"<li><b>r1 在股指上弱</b>：沪深 300 的 r1 系数为 {g(freg, '沪深300', 'r1', 'β_r1×100'):.2f}，只有论文的三分之一，t = {g(freg, '沪深300', 'r1', 't_r1'):.2f}；"
             "四个股指期货都不显著。一种可能的解释是 A 股隔夜消息已在集合竞价和开盘几分钟内消化完，开盘方向未必延续到尾盘（这里只是推测，本次没有检验）。</li>"
             "<li><b>国债期货反过来</b>：r1 显著（t 2.0~3.4），r12 基本不显著。</li>"
             f"<li><b>截距为负</b>：股指期货最后半小时平均是跌的（沪深 300 年化 {g(freg, '沪深300', 'r1+r12', '截距（年化%）'):.1f}%，t = {g(freg, '沪深300', 'r1+r12', 't截距'):.1f}），"
             "论文中 SPY 没有这个现象。择时策略多空对称，不受它影响。</li></ul>")

    p.append('<h2>二、样本外预测与效用增益（对应论文 Table 3、Table 7）</h2>')
    rows = []
    for prod in IDX + BOND:
        q = foos.filter(pl.col('品种') == prod)
        if q.height:
            rows.append([prod, q['样本外起点'][0], q['天数'][0]] + [f"{g(foos, prod, v, 'R2_OS%'):.2f}" for v in ('r1', 'r12', 'r1+r12')])
    p.append(table(['品种', '样本外起点', '天数', 'R²_OS% r1', 'r12', 'r1+r12'], rows + [['论文 SPY', '1998', '', '1.4', '0.9', '2.0']]))
    p.append('<p>中证 1000 上市不足 5 年，没有样本外结果。</p>')
    rows = [[r['品种'], r['预测'], f2(r['年化收益%']), f2(r['年化波动%']), f2(r['夏普']), f2(r['CER%'])] for r in futil.iter_rows(named=True)]
    p.append('<h3>均值-方差投资者（γ = 5，最后半小时仓位 w ∈ [−0.5, 1.5]；CER 增益为相对历史均值预测的年化确定性等价收益之差）</h3>')
    p.append(table(['品种', '预测', '年化收益%', '年化波动%', '夏普', 'CER 增益%'], rows + [['论文 SPY', 'r1', '6.51', '', '1.16', '6.02'], ['论文 SPY', 'r1+r12', '6.68', '', '', '6.18']]))
    p.append(f"<ul><li>沪深 300：只用 r1 时 R²_OS 为负（{g(foos, '沪深300', 'r1', 'R2_OS%'):.2f}%），加入 r12 后为 {g(foos, '沪深300', 'r1+r12', 'R2_OS%'):.2f}%，接近论文的 2.0%；"
             f"中证 500 为 {g(foos, '中证500', 'r1+r12', 'R2_OS%'):.2f}%。</li>"
             "<li>10 年、5 年国债期货的 r1 样本外为正，与样本内一致。上证 50、2 年国债的样本外期较短，结果为负或接近零。</li>"
             "<li>效用增益：只用 r1 在沪深 300 上为负；r1+r12 为正，但比论文小（沪深 300 约 1.4%，中证 500 约 4.7%，论文约 6%）。</li></ul>")

    p.append('<h2>三、什么时候更容易预测（对应论文 Table 4、Table 5）</h2>')
    p.append('<h3>2015 年股灾期 vs 其他时间</h3>')
    p.append(table(['品种 · 样本', '预测变量', '天数', 'β_r1×100（t）', 'β_r12×100（t）', 'R²%', '截距 年化%（t）'],
                   reg_rows(fsub.with_columns((pl.col('品种') + ' · ' + pl.col('样本')).alias('lab')).filter(pl.col('预测变量') == 'r1+r12'), 'lab')))
    p.append('<p>论文：金融危机期 R² 为 4.1%（r1）、6.9%（r1+r12），其他时间为 0.8%、1.1%。沪深 300 股灾期的 R²（r1+r12）约 4.4%，其他时间 0.8%，规律相同；'
             '但股灾期只有 156 天，系数不显著。剔除股灾后，股指的 r12 依然显著（沪深 300 t = 3.3），说明结论不是由股灾驱动的。</p>')
    p.append('<h3>沪深 300：按首半小时波动、成交量三分组（r1+r12）</h3>')
    p.append(table(['分组', '天数', 'β_r1×100（t）', 'β_r12×100（t）', 'R²%'],
                   [[r['样本'], r['天数'], bt(r, 'r1'), bt(r, 'r12'), f2(r['R2%'])] for r in fspl.filter(pl.col('品种') == '沪深300').iter_rows(named=True)]))
    p.append('<p>论文：波动三分组 R² 为 0.6% / 1.0% / 3.3%，成交量三分组为 1.1% / 2.3% / 3.1%。沪深 300 的可预测性同样几乎全部集中在高波动、高成交量的日子（R² 约 3%~4%），低、中两组基本没有。</p>')

    p.append('<h2>四、择时策略与交易成本（对应论文 Table 6、Table 8）</h2>')
    p.append('<p>η(r1)：r1 &gt; 0 就在最后半小时做多，否则做空；η(r12) 同理；η(r1,r12)：两者同号才交易，否则空仓。常多 = 每天最后半小时做多；买入持有 = 持有全天。收益未扣成本。'
             'η(r1,r12) 的成功率与每笔收益只按实际交易的日子计算。</p>')
    rows = []
    for prod in ['沪深300', '中证500', '10年国债']:
        for r in ftim.filter(pl.col('品种') == prod).iter_rows(named=True):
            rows.append([prod, r['策略'], f2(r['年化收益%']), f2(r['t']), f2(r['年化波动%']), f2(r['夏普']), f2(r['偏度']), f2(r['成功率%'], 1), f2(r['每笔平均收益bp'])])
    rows += [['论文 SPY', 'η(r1)', '6.67', '4.36', '6.19', '1.08', '', '54.4', ''], ['论文 SPY', 'η(r12)', '1.77', '', '', '0.29', '', '', ''],
             ['论文 SPY', 'η(r1,r12)', '4.39', '', '', '0.98', '', '', ''], ['论文 SPY', '常多', '−1.11', '', '', '', '', '', ''],
             ['论文 SPY', '买入持有', '6.04', '', '', '0.29', '', '', '']]
    p.append(table(['品种', '策略', '年化收益%', 't', '年化波动%', '夏普', '偏度', '成功率%', '每笔平均 bp'], rows))
    p.append('<h3>每笔毛收益 vs 一跳价差成本（η(r1,r12)）</h3>')
    p.append('<p>一个来回按买卖价差 1 跳计算（进出各付半跳），即 最小变动价位 / 价格，取样本期平均；不含交易所手续费。论文的做法类似：比较策略的盈亏平衡成本与 SPY 的实际买卖价差。</p>')
    p.append(table(['品种', '每笔毛收益 bp', '一跳成本 bp', '扣除后 bp'],
                   [[r['品种'], f2(r['每笔平均收益bp']), f2(r['一跳成本bp']), f2(r['扣一跳后bp'])]
                    for r in cost.filter((pl.col('组') == 'fin') & (pl.col('策略') == 'η(r1,r12)')).iter_rows(named=True)]))
    p.append('<ul><li>股指期货：每笔毛收益 2~6 bp，一跳只要 0.3~0.7 bp，扣除价差后仍为正。但这是当天开平仓，股指期货的平今手续费远高于普通开平仓'
             '（近年约万分之二到三，2015–2018 年更高），仅这一项就可能吃掉沪深 300 的大部分毛收益。</li>'
             '<li>国债期货：统计上最稳（t 约 4），但每笔不到 1 bp，扣掉一跳就所剩无几。</li>'
             '<li>与论文的差别：论文中 SPY 的 η(r1) 最好；中国股指上 η(r12) 更好，与回归结果一致。</li></ul>')

    p.append('<h2>五、补充：商品期货</h2>')
    p.append(f"<p>商品的 r1 从夜盘开盘算起（前收盘为前一日日盘收盘）。{n_c} 个商品中，r1 显著为正的有 {c_r1} 个，r12 显著为正的有 {c_r12} 个，"
             f"η(r1,r12) 择时毛收益显著的有 {c_tim} 个；但扣掉一跳价差后仍为正的只有 {c_net} 个。商品的最小变动价位相对价格偏大（多数为 2~10 bp），日内动量的利润大多被价差吞掉。</p>")
    cm = {r['品种']: r for r in cc.iter_rows(named=True)}
    rows = []
    for r in cb.iter_rows(named=True):
        t = ct.filter(pl.col('品种') == r['品种']).row(0, named=True)
        rows.append([r['品种'], r['天数'], bt(r, 'r1'), bt(r, 'r12'), f2(r['R2%']), f2(g(coos, r['品种'], 'r1+r12', 'R2_OS%')),
                     f"{f2(t['年化收益%'])}（{f2(t['t'])}）", f2(cm[r['品种']]['每笔平均收益bp']), f2(cm[r['品种']]['扣一跳后bp'])])
    p.append(table(['品种', '天数', 'β_r1×100（t）', 'β_r12×100（t）', 'R²%', 'R²_OS%', 'η(r1,r12) 年化%（t）', '每笔毛 bp', '扣一跳后 bp'], rows))

    p.append('<h2>结论</h2>')
    p.append('<ul><li><b>日内动量存在</b>：在股指、国债和绝大多数商品期货上，当天较早的收益都能正向预测最后半小时；可预测性集中在高波动、高成交量和危机时期，与论文一致。</li>'
             '<li><b>但论文的核心变量 r1 在中国股指上不成立</b>：股指期货的预测力来自 r12（收盘前一小时到半小时），r1 只有边际作用、样本外为负。国债期货则以 r1 为主。'
             '所以在 A 股股指上，“开盘信息延续到收盘”的证据不足，“尾盘交易方向延续”更贴合数据。</li>'
             '<li><b>经济价值有限</b>：毛收益显著，但每笔只有几个基点。商品的利润被价差吞掉；股指扣价差后仍为正，但还要面对日内平今手续费。</li></ul>')
    p.append('<h2>代码与数据</h2>')
    p.append(table(['文件', '作用'], [['imom.py', '半小时收益构造、回归、样本外、分组、择时与效用'], ['costs.py', '一跳价差成本'], ['build_report.py', '生成本报告'],
                                    ['output/halfhours_*.parquet', '每个品种逐日的 r1、r12、r13、全天收益、首半小时波动与成交量']]))

    html = ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>日内动量复现</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
