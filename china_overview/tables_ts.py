"""全貌文档第三节（时序择时）的表。数字取自各复现文件夹的 output/；TSMOM 与时序 carry 的结果没有存成 parquet，取自其 report.html 的表格（已核对）。"""
import os
import re
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(HERE, 'tables')
sys.path.insert(0, HERE)
from tables_xs import cell, table  # noqa: E402

STARS = '\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。'


def parse(s):
    """'−0.82（−2.54）' → (−0.82, −2.54)"""
    if s is None:
        return None, None
    m = re.match(r'([+\-−]?[\d.]+)（([+\-−]?[\d.]+)）', s.replace('−', '-'))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


# ---------------------------------------------------------------- 表：TSMOM（Moskowitz-Ooi-Pedersen）
def tsmom():
    # 取自 time_series_momentum/report.html
    rows = [('论文口径（不扣成本、月末收盘成交）', ['9.2%', '0.61', '2.76***', '−40.9%', '9.6% / 0.67', '6.3% / 0.53']),
            ('实盘口径（资金 1 亿、次日 VWAP、扣全部成本）', ['4.5%', '0.31', '1.40', '−43.2%', '— / 0.39', '— / 0.24']),
            ('实盘口径，资金 1000 万', ['5.7%', '0.37', '', '', '', '']),
            ('实盘口径，资金 10 亿', ['2.5%', '0.20', '', '', '', '']),
            ('对照：始终做多', ['', '0.17', '', '', '', ''])]
    note = ('回看 12 个月；仓位 = 40% ÷ 年化波动率，当月可交易品种等权；2006-01 至 2026-09，含商品、股指、国债期货。实盘口径年化成本 2.12%，其中换月 1.48%。'
            't 值的星号按 ' + STARS + '来源：Moskowitz, Ooi & Pedersen，Journal of Financial Economics，2012 的复现（数字取自复现报告）。')
    return table(['年化收益', '夏普', 't', '最大回撤', '2010 年起<br>年化 / 夏普', '2016 年起<br>年化 / 夏普'], rows, [], note)


# ---------------------------------------------------------------- 表：Huang 等的混合回归
def huang_pooled():
    p = pl.read_parquet(f'{R}/tsmom_huang/output/pooled_cluster.parquet').filter((pl.col('h') == 12) & (pl.col('预测变量') == '过去 h 月累计收益'))
    samples = [('全部资产', '全部资产'), ('商品', '商品'), ('全部资产<br>控制品种固定效应', '控制固定效应'), ('全部资产<br>2005–2015', '2005–2015'),
               ('全部资产<br>2016–2026', '2016–2026'), ('全部资产<br>不做波动率缩放', '不做波动率缩放')]
    rs = [p.filter(pl.col('样本') == s).row(0, named=True) for _, s in samples]
    rows = [('过去 12 个月累计收益', [cell(r['β'], r['t']) for r in rs])]
    foot = [('按月整块自助法 p 值（wild）', [f"{r['wild cluster p']:.3f}" for r in rs]), ('观测数（品种-月）', [str(r['观测数']) for r in rs])]
    note = ('混合回归 r(t+1)/σ(t) = α + β·r(t−12, t) + ε，所有品种叠在一起；括号内为按月聚类的 t 值；' + STARS +
            '自助法 p 值为单侧、在"无可预测性"的原假设下按月整块重抽得到（论文逐观测重抽，会破坏同月品种间的相关、抬高临界值，见正文）。'
            '来源：Huang, Li, Wang & Zhou，Journal of Financial Economics，2020 的复现。')
    return table([s for s, _ in samples], rows, foot, note)


# ---------------------------------------------------------------- 表：TSM 与 TSH
def huang_tsh():
    t = pl.read_parquet(f'{R}/tsmom_huang/output/table10.parquet').filter(pl.col('腿') == '多 − 空')
    d = pl.read_parquet(f'{R}/tsmom_huang/output/table10_diff.parquet')
    ws = ['等权', '波动率加权（40%/σ）', '按过去收益加权', '多空各 1 元（Goyal-Jegadeesh）']
    g = lambda s, w: t.filter((pl.col('策略') == s) & (pl.col('权重') == w)).row(0, named=True)   # noqa: E731
    rows = [('TSM：过去 12 个月收益为正做多、为负做空', [cell(g('TSM', w)['月均%'], g('TSM', w)['t'], '{:.2f}') for w in ws]),
            ('TSH：上市以来累计收益为正做多、为负做空', [cell(g('TSH', w)['月均%'], g('TSH', w)['t'], '{:.2f}') for w in ws]),
            ('TSM − TSH', [f"{r['收益差%']:.2f}<br>[p = {r['收益差 p']:.2f}]" for r in (d.filter(pl.col('权重') == w).row(0, named=True) for w in ws)]),
            ('TSM 对"市场 + 商品价值、动量因子"的 α', [cell(g('TSM', w)['α% AMP 中国版'], g('TSM', w)['t(α) AMP 中国版'], '{:.2f}') for w in ws])]
    note = ('每格为多空组合的月均收益（%），括号内为 t 值，方括号内为差值的 p 值；' + STARS + 'TSH 不需要任何可预测性，只按品种的历史均值方向持仓。'
            '"市场"为全部品种等权做多，商品价值、动量因子来自 Asness-Moskowitz-Pedersen 的复现。来源：Huang, Li, Wang & Zhou，Journal of Financial Economics，2020 的复现。')
    return table(['等权', '波动率加权', '按过去收益加权', '多空各 1 元'], rows, [], note)


# ---------------------------------------------------------------- 表：时序 carry
def ts_carry():
    # 取自 carry/report.html 第一部分
    rows = [('截面 carry1-12（参照）', ['11.9%', '13.0%', '0.91', '4.16***', '−31.7%']),
            ('时序 carry：贴水做多、升水做空（阈值 0）', ['10.8%', '13.2%', '0.82', '3.73***', '−33.0%']),
            ('时序 carry：高于自身历史均值做多、否则做空', ['8.2%', '11.7%', '0.70', '3.16***', '−35.7%'])]
    note = ('论文口径：不扣成本、月末收盘成交；商品，2006-01 至 2026-09。carry = ln(近月价 ÷ 远月价) × 12 ÷ 相隔月数。' + STARS +
            '来源：Koijen et al.，Journal of Financial Economics，2018 的复现（数字取自复现报告）。')
    return table(['年化收益', '年化波动', '夏普', 't', '最大回撤'], rows, [], note)


# ---------------------------------------------------------------- 表：Hong-Yogo
def hong_yogo():
    t6 = pl.read_parquet(f'{R}/open_interest/output/table6.parquet')
    sp = pl.read_parquet(f'{R}/open_interest/output/subperiods.parquet')
    rs = [r for r in t6.iter_rows(named=True)] + [r for r in sp.iter_rows(named=True) if r['回归'].startswith('(2)')]
    cols = ['全样本'] * 6 + ['2010–2015', '2016–2026']

    def g(r, k):
        b, t = parse(r.get(k))
        return cell(b, t, '{:.2f}') if b is not None else ''
    rows = [('整体基差', [g(r, 'basis') for r in rs]), ('整体持仓金额增长（12 个月）', [g(r, 'OI12') for r in rs]),
            ('整体持仓张数增长（12 个月）', [g(r, 'OI12_n') for r in rs]), ('过去 12 个月收益', [g(r, 'R12') for r in rs])]
    foot = [('样本数（月）', [str(r['月数']) for r in rs]), ('R²', [f"{r['R2%'] / 100:.3f}" for r in rs])]
    note = ('被解释变量为下月商品整体收益（板块等权）；解释变量标准化，系数 = 每 1 个标准差对应的月收益（%）。括号内为 Newey-West t 值（滞后 12）；' + STARS +
            '论文另有短期利率、期限利差、经济活动指数等控制变量，中国缺少对应数据，未纳入。来源：Hong & Yogo，Journal of Financial Economics，2012 的复现。')
    return table(cols, rows, foot, note)


# ---------------------------------------------------------------- 表：日内动量
def intraday():
    fin = pl.read_parquet(f'{R}/intraday_momentum/output/fin_reg.parquet').filter(pl.col('预测变量') == 'r1+r12')
    com = pl.read_parquet(f'{R}/intraday_momentum/output/com_reg.parquet').filter(pl.col('预测变量') == 'r1+r12')
    cost = pl.read_parquet(f'{R}/intraday_momentum/output/costs.parquet').filter(pl.col('策略') == 'η(r1,r12)')
    names = [('沪深300', fin), ('中证500', fin), ('10年国债', fin), ('螺纹钢', com), ('铜', com), ('豆粕', com), ('白糖', com)]
    rs = [d.filter(pl.col('品种') == n).row(0, named=True) for n, d in names]
    cs = [cost.filter(pl.col('品种') == n) for n, _ in names]
    rows = [('r1：前收盘 → 开盘后 30 分钟', [cell(r['β_r1×100'], r['t_r1'], '{:.2f}') for r in rs]),
            ('r12：收盘前倒数第二个半小时', [cell(r['β_r12×100'], r['t_r12'], '{:.2f}') for r in rs]),
            ('Constant（年化 %）', [cell(r['截距（年化%）'], r['t截距'], '{:.2f}') for r in rs])]
    foot = [('样本数（天）', [str(r['天数']) for r in rs]), ('R²', [f"{r['R2%'] / 100:.4f}" for r in rs]),
            ('择时每笔平均收益（bp）', [f"{c['每笔平均收益bp'][0]:.2f}" if c.height else '' for c in cs]),
            ('一跳价差成本（bp）', [f"{c['一跳成本bp'][0]:.2f}" if c.height else '' for c in cs])]
    n_pos = cost.filter(pl.col('组') == 'com')
    note = ('被解释变量为最后半小时收益 r13；系数 × 100。括号内为 Newey-West t 值；' + STARS +
            f'择时 = 按 r1、r12 的符号在最后半小时做多或做空。23 个商品中，扣一跳价差后每笔仍为正的只有 {int((n_pos["扣一跳后bp"] > 0).sum())} 个。'
            '来源：Gao, Han, Li & Zhou，Journal of Financial Economics，2018 的复现。')
    return table([n for n, _ in names], rows, foot, note)


if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True)
    for name, fn in [('ts_tsmom', tsmom), ('ts_huang_pooled', huang_pooled), ('ts_huang_tsh', huang_tsh), ('ts_carry', ts_carry),
                     ('ts_hong_yogo', hong_yogo), ('ts_intraday', intraday)]:
        s = fn()
        with open(os.path.join(TAB, f'{name}.md'), 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(f'==== {name}\n{s}')
