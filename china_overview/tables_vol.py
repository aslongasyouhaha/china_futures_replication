"""全貌文档第三节（波动率）的表。数字全部取自各复现文件夹的 output/。"""
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(HERE, 'tables')
sys.path.insert(0, HERE)
from tables_xs import cell, table  # noqa: E402

STARS = '\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。'
NAME = {'IF': '沪深 300', 'CU': '铜', 'RB': '螺纹钢', 'M': '豆粕'}


def plain(rows_cols, fmt='{:.3f}'):
    return [fmt.format(v) if v is not None and v == v else '' for v in rows_cols]


# ---------------------------------------------------------------- 表 11：HAR 与 HARQ 的样本内估计
def har_insample():
    d = pl.read_parquet(f'{R}/harq/output/insample.parquet').filter((pl.col('h') == 1) & pl.col('模型').is_in(['HAR', 'HARQ']))
    rs = []
    for p in ('IF', 'CU', 'RB', 'M'):
        for m in ('HAR', 'HARQ'):
            rs.append(d.filter((pl.col('品种') == p) & (pl.col('模型') == m)).row(0, named=True))
    cols = [f"{NAME[r['品种']]}<br>{r['模型']}" for r in rs]

    def g(r, k, fmt='{:.3f}'):
        b, se = r.get(k), r.get('se_' + k)
        return cell(b, b / se, fmt) if b is not None and se else ''
    rows = [('RV(t−1)，日', [g(r, 'β1') for r in rs]), ('过去 5 天平均 RV，周', [g(r, 'β2') for r in rs]),
            ('过去 22 天平均 RV，月', [g(r, 'β3') for r in rs]), ('√RQ(t−1) × RV(t−1)', [g(r, 'β1Q', '{:.4f}') for r in rs]),
            ('Constant', [g(r, 'β0') for r in rs])]
    foot = [('样本数（天）', [str(n) for n in pl.read_parquet(f'{R}/harq/output/summary.parquet').filter(pl.col('品种').is_in(['IF', 'CU', 'RB', 'M']))
                                .sort(pl.col('品种').replace({'IF': '0', 'CU': '1', 'RB': '2', 'M': '3'}))['天数'].to_list() for _ in (0, 1)]),
            ('R²', plain([r['R2'] for r in rs])), ('持续性（β1 + β2 + β3）', plain([r['持续性'] for r in rs])),
            ('平均滞后（天）', plain([r['平均滞后'] for r in rs], '{:.2f}'))]
    # 样本数行按列展开：每个品种两列
    n = pl.read_parquet(f'{R}/harq/output/summary.parquet').select('品种', '天数')
    nd = dict(zip(n['品种'], n['天数']))
    foot[0] = ('样本数（天）', [str(nd[r['品种']]) for r in rs])
    note = ('被解释变量为次日已实现方差 RV(t)（5 分钟收益平方和）；HARQ 让昨天 RV 的系数随测量误差 √RQ 变化。括号内为 t 值（Newey-West 标准误）；'
            + STARS + '其余 19 个商品的结果见 harq 复现报告。来源：Bollerslev, Patton & Quaedvlieg，Journal of Econometrics，2016 的复现。')
    return table(cols, rows, foot, note)


# ---------------------------------------------------------------- 表 12：HAR 与其他模型的样本外比较（22 天）
def har_oos():
    o = pl.read_parquet(f'{R}/har_rv/output/har_oos.parquet').filter(pl.col('h') == 22)
    p = o.pivot(values='RMSE', index='品种', columns='模型')
    models = [('AR1', 'AR(1)'), ('AR3', 'AR(3)'), ('AR22', 'AR(22)'), ('ARFIMA', 'ARFIMA'), ('EWMA', 'EWMA（RiskMetrics）'), ('EWMA20', 'EWMA（20 天）')]
    order = ['沪深300股指', '10年期国债', '螺纹钢', '铜', '豆粕', '原油', '黄金', 'PTA']
    rows = []
    ratios = {m: [] for m, _ in models}
    for prod in order:
        r = p.filter(pl.col('品种') == prod).row(0, named=True)
        cs = []
        for m, _ in models:
            v = r[m] / r['HAR']
            ratios[m].append(v)
            cs.append(f'{v:.3f}')
        rows.append((prod, cs))
    rows.append(('平均', [f'{np.mean(ratios[m]):.3f}' for m, _ in models]))
    note = ('每格为该模型未来 22 天平均波动预测的 RMSE ÷ HAR 的 RMSE；小于 1 表示比 HAR 好。1000 日滚动窗口逐日重估。'
            '来源：Corsi，Journal of Financial Econometrics，2009 的复现。')
    return table([lab for _, lab in models], rows, [], note)


# ---------------------------------------------------------------- 表 13：HARQ 的样本外比较（22 个商品中位数）
def harq_oos():
    q = pl.read_parquet(f'{R}/harq/output/oos.parquet').filter((pl.col('品种') != 'IF') & (pl.col('窗口') == 'RW'))
    c = q.group_by('h', '损失', '模型').agg(pl.col('比值').median())
    models = ['AR', 'ARQ', 'HAR-J', 'CHAR', 'SHAR', 'HARQ', 'HARQ-F', 'HARQ-h']
    specs = [('MSE', 1), ('MSE', 5), ('MSE', 22), ('QLIKE', 1), ('QLIKE', 5), ('QLIKE', 22)]
    rows = []
    for m in models:
        cs = []
        for loss, h in specs:
            v = c.filter((pl.col('模型') == m) & (pl.col('损失') == loss) & (pl.col('h') == h))['比值']
            cs.append(f'{v[0]:.3f}' if len(v) and v[0] is not None else '—')
        rows.append((m, cs))
    note = ('每格为该模型的预测损失 ÷ HAR 的损失，取 22 个商品的中位数；小于 1 表示比 HAR 好。1000 日滚动窗口逐日重估；h 为预测天数。'
            'QLIKE 是对波动率预测更常用的损失函数，对低估波动惩罚更重。来源：Bollerslev, Patton & Quaedvlieg，Journal of Econometrics，2016 的复现。')
    return table([f'{l}<br>h = {h}' for l, h in specs], rows, [], note)


# ---------------------------------------------------------------- 表 14：粗糙度
def rough():
    h = pl.read_parquet(f'{R}/rough_vol/output/h_estimates.parquet')
    cv = pl.read_parquet(f'{R}/rough_vol/output/cov.parquet')
    lm = pl.read_parquet(f'{R}/rough_vol/output/longmem.parquet')
    dist = pl.read_parquet(f'{R}/rough_vol/output/dist.parquet').filter(pl.col('代理') == '1 小时窗口')
    order = ['沪深300股指', '10年期国债', '螺纹钢', '铜', '豆粕', '原油', '黄金', 'PTA']
    rows = []
    for p in order:
        h1 = h.filter((pl.col('品种') == p) & (pl.col('代理') == '1 小时窗口')).row(0, named=True)
        hd = h.filter((pl.col('品种') == p) & (pl.col('代理') == '全天 RV')).row(0, named=True)
        c1 = cv.filter((pl.col('品种') == p) & (pl.col('代理') == '1 小时窗口')).row(0, named=True)
        l = lm.filter(pl.col('品种') == p).row(0, named=True)
        k = dist.filter(pl.col('品种') == p).row(0, named=True)
        rows.append((p, [f"{h1['H(=ζ2/2)']:.3f}", f"{hd['H(=ζ2/2)']:.3f}", f"{h1['前半H']:.3f} / {h1['后半H']:.3f}", f"{c1['R²: Cov 对 Δ^{2H}']:.3f}",
                         f"{k['峰度 Δ=1']:.2f}", f"{l['实际 RV 的 d']:.2f}", f"{l['模拟 RFSV 的 d（20 次均值）']:.2f}"]))
    note = ('H 为 Hurst 指数：对数波动率增量的 q 阶矩对时间间隔 Δ 回归的斜率 ÷ q（这里 q = 2）；布朗运动 H = 0.5，越小越"粗糙"。'
            '1 小时窗口 = 每天 10:30–11:29 的 1 分钟收益已实现波动率；全天 = 5 分钟 RV。Cov 对 Δ^2H 的 R² 检验 RFSV 模型的协方差结构；'
            '峰度为 Δ = 1 天对数波动率增量的峰度（正态 = 3）；d 为 ARFIMA 长记忆参数，"模拟"为按估计的 H 模拟 RFSV 路径后再估 d（20 次均值）。'
            '来源：Gatheral, Jaisson & Rosenbaum，Quantitative Finance，2018 的复现。')
    return table(['H<br>1 小时窗口', 'H<br>全天 RV', 'H<br>前半 / 后半样本', 'Cov 对 Δ^2H<br>R²', '增量峰度<br>Δ = 1', '实际 d', '模拟 RFSV<br>的 d'], rows, [], note)


# ---------------------------------------------------------------- 表 15：RFSV 的样本外预测
def rfsv():
    f = pl.read_parquet(f'{R}/rough_vol/output/forecast.parquet').pivot(values='P', index=['品种', 'Δ'], columns='模型')
    order = ['沪深300股指', '10年期国债', '螺纹钢', '铜', '豆粕', '原油', '黄金', 'PTA']
    rows, acc = [], {d: [] for d in (1, 5, 20)}
    for p in order:
        cs = []
        for d in (1, 5, 20):
            r = f.filter((pl.col('品种') == p) & (pl.col('Δ') == d)).row(0, named=True)
            v = r['RFSV'] / r['HAR']
            acc[d].append(v)
            best = min(('RFSV', 'HAR', 'AR(5)', 'AR(10)'), key=lambda k: r[k])
            cs.append(f'{v:.3f}' + ('（最优）' if best == 'RFSV' else ''))
        rows.append((p, cs))
    rows.append(('平均', [f'{np.mean(acc[d]):.3f}' for d in (1, 5, 20)]))
    note = ('每格为 RFSV 预测的误差指标 P ÷ HAR 的 P；小于 1 表示 RFSV 比 HAR 好；"最优"表示在 RFSV、HAR、AR(5)、AR(10) 中误差最小。'
            'Δ 为预测天数。来源：Gatheral, Jaisson & Rosenbaum，Quantitative Finance，2018 的复现。')
    return table(['Δ = 1 天', 'Δ = 5 天', 'Δ = 20 天'], rows, [], note)


# ---------------------------------------------------------------- 表 16：波动率管理组合
def vmp():
    d = pl.read_parquet(f'{R}/volatility_managed/output/table1.parquet')
    rs = list(d.iter_rows(named=True))
    cols = [r['因子'] for r in rs]
    rows = [('α（年化 %）', [cell(r['α年化'], r['t_α'], '{:.2f}') for r in rs]), ('原因子 f 的系数 β', plain([r['β'] for r in rs]))]
    foot = [('样本起点', [r['起始'][:7] for r in rs]), ('样本数（月）', [str(r['月数']) for r in rs]), ('R²', plain([r['R2'] for r in rs])),
            ('原因子夏普', plain([r['原夏普'] for r in rs], '{:.2f}')), ('管理后夏普', plain([r['管理后夏普'] for r in rs], '{:.2f}'))]
    note = ('回归 f^σ(t) = α + β·f(t) + ε，其中 f^σ = c ÷ 上月已实现方差 × f，c 使两者全样本波动相同。括号内为 α 的 Newey-West t 值；' + STARS +
            '商品平均 = 等权做多全部商品主力；TSMOM、商品 carry 为前两篇复现的论文口径日收益。来源：Moreira & Muir，Journal of Finance，2017 的复现。')
    return table(cols, rows, foot, note)


# ---------------------------------------------------------------- 表 17：波动率管理的拆分
def vmp_split():
    d = pl.read_parquet(f'{R}/volatility_managed/output/regime_split.parquet')
    facs = ['沪深300', '中证500', '10年国债', '商品平均', 'TSMOM', '商品carry']
    rows = []
    for how, lab in (('论文做法（1/上月方差）', '论文做法：1 ÷ 上月方差'), ('只用长期水平', '只按长期波动水平调整'), ('只用短期偏离', '只按短期偏离调整')):
        cs = []
        for f in facs:
            r = d.filter((pl.col('因子') == f) & (pl.col('做法') == how)).row(0, named=True)
            cs.append(cell(r['α年化'], r['t_α'], '{:.2f}'))
        rows.append((lab, cs))
    note = ('每格为管理后组合对原因子回归的 α（年化 %），括号内为 Newey-West t 值；' + STARS +
            '上月方差 = 长期水平（过去 12 个月平均方差）× 短期偏离（上月 ÷ 长期水平），分别只用其中一部分调整仓位；需要 12 个月历史，样本比表 26 少约 11 个月，所以第一行与表 26 略有差别。此为复现中的补充分析，不是论文内容。'
            '来源：Moreira & Muir，Journal of Finance，2017 的复现。')
    return table(facs, rows, [], note)


if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True)
    for name, fn in [('vol_har_insample', har_insample), ('vol_har_oos', har_oos), ('vol_harq_oos', harq_oos), ('vol_rough', rough),
                     ('vol_rfsv', rfsv), ('vol_vmp', vmp), ('vol_vmp_split', vmp_split)]:
        s = fn()
        with open(os.path.join(TAB, f'{name}.md'), 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(f'==== {name}\n{s}')
