"""全貌文档第二节（截面溢价）的论文格式表。数字取自各复现文件夹的 output/；
少数表（三因子 spanning、价值动量对 carry）原输出只存了 α，这里用已保存的因子序列重跑同一回归以列出全部系数。"""
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(HERE, 'tables')
sys.path.insert(0, HERE)
from market import ols_nw  # noqa: E402
from tables import star  # noqa: E402


def cell(b, t, fmt='{:.3f}'):
    if b is None or t is None or (isinstance(b, float) and np.isnan(b)):
        return ''
    return f'{fmt.format(b)}{star(t)}<br>({t:.2f})'


def table(cols, rows, foot, note):
    """cols: 列名；rows: [(行名, [cell...])]；foot: [(行名, [str...])]。"""
    lines = ['| | ' + ' | '.join(f'({i + 1})<br>{c}' for i, c in enumerate(cols)) + ' |', '|---|' + '---|' * len(cols)]
    for lab, cs in rows + foot:
        lines.append(f'| {lab} | ' + ' | '.join(cs) + ' |')
    return '\n'.join(lines) + '\n\n' + note + '\n'


def reg(y, X):
    b, t, r2, T = ols_nw(y, X)
    k = np.asarray(X).reshape(len(y), -1).shape[1]
    return b, t, T, r2, 1 - (1 - r2) * (T - 1) / (T - k - 1)


# ---------------------------------------------------------------- 表 3：carry 与动量的互相解释（Bakshi-Gao-Rossi）
def bgr_spanning():
    cols, res = [], []
    for v, lab in (('paper', '次近月（论文方法）'), ('main', '主力合约')):
        f = pl.read_parquet(f'{R}/commodity_three_factor/output/factors_{v}.parquet').drop_nulls(['AVG', 'CARRY', 'CMOM'])
        for y, x in (('CMOM', 'CARRY'), ('CARRY', 'CMOM')):
            b, t, T, r2, ar2 = reg(f[y].to_numpy(), f.select('AVG', x).to_numpy())
            cols.append(f'{"动量" if y == "CMOM" else "carry"}<br>{lab}')
            res.append({'α': (b[0] * 1200, t[0]), 'AVG': (b[1], t[1]), x: (b[2], t[2]), 'N': T, 'R2': r2, 'aR2': ar2})
    rows = [('α（年化 %）', [cell(*r['α'], '{:.2f}') for r in res]),
            ('平均因子 AVG', [cell(*r['AVG']) for r in res]),
            ('carry 因子', [cell(*r['CARRY']) if 'CARRY' in r else '' for r in res]),
            ('动量因子', [cell(*r['CMOM']) if 'CMOM' in r else '' for r in res])]
    foot = [('样本数（月）', [str(r['N']) for r in res]), ('R²', [f"{r['R2']:.3f}" for r in res]), ('调整 R²', [f"{r['aR2']:.3f}" for r in res])]
    note = ('被解释变量为因子的月收益；α 为年化 %。carry 因子 = 最贴水 5 个 − 最升水 5 个，动量因子 = 过去 6 个月收益最高 5 个 − 最低 5 个，AVG = 全部品种等权做多。'
            '括号内为 Newey-West t 值；\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。来源：Bakshi, Gao & Rossi，Management Science，2019 的复现。')
    return table(cols, rows, foot, note)


# ---------------------------------------------------------------- 表 4：主力合约 12 个月涨幅的拆分（截面 Fama-MacBeth）
def roll_split():
    d = pl.read_parquet(f'{R}/commodity_three_factor/output/cmp_roll_split_fm.parquet')
    rows = [(r['变量'], [cell(r['系数%'], r['t'])]) for r in d.iter_rows(named=True)]
    note = ('每月对下月收益做截面回归，再对系数取时间序列平均（Fama-MacBeth）；解释变量均标准化，系数 = 每 1 个标准差对应的下月收益（%）。'
            '三个变量同时进入回归。来源：Bakshi, Gao & Rossi，Management Science，2019 的复现（主力合约扩展部分）。')
    return table(['下月收益'], rows, [], note)


# ---------------------------------------------------------------- 表 5：价值、动量与 carry（Asness-Moskowitz-Pedersen）
def amp_carry():
    sys.path.insert(0, f'{R}/value_momentum')
    import vme
    p = pl.read_parquet(f'{R}/value_momentum/output/portfolios.parquet')   # C_F 为价值、动量各半的组合
    # carry1-12 因子：与 vme.carry_link 相同的构造（同一品种池、同一月份，按排名加权）
    u = pl.read_parquet(f'{R}/value_momentum/output/panel.parquet')
    cp = pl.read_parquet(vme.FA.CARRY_PANEL).sort(['product', 'trading_day']).with_columns(pl.col('trading_day').dt.truncate('1mo').alias('m'))
    cp = cp.with_columns(pl.col('carry').rolling_mean(12, min_periods=12).over('product').alias('c12')).select('product', 'm', 'c12')
    k = vme.sorts(u.join(cp, on=['product', 'm'], how='left'), 'c12').select('m', pl.col('F').alias('K_F'))
    p = p.join(k, on='m').sort('m')
    cols, res = [], []
    for y, lab in (('M_F', '动量'), ('V_F', '价值'), ('C_F', '价值 + 动量各半')):
        for xs in (['B'], ['B', 'K_F']):
            z = p.select([y] + xs).drop_nulls()
            b, t, T, r2, ar2 = reg(z[y].to_numpy(), z.select(xs).to_numpy())
            r = {'α': (b[0] * 1200, t[0]), 'N': T, 'R2': r2, 'aR2': ar2}
            r.update({x: (b[i + 1], t[i + 1]) for i, x in enumerate(xs)})
            cols.append(lab)
            res.append(r)
    rows = [('α（年化 %）', [cell(*r['α'], '{:.2f}') for r in res]),
            ('等权商品篮子', [cell(*r['B']) for r in res]),
            ('carry1-12 因子', [cell(*r['K_F']) if 'K_F' in r else '' for r in res])]
    foot = [('样本数（月）', [str(r['N']) for r in res]), ('R²', [f"{r['R2']:.3f}" for r in res]), ('调整 R²', [f"{r['aR2']:.3f}" for r in res])]
    note = ('被解释变量为按排名加权的因子月收益（多空各 1 元）；carry1-12 因子 = 过去 12 个月平均 carry 按排名加权，与价值、动量同一品种池；α 为年化 %。2010-07 至 2026-08。'
            '括号内为 Newey-West t 值；\*、\*\*、\*\*\* 分别表示 10%、5%、1% 显著。来源：Asness, Moskowitz & Pedersen，Journal of Finance，2013 的复现（仅商品部分）。')
    return table(cols, rows, foot, note)


# ---------------------------------------------------------------- 表 6：基差动量的混合回归（Boons-Prado Table II）
def bm_pooled():
    d = pl.read_parquet(f'{R}/basis_momentum/output/t2_paper.parquet').filter((pl.col('收益') == '近月') & pl.col('模型').is_in(['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)']))
    d = d.sort('模型')
    g = lambda r, k: cell(r[f'λ_{k}'], r[f't_{k}']) if r[f'λ_{k}'] is not None else ''   # noqa: E731
    rs = list(d.iter_rows(named=True))
    rows = [('基差动量', [g(r, 'BM') for r in rs]), ('基差', [g(r, 'B') for r in rs]), ('动量（近月 12 个月）', [g(r, 'M1') for r in rs])]
    foot = [('时间固定效应', ['是' if r['时间FE'] else '否' for r in rs]), ('品种固定效应', ['是' if r['品种FE'] else '否' for r in rs]),
            ('观测数（品种-月）', [str(r['观测数']) for r in rs]), ('R²', [f"{r['R2']:.3f}" for r in rs])]
    note = ('被解释变量为近月合约下月收益（%），解释变量为上月末特征的原始值（收益之差、基差，未标准化），同论文 Table II。括号内为按月聚类的 t 值；'
            '\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。来源：Boons & Prado，Journal of Finance，2019 的复现（论文口径，合约持仓 ≥ 100 手）。')
    return table(['近月收益'] * len(rs), rows, foot, note)


# ---------------------------------------------------------------- 表 7：偏度的 Fama-MacBeth（Fernandez-Perez et al. Table VI）
def skew_fm():
    d = pl.read_parquet(f'{R}/commodity_skewness/output/t6_paper.parquet')
    rs = list(d.iter_rows(named=True))
    g = lambda r, k: cell(r[f'λ_{k}'], r[f't_{k}']) if r[f'λ_{k}'] is not None else ''   # noqa: E731
    rows = [('Constant', [g(r, '常数') for r in rs]), ('等权 EW', [g(r, 'EW') for r in rs]), ('期限结构 TS', [g(r, 'TS') for r in rs]),
            ('动量 MOM', [g(r, 'MOM') for r in rs]), ('偏度 SKEW', [g(r, 'SKEW') for r in rs])]
    foot = [('样本数（月）', [str(r['月数']) for r in rs]), ('平均截面 R²', [f"{r['平均 R2%'] / 100:.3f}" for r in rs])]
    note = ('两步法 Fama-MacBeth：先估计各组合对因子的 β，再每月做截面回归，表中为风险价格 λ（年化 %）的时间序列平均。括号内为 Newey-West t 值；'
            '\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。来源：Fernandez-Perez et al.，Journal of Banking & Finance，2018 的复现。')
    return table([r['模型'] for r in rs], rows, foot, note)


# ---------------------------------------------------------------- 表 8：大户交易与后续收益（Kang-Rouwenhorst-Tang Table 3、6）
def krt():
    t3 = pl.read_parquet(f'{R}/kang_rouwenhorst_tang/output/t3.parquet')
    t6 = pl.read_parquet(f'{R}/kang_rouwenhorst_tang/output/t6.parquet').filter(pl.col('设定') == 'Q + 平滑 HP̄')
    rs = list(t3.iter_rows(named=True)) + list(t6.iter_rows(named=True))
    cols = [f"{r['因变量']}收益" for r in rs]

    def g(r, k):
        return cell(r.get(f'b_{k}'), r.get(f't_{k}')) if r.get(f'b_{k}') is not None else ''
    rows = [('大户净买入 Q', [g(r, 'Q') for r in rs]), ('大户长期净空头 HP̄', [g(r, 'HPbar') for r in rs]), ('基差', [g(r, 'B') for r in rs]),
            ('S × 特质波动', [g(r, 'SIV') for r in rs]), ('当周收益 R_t', [g(r, 'R') for r in rs]), ('Constant', [g(r, '常数') for r in rs])]
    foot = [('样本数（周）', [str(r['周数']) for r in rs]), ('平均截面 R²', [f"{r['R2%'] / 100:.3f}" for r in rs])]
    note = ('每周对下一周（或第二周）收益（%）做截面回归，再取时间序列平均（Fama-MacBeth）。Q = 前 20 名会员净多头的周变化 ÷ 上周总持仓；HP̄ = 前 20 名净空头 52 周平均 ÷ 当周总持仓。'
            '括号内为 Newey-West t 值；\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。来源：Kang, Rouwenhorst & Tang，Journal of Finance，2020 的复现。')
    return table(cols, rows, foot, note)


# ---------------------------------------------------------------- 表 9：各信号的多空收益（Fan-Zhang Table 4，第 1 近月）
def fz_sorts():
    d = pl.read_parquet(f'{R}/fan_zhang/output/table4.parquet').filter(pl.col('设定').str.starts_with('论文'))
    order = ['期限结构', '动量', '价值', '偏度', '波动率', '套保压力', '持仓变化', '流动性', '通胀', '汇率']
    specs = [(1, '2004-02~2017-05（论文样本期）'), (1, '2017-06~2026-08（样本外）'), (2, '2004-02~2017-05（论文样本期）'), (2, '2017-06~2026-08（样本外）')]
    rows = []
    for s in order:
        cs = []
        for m, p in specs:
            q = d.filter((pl.col('策略') == s) & (pl.col('时期') == p) & (pl.col('m') == m))
            cs.append(cell(q['年化%'][0], q['t'][0], '{:.2f}') if q.height and q['月数'][0] > 0 and q['年化%'][0] == q['年化%'][0] else '')
        rows.append((s, cs))
    note = ('每格为四分组多空组合的年化收益（%），括号内为 Newey-West t 值；\*、\*\*、\*\*\* 分别表示 10%、5%、1% 显著。'
            '汇率数据 2012 年起，论文样本期内只有 27 个月。来源：Fan & Zhang，Journal of Futures Markets，2020 的复现。')
    return table(['第 1 近月<br>2004–2017', '第 1 近月<br>2017–2026', '第 2 近月<br>2004–2017', '第 2 近月<br>2017–2026'], rows, [], note)


# ---------------------------------------------------------------- 表 10：按各变量分组的现货溢价与期限溢价（Szymanowska Table II、III）
def szy_sorts():
    d = pl.read_parquet(f'{R}/szymanowska/output/sorts.parquet').filter(pl.col('样本') == '全样本')
    names = {'BASIS': '基差', 'MOM': '动量', 'CV': '变异系数', 'INF': '通胀 beta', 'FX': '美元 beta', 'HP': '套保压力（前 20 名净空）', 'OI': '总持仓', 'LIQ': '流动性'}
    specs = [('SR', 1), ('SR', 4), ('EH', 2), ('EH', 4)]
    rows = []
    for v, lab in names.items():
        cs = []
        for a, n in specs:
            q = d.filter((pl.col('变量') == v) & (pl.col('收益') == a) & (pl.col('n') == n))
            cs.append(cell(q['P4-P1'][0], q['P4-P1_t'][0], '{:.2f}') if q.height else '')
        rows.append((lab, cs))
    note = ('每格为最高组减最低组（P4 − P1）的年化收益（%），括号内为 Newey-West t 值；双月频率，2005–2026。Short Roll = 滚动持有最近期合约（现货溢价）；'
            'Excess Holding = 持有 n 期合约到期 − Short Roll（期限溢价）。\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。来源：Szymanowska et al.，Journal of Finance，2014 的复现。')
    return table(['现货溢价<br>n = 1（2 个月）', '现货溢价<br>n = 4（8 个月）', '期限溢价<br>n = 2', '期限溢价<br>n = 4'], rows, [], note)


if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True)
    for name, fn in [('xs_bgr_spanning', bgr_spanning), ('xs_roll_split', roll_split), ('xs_amp_carry', amp_carry), ('xs_bm_pooled', bm_pooled),
                     ('xs_skew_fm', skew_fm), ('xs_krt', krt), ('xs_fz_sorts', fz_sorts), ('xs_szy_sorts', szy_sorts)]:
        s = fn()
        with open(os.path.join(TAB, f'{name}.md'), 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(f'==== {name}\n{s}')
