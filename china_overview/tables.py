"""全貌文档的回归表（论文格式：列 = 被解释变量，行 = 解释变量，系数下方括号内为 Newey-West t 值）。
读取 market.py 的输出与宏观数据，结果写入 tables/*.md。"""
import os

import numpy as np
import polars as pl

import market as M

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, 'tables')


def reg(y, X):
    """OLS + NW t；返回系数、t、N、R²、调整 R²。"""
    y, X = np.asarray(y, float), np.asarray(X, float).reshape(len(y), -1)
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    b, t, r2, T = M.ols_nw(y[ok], X[ok])
    k = X.shape[1]
    return b, t, T, r2, 1 - (1 - r2) * (T - 1) / (T - k - 1)


def star(t):
    a = abs(t)
    return '***' if a >= 2.576 else '**' if a >= 1.96 else '*' if a >= 1.645 else ''


def md_table(cols, rows, results, note):
    """cols: [(列名, 结果键)]；rows: [(行名, 变量名)]；results[键] = dict(var -> (b, t), N, R2, aR2)。"""
    head = '| | ' + ' | '.join(f'({i + 1})<br>{c}' for i, (c, _) in enumerate(cols)) + ' |'
    sep = '|---|' + '---|' * len(cols)
    lines = [head, sep]
    for lab, v in rows:
        cells = []
        for _, key in cols:
            r = results[key]
            if v in r['coef']:
                b, t = r['coef'][v]
                cells.append(f'{b:.3f}{star(t)}<br>({t:.2f})')
            else:
                cells.append('')
        lines.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    for lab, k, fmt in [('Constant', None, None), ('样本数（月）', 'N', '{:d}'), ('R²', 'R2', '{:.3f}'), ('调整 R²', 'aR2', '{:.3f}')]:
        if k is None:
            cells = []
            for _, key in cols:
                b, t = results[key]['coef']['const']
                cells.append(f'{b:.3f}{star(t)}<br>({t:.2f})')
        else:
            cells = [fmt.format(results[key][k]) for _, key in cols]
        lines.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n\n' + note + '\n'


def run(df, ycol, xs):
    b, t, T, r2, ar2 = reg(df[ycol].to_numpy(), df.select(xs).to_numpy())
    coef = {'const': (b[0], t[0])}
    coef.update({x: (b[i + 1], t[i + 1]) for i, x in enumerate(xs)})
    return {'coef': coef, 'N': T, 'R2': r2, 'aR2': ar2}


def macro_table():
    f = pl.read_parquet(os.path.join(HERE, 'output', 'factors.parquet'))
    mac = M.macro()
    z = f.join(mac, on='m', how='left')
    # 单位：收益为月度 %；ΔPMI 为点；ΔPPI、ΔM1 为同比的百分点变化
    pct = ['EW', 'OIW', '黑色', '有色贵金属', '能源化工', '农产品', 'HS300', 'USDCNH', 'OIL']
    z = z.with_columns([pl.col(c) * 100 for c in pct])
    ys = [('等权', 'EW'), ('持仓加权', 'OIW'), ('黑色', '黑色'), ('有色贵金属', '有色贵金属'), ('能源化工', '能源化工'), ('农产品', '农产品')]
    rows = [('ΔPMI（点）', 'dPMI'), ('ΔPPI 同比（百分点）', 'dPPI'), ('ΔM1 同比（百分点）', 'dM1'), ('沪深 300 收益（%）', 'HS300'),
            ('USDCNH 变化（%）', 'USDCNH'), ('原油 SC 收益（%）', 'OIL')]
    out = []
    xa = ['dPMI', 'dPPI', 'dM1', 'HS300']
    res = {y: run(z, y, xa) for _, y in ys}
    out.append('**Panel A：2005–2026（不含汇率、原油，二者数据较短）**\n\n' + md_table(ys, rows[:4], res, ''))
    zb = z.filter(pl.col('OIL').is_not_null())
    xb = xa + ['USDCNH', 'OIL']
    res = {y: run(zb, y, xb) for _, y in ys}
    out.append('**Panel B：2018-04 起（加入汇率与原油）**\n\n' + md_table(ys, rows, res, ''))
    note = ('被解释变量为各版本市场因子或板块等权指数的月收益（%）；解释变量为同月数据。括号内为 Newey-West t 值；'
            '\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。宏观数据为统计月份（实际公布在次月），这里只看同期关系，不用于预测。')
    return '\n'.join(out) + '\n' + note + '\n'


def predict_table():
    f = pl.read_parquet(os.path.join(HERE, 'output', 'factors.parquet'))
    mac = M.macro()
    state = pl.read_parquet(os.path.join(HERE, 'output', 'state.parquet'))
    rows = [('上月市场收益（%）', 'TS1'), ('过去 12 个月市场收益（%）', 'TS12'), ('整体基差（年化 %）', 'BASIS'), ('整体持仓金额 12 个月增长（%）', 'OIG'),
            ('ΔPMI（点，滞后一月）', 'dPMI'), ('ΔPPI 同比（百分点，滞后一月）', 'dPPI'), ('ΔM1 同比（百分点，滞后一月）', 'dM1'), ('沪深 300 上月收益（%）', 'HS300')]
    xs = [r[1] for r in rows]
    ys = [('等权', 'EW'), ('持仓加权', 'OIW'), ('固定池一', 'FIX'), ('固定池二', 'FIX2')]
    res_m, res_u = {}, {}
    for _, y in ys:
        _, d = M.timing(f, mac, state, y)
        d = d.with_columns([pl.col(c) * 100 for c in ('y_next', 'TS1', 'TS12', 'BASIS', 'OIG', 'HS300')])
        res_m[y] = run(d, 'y_next', xs)
        res_u[y] = {'coef': {}, 'r2': {}}
        for x in xs:
            r = run(d, 'y_next', [x])
            res_u[y]['coef'][x] = r['coef'][x]
            res_u[y]['r2'][x] = r['R2']
    out = ['**Panel A：所有预测变量一起回归**\n\n' + md_table(ys, rows, res_m, '')]
    # Panel B：逐个单变量回归，每格 系数 / (t) / [R²]
    head = '| | ' + ' | '.join(f'({i + 1})<br>{c}' for i, (c, _) in enumerate(ys)) + ' |'
    lines = [head, '|---|' + '---|' * len(ys)]
    for lab, x in rows:
        cells = []
        for _, y in ys:
            b, t = res_u[y]['coef'][x]
            cells.append(f'{b:.3f}{star(t)}<br>({t:.2f})<br>[{res_u[y]["r2"][x] * 100:.2f}%]')
        lines.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    out.append('**Panel B：逐个单变量回归**（每格为一次单独回归：系数、(t 值)、[R²]）\n\n' + '\n'.join(lines) + '\n')
    note = ('被解释变量为下月市场因子收益（%）；解释变量均为月末已知（宏观数据滞后一月，因次月才公布）。固定池二从 2010 年起，其余从 2005 年起。'
            '括号内为 Newey-West t 值；\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。')
    return '\n'.join(out) + '\n' + note + '\n'


if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True)
    for name, fn in [('macro', macro_table), ('predict', predict_table)]:
        s = fn()
        with open(os.path.join(TAB, f'{name}.md'), 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(s)
