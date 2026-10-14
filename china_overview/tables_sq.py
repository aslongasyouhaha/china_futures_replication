"""全貌文档第五节（交割前挤仓）的表。数字取自 commodity_squeeze/output/；t 值用该复现的 nw_t。"""
import json
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(HERE, 'tables')
sys.path.insert(0, HERE)
sys.path.insert(0, f'{R}/commodity_squeeze')
from tables_xs import cell, table  # noqa: E402
from exit_test import nw_t  # noqa: E402

STARS = '\\*、\\*\\*、\\*\\*\\* 分别表示 10%、5%、1% 显著。'
SRC = '来源：Zimbidis & Sircar，arXiv 工作论文，2026 的复现。'


def model():
    m = json.load(open(f'{R}/commodity_squeeze/output/model_results.json', encoding='utf-8'))
    rows = [('无反馈时的展期期权价值（Figure 1）', ['约 3.10', f"{m['fig1']['0'][0]:.2f}"]),
            ('ε = 0.95 时的期权价值 V+ / V−（Figure 1）', ['11.25 / 8.06', f"{m['fig1']['0.95'][0]:.2f} / {m['fig1']['0.95'][1]:.2f}"]),
            ('WTI 2020-04-20 隐含失衡 ε', ['0.87', f"{m['wti']:.3f}"]),
            ('LME 镍 2022-03-08 隐含失衡 ε', ['0.9346', f"{m['nickel']['T−t = 1 天（1/365 年）']:.4f}（剩余 1 天）；{m['nickel']['表中所写 T−t = 1/4 年']:.4f}（按表中所写 1/4 年）"]),
            ('蒙特卡洛：近月跌破零的比例（μ = 0）', ['61.0%', f"{m['mc']['12|0.0'] * 100:.1f}%（每天 12 步）；{m['mc']['2|0.0'] * 100:.1f}%（每天 2 步）"])]
    note = '镍的结果说明论文 Table 2 所写的剩余期限与实际计算不一致；蒙特卡洛比例随时间步长变细而上升。' + SRC
    return table(['论文', '复现'], rows, [], note)


def stress():
    d = pl.read_parquet(f'{R}/commodity_squeeze/output/exit_vs_placebo.parquet')
    rows = []
    for lab, c in (('压力月：个人须在月末前平仓', '压力月'), ('对照月：再往前一个月', '对照月')):
        e = d[f'{c}_end'].drop_nulls().to_numpy()
        rows.append((lab, [f"{d[f'{c}_max_abs'].median() * 100:.2f}%", f"{d[f'{c}_max_abs'].quantile(0.99) * 100:.1f}%",
                           str(int((d[f'{c}_max_abs'] > 0.10).sum())), cell(e.mean() * 100, nw_t(e), '{:.2f}'), f'{(e < 0).mean() * 100:.0f}%',
                           f"{d[f'{c}_oi_change'].median() * 100:.0f}%"]))
    diff = (d['压力月_max_abs'] - d['对照月_max_abs']).to_numpy()
    note = (f'共 {d.height} 次合约到期，每个合约在两个月份各算一次。偏离 = 近月价格相对"窗口开始前近月/次月比值不变"所推算价格的偏离；'
            f'压力月偏离大于对照月的占 {(diff > 0).mean() * 100:.0f}%。月末偏离一列括号内为 Newey-West t 值；' + STARS + SRC)
    return table(['最大偏离<br>中位数', '最大偏离<br>99% 分位', '偏离 > 10%<br>的次数', '月末偏离<br>平均 %', '月末被压低<br>的比例', '近月持仓变化<br>中位数'], rows, [], note)


def by_product():
    d = pl.read_parquet(f'{R}/commodity_squeeze/output/exit_by_product.parquet').filter(pl.col('n') >= 30)
    d = d.sort('压力月多出的偏离%（中位数）', descending=True)
    names = {'LU': '低硫燃料油', 'LH': '生猪', 'JM': '焦煤', 'J': '焦炭', 'FU': '燃料油', 'AP': '苹果', 'SC': '原油', 'FG': '玻璃',
             'PB': '铅', 'AL': '铝', 'BR': '丁二烯橡胶', 'SN': '锡', 'CU': '铜', 'ZN': '锌', 'NI': '镍', 'AU': '黄金'}
    pick = pl.concat([d.head(8), d.tail(4)])
    rows = [(f"{names.get(r['product'], r['product'])}（{r['product']}）", [str(r['n']), f"{r['压力月多出的偏离%（中位数）']:.2f}%", f"{r['压力月偏离%']:.2f}%"])
            for r in pick.iter_rows(named=True)]
    note = '到期次数 ≥ 30 的品种中，压力月比对照月多出的偏离最大的 8 个和最小的 4 个。' + SRC
    return table(['到期次数', '压力月比对照月<br>多出的偏离（中位数）', '压力月偏离<br>（中位数）'], rows, [], note)


if __name__ == '__main__':
    for name, fn in [('sq_model', model), ('sq_stress', stress), ('sq_product', by_product)]:
        s = fn()
        with open(os.path.join(TAB, f'{name}.md'), 'w', encoding='utf-8') as fh:
            fh.write(s)
        print(f'==== {name}\n{s}')
