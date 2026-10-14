"""全貌文档的配图（读取 output/，写入 figures/）"""
import os

import matplotlib
import numpy as np
import polars as pl

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

VERS = ['EW', 'OIW', 'FIX', 'FIX2', 'NHCI']
LAB = {'EW': '等权', 'OIW': '持仓加权', 'FIX': '固定池一', 'FIX2': '固定池二', 'NHCI': '南华指数'}
PERIODS = [('2010–2026', 2010, 2026), ('2005–2009', 2005, 2009), ('2010–2015', 2010, 2015), ('2016–2020', 2016, 2020), ('2021–2026', 2021, 2026)]


def corr(g):
    C = np.full((len(VERS), len(VERS)), np.nan)
    for i, a in enumerate(VERS):
        for j, b in enumerate(VERS):
            x = g.select(pl.col(a).alias('a'), pl.col(b).alias('b')).drop_nulls().to_numpy()
            if len(x) >= 24:
                C[i, j] = np.corrcoef(x.T)[0, 1]
    return C


def market_versions():
    f = pl.read_parquet(os.path.join(HERE, 'output', 'factors.parquet'))
    fig, axes = plt.subplots(1, len(PERIODS), figsize=(4.2 * len(PERIODS), 4.4))
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('#eeeeee')
    for ax, (lab, lo, hi) in zip(axes, PERIODS):
        g = f.filter((pl.col('m').dt.year() >= lo) & (pl.col('m').dt.year() <= hi))
        C = corr(g)
        ax.imshow(np.ma.masked_invalid(C), cmap=cmap, vmin=0.6, vmax=1.0)
        for i in range(len(VERS)):
            for j in range(len(VERS)):
                ax.text(j, i, '—' if np.isnan(C[i, j]) else f'{C[i, j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if (not np.isnan(C[i, j]) and C[i, j] > 0.9) else 'black')
        ax.set_xticks(range(len(VERS)), [LAB[v] for v in VERS], rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(VERS)), [LAB[v] for v in VERS] if ax is axes[0] else [''] * len(VERS), fontsize=9)
        ax.set_title(f'{lab}（{g.height} 个月）', fontsize=11)
    fig.suptitle('市场因子各版本的月收益相关（固定池二 2010 年起，南华指数 2006 年起）', fontsize=12, y=1.04)
    fig.tight_layout()
    out = os.path.join(FIG, 'market_versions_corr.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    market_versions()
