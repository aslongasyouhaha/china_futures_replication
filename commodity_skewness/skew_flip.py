"""偏度效应在 2018 年前后翻转的原因检验。
检验 1（是不是动量）：(a) 偏度与过去 12 个月收益的截面秩相关；(b) 动量两组内各自按偏度做多空；(c) 偏度对动量截面回归取残差再排序；
                     (d) Fama-MacBeth：下月收益对偏度、动量（截面标准化）回归。
检验 2（一次性翻转还是衰减）：逐年多空收益、36 个月滚动均值。
口径：B 交割月 ≥ 当前月 + 2、无持仓门槛（对应 Li, Sun & Zhou）；C 同 B 加持仓 ≥ 100 手（本库主口径）。
多空 = 偏度最低四分位 − 最高四分位（满额）；负值表示右偏品种跑赢。输出 output/flip_*.parquet
"""
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
BM = os.path.join(HERE, '..', 'basis_momentum', 'output')
sys.path.insert(0, HERE)
from lsz_check import rule_panel, nw_t  # noqa: E402

CUT = pl.date(2018, 7, 1)
PER = {'2005 ~ 2018-06': pl.col('m') < CUT, '2018-07 ~ 2026': pl.col('m') >= CUT, '全样本': pl.lit(True)}


def add_mom(x):
    c = pl.read_parquet(os.path.join(BM, 'curve_paper.parquet')).filter(pl.col('n') == 1).select('product', 'm', 'ret').sort(['product', 'm'])
    h = c.select('product', pl.col('m').dt.offset_by('1mo').alias('m'), (pl.col('ret') + 1).log().alias('lr_in'))
    h = h.sort(['product', 'm']).with_columns(pl.col('lr_in').rolling_sum(12).over('product').alias('MOM'),
                                              (pl.col('m').shift(11).over('product') == pl.col('m').dt.offset_by('-11mo')).alias('ok'))
    return x.join(h.filter(pl.col('ok')).select('product', 'm', 'MOM'), on=['product', 'm'], how='left')


def q_ls(g, col):
    o = g.sort([col, 'product'])
    k = max(1, g.height // 4)
    return o.head(k)['ret'].mean() - o.tail(k)['ret'].mean()


def monthly(x):
    rows = []
    for (m,), g in x.drop_nulls(['SK', 'MOM', 'ret']).group_by(['m'], maintain_order=True):
        if g.height < 8:
            continue
        sk, mo, r = g['SK'].to_numpy(), g['MOM'].to_numpy(), g['ret'].to_numpy()
        rk = lambda a: np.argsort(np.argsort(a)).astype(float)   # noqa: E731
        rho = np.corrcoef(rk(sk), rk(mo))[0, 1]
        # (b) 动量两组内
        med = np.median(mo)
        within = []
        for sel in (mo > med, mo <= med):
            if sel.sum() >= 4:
                within.append(q_ls(g.filter(pl.Series(sel)), 'SK'))
        # (c) 残差
        X = np.column_stack([np.ones(len(mo)), (mo - mo.mean()) / mo.std()])
        res = sk - X @ np.linalg.lstsq(X, sk, rcond=None)[0]
        gr = g.with_columns(pl.Series('SKr', res))
        # (d) FMB
        z = lambda a: (a - a.mean()) / a.std()   # noqa: E731
        Z = np.column_stack([np.ones(len(r)), z(sk), z(mo)])
        b = np.linalg.lstsq(Z, r, rcond=None)[0]
        b1 = np.linalg.lstsq(Z[:, :2], r, rcond=None)[0]
        rows.append({'m': m, 'N': g.height, 'rho': rho, 'raw': q_ls(g, 'SK'), 'mom_ls': q_ls(g, 'MOM') * -1,
                     'within': np.mean(within) if within else np.nan, 'resid': q_ls(gr, 'SKr'),
                     'fm_sk': b[1], 'fm_mom': b[2], 'fm_sk_alone': b1[1]})
    return pl.DataFrame(rows).sort('m')


def summarize(mm, lab):
    out = []
    for per, cond in PER.items():
        q = mm.filter(cond)
        row = {'口径': lab, '时期': per, '月数': q.height, '偏度-动量秩相关': q['rho'].mean()}
        for c, nm in (('raw', '原始偏度多空%'), ('within', '动量组内偏度多空%'), ('resid', '残差偏度多空%')):
            y = q[c].to_numpy()
            row[nm] = np.nanmean(y) * 1200
            row[f't_{c}'] = nw_t(y)
        for c, nm in (('fm_sk_alone', 'FM 偏度（单独）'), ('fm_sk', 'FM 偏度（控制动量）'), ('fm_mom', 'FM 动量')):
            y = q[c].to_numpy()
            row[nm] = np.nanmean(y) * 1200
            row[f't_{c}'] = nw_t(y)
        y = q['mom_ls'].to_numpy()
        row['动量多空%（高−低）'] = np.nanmean(y) * 1200
        row['t_mom'] = nw_t(y)
        out.append(row)
    return out


def main():
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
    B = add_mom(rule_panel()).filter(~((pl.col('product') == 'FB') & (pl.col('m') >= pl.date(2019, 9, 1))))
    C = B.filter((pl.col('oi') >= 100) & (pl.col('vol') > 0))
    rows, yearly = [], []
    for lab, x in (('B 无持仓门槛', B), ('C 持仓 ≥ 100 手', C)):
        mm = monthly(x)
        mm.write_parquet(os.path.join(OUT, f"flip_monthly_{lab[0]}.parquet"))
        rows += summarize(mm, lab)
        y = mm.group_by(pl.col('m').dt.year().alias('年')).agg((pl.col('raw').mean() * 1200).alias('原始'), (pl.col('resid').mean() * 1200).alias('残差'),
                                                             (pl.col('mom_ls').mean() * 1200).alias('动量'), pl.col('rho').mean().alias('秩相关'), pl.len().alias('月数')).sort('年')
        yearly.append(y.with_columns(pl.lit(lab).alias('口径')))
    t = pl.DataFrame(rows)
    t.write_parquet(os.path.join(OUT, 'flip_summary.parquet'))
    yy = pl.concat(yearly)
    yy.write_parquet(os.path.join(OUT, 'flip_yearly.parquet'))
    show = t.select('口径', '时期', '月数', '偏度-动量秩相关', '原始偏度多空%', 't_raw', '动量组内偏度多空%', 't_within', '残差偏度多空%', 't_resid',
                    'FM 偏度（单独）', 't_fm_sk_alone', 'FM 偏度（控制动量）', 't_fm_sk', '动量多空%（高−低）', 't_mom')
    print(show.with_columns(pl.col(pl.Float64).round(2)))
    print(yy.filter(pl.col('口径').str.starts_with('C')).with_columns(pl.col(pl.Float64).round(1)))


if __name__ == '__main__':
    main()
