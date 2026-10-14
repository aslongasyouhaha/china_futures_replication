"""日内分析（仿论文第 4.1 节对 WTI 2020-04-20 的做法）：5 分钟数据，逐时点反推隐含失衡 ε。

  F_obs：近月 5 分钟收盘；F2：下一合约同一时点收盘；F_lic = λ·F2
  λ 两个版本：A = 压力月开始前一刻（上月最后一个交易日收盘）的比值（与每日扫描一致）；B = 所画第一天开盘的比值（论文做法）
  σ_M：λ 所在时点之前一个月内 ln(F_obs/F2) 的 5 分钟变化（相邻 5 分钟、同一交易日内）的标准差，按 243 × 每日 5 分钟数年化
  τ：到压力月最后一个交易日收盘的剩余 5 分钟数（含当前）÷（243 × 每日 5 分钟数）
"""
import os

import numpy as np
import polars as pl

import model as M
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
DATA = RAW_DATA
EVENTS = [('INE', 'LU', 'LU2410', 'LU2411', '2024-09'), ('DCE', 'A', 'A1801', 'A1805', '2017-12'), ('CZCE', 'SA', 'SA2005', 'SA2009', '2020-04'),
          ('DCE', 'JM', 'JM2201', 'JM2203', '2021-12')]   # 最后一个为反例：原有价差为贴水而近月被压低，期权虚值，ε 无区分度
SHOW_DAYS = 5


def bars(ex, p, s1, s2):
    m = pl.read_parquet(f'{DATA}/{ex}/{p}/minute_contracts.parquet', columns=['symbol', 'datetime', 'trading_day', 'close'])
    m = m.filter(pl.col('symbol').is_in([s1, s2]) & (pl.col('close') > 0)).with_columns(pl.col('datetime').dt.truncate('5m').alias('b'))
    b = m.sort('datetime').group_by(['symbol', 'trading_day', 'b']).agg(pl.col('close').last())
    w = b.pivot(index=['trading_day', 'b'], columns='symbol', values='close').drop_nulls([s1, s2]).sort('b')
    return w.rename({s1: 'F', s2: 'F2'})


def run(ex, p, s1, s2, ym):
    w = bars(ex, p, s1, s2)
    y, mo = map(int, ym.split('-'))
    win = w.filter((pl.col('trading_day').dt.year() == y) & (pl.col('trading_day').dt.month() == mo))
    days = win['trading_day'].unique().sort().to_list()
    before = w.filter(pl.col('trading_day') < days[0])
    per_day = int(win.group_by('trading_day').len()['len'].median())
    ann = 243 * per_day
    show = win.filter(pl.col('trading_day') >= days[-SHOW_DAYS]).with_row_index('k')
    n = show.height
    out = {}
    for lab, anchor in (('A 压力月开始', before.tail(1)), ('B 首日开盘（论文）', show.head(1))):
        lam = float(anchor['F'][0] / anchor['F2'][0])
        t_anchor = anchor['b'][0]
        hist = w.filter((pl.col('b') <= t_anchor) & (pl.col('b') > t_anchor - pl.duration(days=31)) if False else
                        (pl.col('b') <= t_anchor)).tail(per_day * 21)
        hist = hist.with_columns((pl.col('F') / pl.col('F2')).log().alias('x'))
        dx = hist.with_columns(pl.col('x').diff().over('trading_day').alias('dx'))['dx'].drop_nulls().to_numpy()
        sig = float(np.std(dx) * np.sqrt(ann))
        rows = []
        for k, t, fo, f2 in show.select('k', 'b', 'F', 'F2').iter_rows():
            tau = (n - k) / ann
            fl = lam * f2
            eps, case = M.implied_eps(fo, f2, fl, tau, sig)
            rows.append({'事件': s1, '版本': lab, 'time': t, 'F': fo, 'F2': f2, 'F_lic': fl, 'distortion': np.log(fo / fl), 'eps': eps, 'case': case})
        r = pl.DataFrame(rows)
        out[lab] = (r, lam, sig)
    return out, per_day


def main():
    os.makedirs(OUT, exist_ok=True)
    allr = []
    for ex, p, s1, s2, ym in EVENTS:
        res, per_day = run(ex, p, s1, s2, ym)
        print(f'\n== {s1} 对 {s2}，压力月 {ym}，展示最后 {SHOW_DAYS} 个交易日（每日 {per_day} 根 5 分钟 K 线）==')
        for lab, (r, lam, sig) in res.items():
            pk = r.sort('eps', descending=True).row(0, named=True)
            mx = r.with_columns(pl.col('distortion').abs().alias('a')).sort('a', descending=True).row(0, named=True)
            last = r.row(-1, named=True)
            print(f"  基准 {lab}：λ={lam:.4f}，σ_M={sig:.2f}；最大偏离 {mx['distortion']*100:+.1f}%（{mx['time']}，ε={mx['eps']:.3f}，Case {mx['case']}）；"
                  f"ε 最高 {pk['eps']:.3f}（{pk['time']}）；最后一根 偏离 {last['distortion']*100:+.1f}%、ε={last['eps']:.3f}")
            daily = (r.with_columns(pl.col('time').dt.date().alias('d')).group_by('d')
                      .agg((pl.col('distortion').abs().max() * 100).round(1).alias('当日最大偏离%'), pl.col('eps').max().round(3).alias('当日最高ε'),
                           pl.col('eps').median().round(3).alias('当日中位ε')).sort('d'))
            print('   ', daily.rows())
            allr.append(r)
    pl.concat(allr).write_parquet(os.path.join(OUT, 'intraday.parquet'))


if __name__ == '__main__':
    main()
