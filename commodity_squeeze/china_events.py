"""把论文第 4 节的反推方法用于中国商品期货的每一次合约到期。

中国对应：个人客户须在交割月前平仓（不能进入交割），相当于论文的“不能交割的一方”。
  压力窗口：合约交割月的前一个月（个人强制退出的月份），T' = 该月最后一个交易日
  近月 F_obs：该合约结算价；F2：下一个挂牌合约的结算价（论文用次月合约，视为不受挤压）
  基准 F_lic = λ·F2，λ = 窗口开始前一个交易日的 F_obs/F2（同论文：保持初始比值）
  σ_M：窗口开始前 60 个交易日 ln(F_obs/F2) 日变化的标准差 × √243；剩余期限 τ = (到 T' 的剩余交易日 + 1)/243；D = 1
  每天按式 (57) 反推隐含失衡 ε 与方向（Case 1 被迫卖出压低 / Case 2 被迫买入抬高）
过滤：窗口开始时近月持仓 ≥ 500 手、下一合约持仓 ≥ 100 手；排除中金所、集运指数（现金交割）、月均价合约。
"""
import glob
import os
import sys

import numpy as np
import polars as pl

import model as M
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
DATA = RAW_DATA
sys.path.insert(0, os.path.join(HERE, '..', 'carry'))
import carry as C  # noqa: E402

EXCLUDE = {'EC', 'LF', 'PPF', 'VF'}
ANN = 243
MIN_OI_NEAR, MIN_OI_NEXT, SIG_WIN = 500, 100, 60


def load(f):
    d = pl.read_parquet(f, columns=['symbol', 'trading_day', 'close', 'settle', 'open_interest', 'volume'])
    d = d.with_columns(pl.col('settle').fill_null(pl.col('close')).alias('px')).filter(pl.col('px') > 0)
    return d.with_columns(pl.col('symbol').map_elements(C.month_index, return_dtype=pl.Int64).alias('cm'),
                          (pl.col('trading_day').dt.year() * 12 + pl.col('trading_day').dt.month() - 1).alias('now'))


def scan_product(p, d):
    days = d['trading_day'].unique().sort().to_list()
    pos = {x: i for i, x in enumerate(days)}
    px = {(s, t): (v, oi) for s, t, v, oi in d.select('symbol', 'trading_day', 'px', 'open_interest').iter_rows()}
    by_sym = {k[0]: v.sort('trading_day') for k, v in d.partition_by(['symbol'], as_dict=True).items()}
    cms = d.select('symbol', 'cm').unique().sort('cm').rows()
    daily, events = [], []
    for k, (sym, cm) in enumerate(cms):
        g = by_sym[sym]
        win = g.filter(pl.col('now') == cm - 1)                        # 交割月前一个月
        if win.height < 5:
            continue
        w_days = win['trading_day'].to_list()
        i0 = pos[w_days[0]] - 1
        if i0 < SIG_WIN:
            continue
        t0 = days[i0]
        nxt = [s for s, c in cms[k + 1:] if (s, t0) in px and px[(s, t0)][1] >= MIN_OI_NEXT]
        if not nxt or (sym, t0) not in px or px[(sym, t0)][1] < MIN_OI_NEAR:
            continue
        s2 = nxt[0]
        lam = px[(sym, t0)][0] / px[(s2, t0)][0]
        hist = [np.log(px[(sym, days[i])][0] / px[(s2, days[i])][0]) for i in range(i0 - SIG_WIN, i0 + 1)
                if (sym, days[i]) in px and (s2, days[i]) in px]
        if len(hist) < 40:
            continue
        sig = max(np.std(np.diff(hist)) * np.sqrt(ANN), 0.05)
        oi_peak = g['open_interest'].max()
        rows = []
        for j, t in enumerate(w_days):
            if (s2, t) not in px:
                continue
            fo, oi = px[(sym, t)]
            f2 = px[(s2, t)][0]
            fl = lam * f2
            tau = (len(w_days) - 1 - j + 1) / ANN
            eps, case = M.implied_eps(fo, f2, fl, tau, sig)
            rows.append({'product': p, 'symbol': sym, 'next': s2, 'trading_day': t, 'F_obs': fo, 'F2': f2, 'F_lic': fl,
                         'distortion': float(np.log(fo / fl)), 'eps': float(eps), 'case': case, 'oi': oi, 'tau_days': len(w_days) - j})
        if not rows:
            continue
        daily += rows
        r = pl.DataFrame(rows)
        pk = r.sort('eps', descending=True).row(0, named=True)
        last = r.row(-1, named=True)
        events.append({'product': p, 'symbol': sym, 'next': s2, 'cm': cm, 'window_start': w_days[0], 'T_prime': w_days[-1],
                       'lambda': lam, 'sigma_M': sig, 'oi_t0': px[(sym, t0)][1], 'oi_Tprime': last['oi'], 'oi_peak': oi_peak,
                       'resid_oi_ratio': last['oi'] / oi_peak if oi_peak else None,
                       'peak_eps': pk['eps'], 'peak_case': pk['case'], 'peak_day': pk['trading_day'], 'peak_distortion': pk['distortion'],
                       'end_eps': last['eps'], 'end_case': last['case'], 'end_distortion': last['distortion']})
    return daily, events


def main():
    os.makedirs(OUT, exist_ok=True)
    daily, events = [], []
    for f in sorted(glob.glob(f'{DATA}/*/*/daily_contracts.parquet')):
        ex = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = os.path.basename(os.path.dirname(f))
        if ex == 'CFFEX' or p in EXCLUDE:
            continue
        dd, ee = scan_product(p, load(f))
        daily += dd
        events += ee
        print(p, len(ee), flush=True)
    dl = pl.DataFrame(daily)
    ev = pl.DataFrame(events)
    dl.write_parquet(os.path.join(OUT, 'china_daily.parquet'))
    ev.write_parquet(os.path.join(OUT, 'china_events.parquet'))
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(14)
    print(f'\n合约到期事件 {ev.height} 个（{ev["product"].n_unique()} 个品种）')
    print('窗口内最大隐含失衡 ε 的分布：', [round(ev['peak_eps'].quantile(q), 3) for q in (0.5, 0.75, 0.9, 0.95, 0.99)])
    for th in (0.5, 0.7, 0.87):
        q = ev.filter(pl.col('peak_eps') >= th)
        print(f'  ε ≥ {th}：{q.height} 个（{q.height/ev.height:.1%}），其中被迫卖出压低（Case 1）{(q["peak_case"]==1).sum()}、被迫买入抬高（Case 2）{(q["peak_case"]==2).sum()}')
    print('\n== ε 最大的 25 次 ==')
    print(ev.sort('peak_eps', descending=True).head(25).select('product', 'symbol', 'next', 'peak_day', 'peak_case', 'peak_eps',
          (pl.col('peak_distortion') * 100).round(1).alias('偏离%'), (pl.col('sigma_M')).round(2), 'oi_t0', 'oi_Tprime'))
    print('\n== 偏离（近月相对基准）绝对值最大的 15 次 ==')
    print(ev.with_columns(pl.col('peak_distortion').abs().alias('a')).sort('a', descending=True).head(15)
            .select('product', 'symbol', 'next', 'peak_day', 'peak_case', 'peak_eps', (pl.col('peak_distortion') * 100).round(1).alias('偏离%')))


if __name__ == '__main__':
    main()
