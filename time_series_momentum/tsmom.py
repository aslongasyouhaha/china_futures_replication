"""时间序列动量（TSMOM）复现 —— Moskowitz, Ooi & Pedersen (2012, JFE)

信号:   sign(过去 k 个月累计收益)，月末收盘生成
仓位:   40% / 事前年化波动率（日收益 EWMA，δ=60/61，中心质量 60 天）
组合:   当月可交易品种等权，每月调仓，持有 1 个月
收益:   主力连续后复权收盘价 close_adj 的日收益（换月无虚假跳空）
"""
import glob
import os

import numpy as np
import polars as pl
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW_DATA, RESEARCH  # noqa: E402

DATA = RAW_DATA
OUT = f'{RESEARCH}/time_series_momentum/output'
TARGET_VOL = 0.40
DELTA = 60 / 61            # EWMA 衰减，中心质量 δ/(1-δ) = 60 天
ANN = 243                  # 中国期货每年约 243 个交易日（论文用 261）
MIN_VOL_LOTS = 1000        # 流动性门槛：过去 60 日主力日均成交量（手）
MIN_OBS = 60               # 波动率预热天数
EQUITY = {'IF', 'IH', 'IC', 'IM'}
BOND = {'T', 'TF', 'TS', 'TL'}


def asset_class(p):
    return 'equity' if p in EQUITY else 'bond' if p in BOND else 'commodity'


def load():
    """所有品种的主力连续日线 → 日收益、事前波动率、流动性。"""
    parts = []
    for f in glob.glob(f'{DATA}/*/*/daily_main.parquet'):
        p = os.path.basename(os.path.dirname(f))
        parts.append(pl.read_parquet(f, columns=['trading_day', 'close_adj', 'volume'])
                       .with_columns(pl.lit(p).alias('product')))
    df = pl.concat(parts).sort(['product', 'trading_day'])
    df = df.with_columns(
        (pl.col('close_adj') / pl.col('close_adj').shift(1).over('product') - 1).alias('ret'),
        pl.col('product').map_elements(asset_class, return_dtype=pl.Utf8).alias('cls'))
    df = df.with_columns(
        # 论文: σ² = 261·Σ(1-δ)δ^i (r - r̄)²；这里用指数加权方差，年化系数改为 243
        (pl.col('ret').ewm_var(alpha=1 - DELTA, adjust=True, min_periods=MIN_OBS).over('product') * ANN)
          .sqrt().alias('sigma'),
        pl.col('volume').rolling_mean(window_size=60, min_periods=40).over('product').alias('vol60'),
        pl.col('ret').is_not_null().cum_sum().over('product').alias('n_obs'))
    return df


_CACHE = {}


def load_cache():
    if 'df' not in _CACHE:
        _CACHE['df'] = load()
    return _CACHE['df']


def monthly_signals(df, k):
    """每个品种每月最后一个交易日的信号和仓位。"""
    me = (df.with_columns(pl.col('trading_day').dt.truncate('1mo').alias('month'))
            .group_by(['product', 'month']).agg(pl.all().sort_by('trading_day').last())
            .sort(['product', 'month']))
    me = me.with_columns((pl.col('close_adj') / pl.col('close_adj').shift(k).over('product') - 1).alias('past_ret'),
                         pl.col('month').shift(k).over('product').alias('month_k'))
    # 要求 k 个月前那个月末确实存在（月份连续），且流动性、波动率可用
    ok = ((pl.col('month_k') == pl.col('month').dt.offset_by(f'-{k}mo')) & pl.col('past_ret').is_not_null()
          & (pl.col('vol60') >= MIN_VOL_LOTS) & (pl.col('n_obs') >= MIN_OBS) & (pl.col('sigma') > 0))
    return me.with_columns(pl.when(ok).then(pl.col('past_ret').sign()).alias('signal')) \
             .select('product', 'trading_day', 'signal', 'sigma')


def backtest(df, k=12, lag=0, long_only=False):
    """返回每个品种每天的 (权重, 收益)。lag=0: 月末收盘成交；lag=1: 次日收盘成交。"""
    sig = monthly_signals(df, k)
    if long_only:
        sig = sig.with_columns(pl.when(pl.col('signal').is_not_null()).then(1.0).alias('signal'))
    # 每个月末都写一行仓位；不满足条件（流动性、历史长度）的月份仓位为 0
    sig = sig.with_columns((pl.col('signal') * TARGET_VOL / pl.col('sigma')).fill_null(0.0).alias('w_set'))
    d = (df.select('product', 'cls', 'trading_day', 'ret')
           .join(sig.select('product', 'trading_day', 'w_set'), on=['product', 'trading_day'], how='left')
           .sort(['product', 'trading_day']))
    # 月末设定的仓位一直持有到下一个月末
    d = d.with_columns(pl.col('w_set').forward_fill().over('product').alias('w_hold'))
    # 当天收盘设定的仓位，从下一天的收益开始生效；lag 再往后推
    d = d.with_columns(pl.col('w_hold').shift(1 + lag).over('product').alias('w'))
    return d.filter(pl.col('w').is_not_null())


def portfolio(d, by=None, cost=0.0):
    """等权组合日收益。cost: 每单位名义成交额的成本（如 0.0003 = 3bp）。"""
    d = d.with_columns((pl.col('w') - pl.col('w').shift(1).over('product')).abs().fill_null(pl.col('w').abs()).alias('dw'))
    keys = ['trading_day'] + ([by] if by else [])
    # 只把有持仓的品种计入 S_t（权重为 0 的月份视为不在池子里）
    act = d.filter(pl.col('w') != 0)
    port = act.group_by(keys).agg(((pl.col('w') * pl.col('ret').fill_null(0)).mean()).alias('gross'),
                                  (pl.col('dw').sum() / pl.len()).alias('turnover'),
                                  pl.len().alias('n'))
    return port.with_columns((pl.col('gross') - cost * pl.col('turnover')).alias('net')).sort(keys)


def stats(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan, t=np.nan)
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum.accumulate(eq) - 1
    mu, sd = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
    return dict(ann_ret=mu, ann_vol=sd, sharpe=mu / sd, max_dd=dd.min(), t=r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


def fmt(s):
    return (f"年化收益 {s['ann_ret']:7.2%}  年化波动 {s['ann_vol']:6.2%}  夏普 {s['sharpe']:5.2f}  "
            f"最大回撤 {s['max_dd']:7.2%}  t值 {s['t']:5.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    print(f'品种数 {df["product"].n_unique()}，日期 {df["trading_day"].min()} ~ {df["trading_day"].max()}')

    base = backtest(df, k=12, lag=0)
    p = portfolio(base, cost=0.0003)
    print('\n== 基准：k=12，月末收盘成交，全部品种等权 ==')
    for start in ('2006-01-01', '2010-01-01', '2016-01-01'):
        q = p.filter(pl.col('trading_day') >= pl.lit(start).str.to_date())
        print(f'{start[:4]} 起  毛收益: {fmt(stats(q["gross"]))}')
        print(f'{"":9s}扣 3bp 成本: {fmt(stats(q["net"]))}')
    print(f'平均持仓品种数 {p["n"].mean():.1f}，年化换手 {p["turnover"].mean() * ANN:.1f} 倍名义本金')

    lag1 = portfolio(backtest(df, k=12, lag=1))
    print(f'\n次日收盘成交（lag=1），毛收益: {fmt(stats(lag1["gross"]))}')
    print('\n== 成本敏感性（月末收盘成交，2006 起） ==')
    for c in (0.0003, 0.0010):
        print(f'扣 {c*1e4:2.0f}bp 成本: {fmt(stats(portfolio(base, cost=c)["net"]))}')

    print('\n== 按资产类别（各类内部等权） ==')
    pc = portfolio(base, by='cls')
    for c in ('commodity', 'equity', 'bond'):
        q = pc.filter(pl.col('cls') == c)
        print(f'{c:10s} 平均品种数 {q["n"].mean():5.1f}  {fmt(stats(q["gross"]))}')

    lo = portfolio(backtest(df, k=12, lag=0, long_only=True))
    print(f'\n== 对照：始终做多（同样按波动率缩放） ==\n{fmt(stats(lo["gross"]))}')

    print('\n== 回看期网格（持有 1 个月，月末收盘成交，毛收益） ==')
    grid = {}
    for k in (1, 3, 6, 9, 12, 24):
        g = portfolio(backtest(df, k=k, lag=0))
        grid[k] = g
        print(f'k={k:2d}  {fmt(stats(g["gross"]))}')

    # 逐年
    yr = (p.group_by(pl.col('trading_day').dt.year().alias('year'))
            .agg(((1 + pl.col('gross')).product() - 1).alias('tsmom'), pl.col('n').mean().alias('n'))
            .join(lo.group_by(pl.col('trading_day').dt.year().alias('year'))
                    .agg(((1 + pl.col('gross')).product() - 1).alias('long_only')), on='year')
            .sort('year'))
    print('\n== 逐年收益 ==')
    for r in yr.iter_rows(named=True):
        print(f"{r['year']}  TSMOM {r['tsmom']:8.2%}   始终做多 {r['long_only']:8.2%}   品种数 {r['n']:5.1f}")

    # 微笑形：月度 TSMOM 对月度“始终做多”组合及其平方回归
    m = (p.join(lo.select('trading_day', pl.col('gross').alias('mkt')), on='trading_day')
           .group_by(pl.col('trading_day').dt.truncate('1mo'))
           .agg(((1 + pl.col('gross')).product() - 1).alias('y'), ((1 + pl.col('mkt')).product() - 1).alias('x')))
    X = np.column_stack([np.ones(m.height), m['x'].to_numpy(), m['x'].to_numpy() ** 2])
    y = m['y'].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(y) - 3))
    print(f'\n== 微笑形检验（月度）: TSMOM = a + b·做多 + c·做多² ==\n'
          f'a={b[0]:.4f} (t={b[0]/se[0]:.2f})  b={b[1]:.3f} (t={b[1]/se[1]:.2f})  c={b[2]:.3f} (t={b[2]/se[2]:.2f})')

    # 图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2, 1]})
    def cum(x):
        return np.cumsum(np.log1p(x.fill_null(0).to_numpy()))
    ax[0].plot(p['trading_day'], cum(p['gross']), label='TSMOM 全部品种（毛）', lw=1.8)
    ax[0].plot(p['trading_day'], cum(p['net']), label='TSMOM 扣 3bp 成本', lw=1.2)
    for c, col in (('commodity', '#2a9d8f'), ('equity', '#e76f51'), ('bond', '#8d99ae')):
        q = pc.filter(pl.col('cls') == c)
        ax[0].plot(q['trading_day'], cum(q['gross']), label=f'仅{ {"commodity":"商品","equity":"股指","bond":"国债"}[c]}', lw=0.9, color=col)
    ax[0].plot(lo['trading_day'], cum(lo['gross']), label='始终做多（对照）', lw=0.9, color='grey', ls='--')
    ax[0].set_title('时间序列动量（k=12，h=1，目标波动 40%/品种）累计对数收益')
    ax[0].legend(loc='upper left', fontsize=9)
    ax[0].grid(alpha=0.3)
    sh = [stats(grid[k]['gross'])['sharpe'] for k in grid]
    ax[1].bar([str(k) for k in grid], sh, color='#264653')
    ax[1].set_title('不同回看期 k 的夏普比率（持有 1 个月）')
    ax[1].set_xlabel('回看期（月）')
    ax[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{OUT}/tsmom.png', dpi=130)
    print(f'\n图已保存 {OUT}/tsmom.png')


if __name__ == '__main__':
    main()
