"""生成《Volatility is Rough》复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 里的结果文件读取。输出: report.html"""
import html
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rough_vol as RV  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
NAMES = [n for _, _, n in RV.PRODUCTS]
MODELS = ['RFSV', 'HAR', 'AR(10)', 'AR(5)']


def load():
    h = pl.read_parquet(f'{RV.OUT}/h_estimates.parquet')
    dist = pl.read_parquet(f'{RV.OUT}/dist.parquet')
    cov = pl.read_parquet(f'{RV.OUT}/cov.parquet')
    fc = pl.read_parquet(f'{RV.OUT}/forecast.parquet')
    lm = pl.read_parquet(f'{RV.OUT}/longmem.parquet')
    return h, dist, cov, fc, lm


def cell(df, name, label, k):
    """按位置取列，避免列名里的特殊字符。"""
    r = df.filter((pl.col('品种') == name) & (pl.col('代理') == label)).row(0)
    return r[k]


def scaling_svg():
    """沪深300（1 小时窗口）：log m(q,Δ) 对 log Δ，各 q 一条，近似直线即为标度成立。"""
    x = np.log(pl.read_parquet(f'{RV.OUT}/vol1h_IF.parquet')['vol'].to_numpy())
    W, Hh, left, y0, ph, pw = 760, 330, 56, 20, 250, 520
    lx = np.log(RV.LAGS)
    lines = {q: np.log([RV.m_q(x, q, l) for l in RV.LAGS]) for q in RV.QS}
    lo, hi = min(v.min() for v in lines.values()) - 0.1, max(v.max() for v in lines.values()) + 0.1
    xs = lambda v: left + (v - lx[0]) / (lx[-1] - lx[0]) * pw
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="沪深300 波动率标度图">']
    ends = []
    for t in np.arange(np.ceil(lo), hi, 1.0):
        s.append(f'<line x1="{left}" x2="{left+pw}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="grid"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t:.0f}</text>')
    for d in (1, 2, 5, 10, 20, 30):
        s.append(f'<text x="{xs(np.log(d)):.1f}" y="{y0+ph+18}" text-anchor="middle" class="sub">Δ={d}</text>')
    for k, q in enumerate(RV.QS):
        v = lines[q]
        b = np.polyfit(lx, v, 1)
        cls = 'accent' if q == 2.0 else 'muted'
        s.append(f'<line x1="{xs(lx[0]):.1f}" y1="{yv(np.polyval(b, lx[0])):.1f}" x2="{xs(lx[-1]):.1f}" y2="{yv(np.polyval(b, lx[-1])):.1f}" class="ln {cls} dash" stroke-width="1"/>')
        for a, c in zip(lx, v):
            s.append(f'<circle cx="{xs(a):.1f}" cy="{yv(c):.1f}" r="2.5" class="dot {cls}"><title>q={q}，Δ={np.exp(a):.0f}：{c:.2f}</title></circle>')
        ends.append([yv(v[-1]), yv(v[-1]), f'q={q:g}：斜率 {b[0]:.3f} = {b[0]/q:.3f}×q', cls])
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 15)
    for ly, py, txt, cls in ends:
        if abs(ly - py) > 2:
            s.append(f'<line x1="{left+pw+2}" y1="{py:.1f}" x2="{left+pw+10}" y2="{ly:.1f}" class="axis"/>')
        s.append(f'<text x="{left+pw+12}" y="{ly+4:.1f}" class="lbl {cls}t">{txt}</text>')
    s.append('</svg>')
    return ''.join(s)


def h_svg(h):
    W, Hh, left, y0, ph = 760, 300, 40, 20, 220
    band = (W - left - 20) / len(NAMES)
    bw = band * 0.32
    top = 0.2
    yv = lambda v: y0 + (top - v) / top * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="各品种 H 估计">']
    s.append(f'<rect x="{left}" y="{yv(0.16):.1f}" width="{W-left-20}" height="{yv(0.06)-yv(0.16):.1f}" class="bandr"/>'
             f'<text x="{W-24}" y="{yv(0.16)+14:.1f}" text-anchor="end" class="sub">论文各指数的 H 范围 0.06~0.16</text>')
    for t in (0.05, 0.1, 0.15, 0.2):
        s.append(f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t:.2f}</text>'
                 f'<line x1="{left}" x2="{W-20}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="grid"/>')
    s.append(f'<line x1="{left}" x2="{W-20}" y1="{yv(0):.1f}" y2="{yv(0):.1f}" class="axis"/>')
    for i, n in enumerate(NAMES):
        for k, (lab, cls) in enumerate((('1 小时窗口', 'c1'), ('全天 RV', 'c2'))):
            v = cell(h, n, lab, 3)
            x = left + i * band + band * 0.16 + k * bw
            s.append(f'<rect x="{x:.1f}" y="{yv(v):.1f}" width="{bw-2:.1f}" height="{yv(0)-yv(v):.1f}" class="{cls}"><title>{n} {lab}：H = {v:.3f}</title></rect>'
                     f'<text x="{x+bw/2-1:.1f}" y="{yv(v)-4:.1f}" text-anchor="middle" class="sm">{v:.2f}</text>')
        s.append(f'<text x="{left+i*band+band/2:.1f}" y="{yv(0)+18:.1f}" text-anchor="middle" class="sub">{html.escape(n)}</text>')
    for k, (cls, nm) in enumerate((('c1', '1 小时窗口'), ('c2', '全天 RV'))):
        s.append(f'<rect x="{left+k*120}" y="{Hh-22}" width="12" height="12" class="{cls}"/><text x="{left+k*120+18}" y="{Hh-12}" class="lbl">{nm}</text>')
    s.append('</svg>')
    return ''.join(s)


def fc_svg(fc):
    """各模型相对 HAR 的 P 值差距（8 品种平均），按预测期。"""
    har = fc.filter(pl.col('模型') == 'HAR').select('品种', 'Δ', pl.col('P').alias('har'))
    rel = (fc.join(har, on=['品种', 'Δ']).with_columns(((pl.col('P') / pl.col('har') - 1) * 100).alias('rel'))
             .group_by(['模型', 'Δ']).agg(pl.col('rel').mean()))
    val = {(m, d): v for m, d, v in rel.iter_rows()}
    W, Hh, left, y0, ph = 760, 300, 56, 20, 230
    hs = [1, 5, 20]
    lo, hi = min(val.values()) - 1, max(val.values()) + 1
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    xs = lambda i: left + 40 + i * (W - left - 240) / (len(hs) - 1)
    style = {'RFSV': ('accent', 3), 'HAR': ('fg', 2)}
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="预测误差相对 HAR">']
    for t in range(int(np.floor(lo / 2) * 2), int(hi) + 1, 2):
        s.append(f'<line x1="{left}" x2="{W-180}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="{"axis" if t == 0 else "grid"}"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t:+d}%</text>')
    for i, d in enumerate(hs):
        s.append(f'<text x="{xs(i):.1f}" y="{y0+ph+22}" text-anchor="middle" class="sub">未来 {d} 天</text>')
    ends = []
    for m in MODELS:
        cls, w = style.get(m, ('muted', 1.5))
        pts = [(xs(i), yv(val[(m, d)])) for i, d in enumerate(hs)]
        s.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" class="ln {cls}" stroke-width="{w}"/>')
        for (a, b), d in zip(pts, hs):
            s.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" class="dot {cls}"><title>{m}，未来 {d} 天：{val[(m, d)]:+.1f}%</title></circle>')
        ends.append([pts[-1][1], pts[-1][0], pts[-1][1], m, cls])
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 15)
    for ly, lx, py, m, cls in ends:
        if abs(ly - py) > 2:
            s.append(f'<line x1="{lx+4:.1f}" y1="{py:.1f}" x2="{lx+12:.1f}" y2="{ly:.1f}" class="axis"/>')
        s.append(f'<text x="{lx+14:.1f}" y="{ly+4:.1f}" class="lbl {cls}t">{m}</text>')
    s.append('</svg>')
    return ''.join(s)


def lm_svg(lm):
    """实际 d 与模拟 RFSV 的 d 区间。"""
    W, Hh, left, y0, rh = 760, 50 + 30 * len(NAMES), 110, 20, 30
    lo, hi = 0.4, 0.75
    xs = lambda v: left + (v - lo) / (hi - lo) * (W - left - 30)
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="伪长记忆对比">']
    for t in (0.4, 0.5, 0.6, 0.7):
        s.append(f'<line x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{y0}" y2="{y0+rh*len(NAMES)}" class="{"ref dash" if t == 0.5 else "grid"}"/>'
                 f'<text x="{xs(t):.1f}" y="{y0+rh*len(NAMES)+16}" text-anchor="middle" class="sub">d={t:.1f}</text>')
    for i, r in enumerate(lm.iter_rows()):
        name, _, d_act, d_sim, rng = r
        a, b = (float(v) for v in rng.split('~'))
        y = y0 + i * rh + rh / 2
        s.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="lbl">{html.escape(name)}</text>'
                 f'<line x1="{xs(a):.1f}" x2="{xs(b):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="rng"><title>模拟 5%~95%：{rng}</title></line>'
                 f'<circle cx="{xs(d_sim):.1f}" cy="{y:.1f}" r="4" class="dot muted"><title>模拟均值 {d_sim:.3f}</title></circle>'
                 f'<circle cx="{xs(d_act):.1f}" cy="{y:.1f}" r="5" class="dot accent"><title>实际 {d_act:.3f}</title></circle>')
    s.append('</svg>')
    return ''.join(s)


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--muted:#b9b6ab;--accent:#2f6fd0;--axis:#9a978e;--c1:#2f6fd0;--c2:#9dbbea;--band:#eef3fb;--rng:#cfccc3}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c;--c1:#5b93e6;--c2:#36598f;--band:#1f2a3a;--rng:#4a4842}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#77746c;--accent:#5b93e6;--axis:#77746c;--c1:#5b93e6;--c2:#36598f;--band:#1f2a3a;--rng:#4a4842}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.meta{color:var(--sub);font-size:13px;margin-bottom:12px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--bg)}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}figcaption{color:var(--sub);font-size:12px;margin-top:6px}
figure .title{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--line)}svg .axis{stroke:var(--axis)}svg .ref{stroke:var(--axis)}svg .dash{stroke-dasharray:4 3}svg .bandr{fill:var(--band)}svg .rng{stroke:var(--rng);stroke-width:8;stroke-linecap:round}
svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}svg .ln.fg{stroke:var(--fg)}svg .ln.muted{stroke:var(--muted)}
svg .dot.accent{fill:var(--accent)}svg .dot.fg{fill:var(--fg)}svg .dot.muted{fill:var(--muted)}
svg text{fill:var(--fg);font-size:12px}svg .sm{font-size:10px;fill:var(--sub)}svg .sub{fill:var(--sub)}svg .accentt{fill:var(--accent);font-weight:600}svg .fgt{fill:var(--fg)}svg .mutedt{fill:var(--sub)}
svg .c1{fill:var(--c1)}svg .c2{fill:var(--c2)}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    h, dist, cov, fc, lm = load()
    avg = {(m, d): v for d, m, v in fc.group_by(['Δ', '模型']).agg(pl.col('P').mean()).iter_rows()}
    win20 = sum(1 for n in NAMES if fc.filter((pl.col('品种') == n) & (pl.col('Δ') == 20)).sort('P')['模型'][0] == 'RFSV')
    hh = h.filter(pl.col('代理') == '1 小时窗口')['H(=ζ2/2)']
    p = []
    p.append('<h1>中国期货波动率粗糙度（Rough Volatility）复现报告</h1><div class="meta">复现论文：Jim Gatheral, Thibault Jaisson &amp; Mathieu Rosenbaum，'
             '《Volatility is Rough》，<i>Quantitative Finance</i>，2018 · 8 个品种，样本至 2026-09 · 只用期货数据（无期权）</div>')
    p.append(f'<p class="lead">论文的结论在中国期货上基本成立：对数波动率像 Hurst 指数 H≈0.1 的分数布朗运动，远比布朗运动（H=0.5）粗糙；'
             f'只有一个参数的 RFSV 预测在 5 天、20 天上胜过 HAR 和 AR；HAR 研究里看到的“长记忆”（d≈0.5~0.6）可以完全由粗糙波动率解释。'
             f'与论文不同的是：中国期货的波动率增量明显厚尾。</p>')
    p.append(f'<div class="kpis"><div class="kpi"><b>{hh.min():.2f}~{hh.max():.2f}</b><span>H（1 小时窗口；论文 0.06~0.16）</span></div>'
             f'<div class="kpi"><b>0.97~1.00</b><span>Cov 对 Δ^2H 线性 R²</span></div>'
             f'<div class="kpi"><b>{(avg[("RFSV", 20)] / avg[("HAR", 20)] - 1) * 100:+.1f}%</b><span>RFSV 相对 HAR 的 20 天预测误差</span></div>'
             f'<div class="kpi"><b>{win20}/8</b><span>20 天预测 RFSV 最优的品种数</span></div></div>')

    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['资产', '21 个股指（Oxford-Man 已实现波动率库）、DAX/Bund 期货高频数据', '沪深 300、10 年期国债、螺纹钢、铜、豆粕、原油、黄金、PTA（主力连续）'],
        ['波动率代理', '每日已实现方差；另用固定 1 小时窗口减少积分平滑', '① 每天 10:30–11:29 的 1 分钟收益已实现波动率；② 全天 5 分钟 RV（沿用 HAR 研究）'],
        ['H 的估计', 'm(q,Δ) = E|log σ(t+Δ) − log σ(t)|^q，log m 对 log Δ 回归，斜率 ζ_q = qH', '同左；q = 0.5/1/1.5/2/3，Δ = 1~30 个交易日；H = ζ₂/2'],
        ['分布检验', '增量近似正态，标准差按 Δ^H 缩放', '峰度 + 标准差比（Δ = 5、25 对 Δ = 1）'],
        ['协方差结构', 'Cov(log σ_t, log σ_t+Δ) 对 Δ^2H 线性（RFSV）', '同左，对比长记忆的幂律形式；Δ = 1~100'],
        ['预测', 'RFSV 公式 vs AR(5)、AR(10)、HAR；500 日滚动；P = 预测误差平方和 / 方差', '同左，预测对数方差，Δ = 1、5、20 天'],
        ['伪长记忆', '模拟 RFSV 会被长记忆检验误判', '用估计的 H、ν 模拟分数布朗运动（Davies-Harte），20 次，Local Whittle 估计 d']]))

    p.append('<h2>一、H 很小：波动率是粗糙的</h2>')
    p.append(f'<figure><div class="title">8 个品种的 H 都在 0.06~0.14，落在论文的范围内</div>{h_svg(h)}'
             f'<figcaption>来源：rough_vol.py · q = 2，Δ = 1~30 天；浅色带为论文中各股指 H 的范围</figcaption></figure>')
    rows = []
    for n in NAMES:
        for lab in ('1 小时窗口', '全天 RV'):
            r = h.filter((pl.col('品种') == n) & (pl.col('代理') == lab)).row(0)
            rows.append([n, lab, r[2], f'{r[3]:.3f}', f'{r[4]:.3f}'] + [f'{v:.3f}' for v in r[5:10]] + [f'{r[10]:.3f} / {r[11]:.3f}'])
    p.append(table(['品种', '代理', '天数', 'H', 'ν'] + [f'ζ{q:g}/{q:g}' for q in RV.QS] + ['前半 / 后半 H'], rows))
    p.append('<p>全天 RV 的 H 比 1 小时窗口普遍高 0.01~0.02，与论文“积分平滑抬高 H”的说法一致。前后半样本的 H 都在 0.06~0.17，结论稳定。</p>')

    p.append('<h2>二、单分形标度成立</h2>')
    p.append(f'<figure><div class="title">沪深 300：不同 q 的 log m(q,Δ) 都是直线，斜率 ≈ q × 0.11</div>{scaling_svg()}'
             f'<figcaption>来源：rough_vol.py · 1 小时窗口；纵轴为 log m(q,Δ)，横轴为 log Δ（对数刻度）；虚线为 OLS 拟合</figcaption></figure>')
    p.append('<p>q ≤ 2 时 ζ_q/q 几乎不变（单分形）；q = 3 时略低，铜、黄金、PTA 最明显，这是极端值（厚尾）的影响，见下一节。</p>')

    p.append('<h2>三、增量分布：标度成立，但明显厚尾（与论文不同）</h2>')
    p.append(table(['品种', '代理', '峰度 Δ=1', '峰度 Δ=5', '峰度 Δ=25', '标准差比 Δ=5', '标准差比 Δ=25'],
                   [[n, lab] + [f'{cell(dist, n, lab, k):.2f}' for k in (3, 4, 6)] + [f'{cell(dist, n, lab, k):.3f}' for k in (5, 7)]
                    for n in NAMES for lab in ('1 小时窗口', '全天 RV')]))
    p.append('<p>标准差比 = 实际标准差 ÷（Δ=1 的标准差 × Δ^H），全部在 0.93~1.03，说明标度规律成立。但峰度远高于正态的 3：'
             'Δ=1 时 3.2~15，Δ=25 时仍有 3.8~7.6。论文中各股指的增量近似正态；中国期货的政策冲击、涨跌停、节假日跳空更多，'
             '用纯高斯 fBm 建模会低估波动率的跳跃。</p>')

    p.append('<h2>四、协方差结构：支持 RFSV，而不是长记忆</h2>')
    p.append(table(['品种', '代理', 'R²：Cov 对 Δ^2H 线性（RFSV）', 'R²：log Cov 对 log Δ（长记忆幂律）'],
                   [[n, lab, f'{cell(cov, n, lab, 2):.3f}', f'{cell(cov, n, lab, 3):.3f}'] for n in NAMES for lab in ('1 小时窗口', '全天 RV')]))
    p.append('<p>16 个组合中，RFSV 的线性形式 R² 全部在 0.965 以上，且全部高于长记忆的幂律形式（0.82~0.96）。</p>')

    p.append('<h2>五、预测：期限越长，RFSV 优势越大</h2>')
    p.append(f'<figure><div class="title">1 天 RFSV 略差；5 天、20 天 RFSV 最好</div>'
             f'<div class="meta">P 值（预测误差平方和 ÷ 方差）相对 HAR 的差距，8 个品种平均；负数表示比 HAR 更准</div>{fc_svg(fc)}'
             f'<figcaption>来源：rough_vol_tests.py · 全天 RV 的对数方差，500 日滚动窗口</figcaption></figure>')
    piv = fc.pivot(index=['品种', 'Δ'], columns='模型', values='P')
    rows = []
    for r in piv.iter_rows(named=True):
        best = min(MODELS, key=lambda m: r[m])
        rows.append([r['品种'], r['Δ']] + [f'<b>{r[m]:.3f}</b>' if m == best else f'{r[m]:.3f}' for m in MODELS])
    rows += [['<i>8 品种平均</i>', d] + [f'{avg[(m, d)]:.3f}' for m in MODELS] for d in (1, 5, 20)]
    p.append(table(['品种', 'Δ（天）'] + MODELS, rows))
    p.append('<p>RFSV 只用一个参数 H（全样本估计，同论文），不做任何回归，但 20 天预测在全部 8 个品种上最好，原油、螺纹钢上比 HAR 好 10% 以上。'
             '1 天预测 HAR、AR 略好，这与论文一致：短期内各模型差别很小，带回归截距的模型能更好地贴合均值。</p>')

    p.append('<h2>六、伪长记忆：HAR 研究里 d≈0.5 的答案</h2>')
    p.append(f'<figure><div class="title">实际 d（蓝点）几乎都落在模拟 RFSV 的 d 区间（灰条）内</div>{lm_svg(lm)}'
             f'<figcaption>来源：rough_vol_tests.py · Local Whittle，带宽 n^0.65；灰条为 20 次模拟的 5%~95%，灰点为模拟均值；RFSV 本身没有长记忆</figcaption></figure>')
    p.append(table(['品种', 'H', '实际 RV 的 d', '模拟 RFSV 的 d（均值）', '模拟 5%~95%'],
                   [[r[0], f'{r[1]:.3f}', f'{r[2]:.3f}', f'{r[3]:.3f}', r[4]] for r in lm.iter_rows()]))
    p.append('<p>HAR 研究中估出 d = 0.47~0.64、部分超过平稳边界 0.5，当时排除了夜盘等结构变化的解释。这里给出答案：'
             '一个没有长记忆、H≈0.1 的 RFSV 过程，被同样的方法估计时 d 就在 0.55~0.60 附近。所谓“比海外更强的长记忆”，主要是粗糙波动率带来的统计假象。'
             '这也解释了为什么只看 22 天以内信息的 HAR 就能与 ARFIMA 打平。</p>')

    p.append('<h2>与论文的差异</h2><ul><li>论文用 Oxford-Man 库的股指已实现方差；这里用期货主力连续的 1 分钟、5 分钟收益。</li>'
             '<li>1 小时窗口选 10:30–11:29（避开开盘和午休）；全天 RV 含夜盘、跳过休市跳空。</li>'
             '<li>RFSV 预测的核积分截断在 500 天并归一化；H 用全样本估计（同论文，对 RFSV 略有利）。</li>'
             '<li>没有期权数据，论文后续关于隐含波动率曲面的部分（rBergomi、rough Heston）未做。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [
        ['<code>rough_vol.py</code>', '波动率代理计算；m(q,Δ) 标度回归估计 H、ν'],
        ['<code>rough_vol_tests.py</code>', '增量分布、协方差结构、预测对比、伪长记忆模拟'], ['<code>build_report.py</code>', '生成本报告'],
        ['<code>output/vol1h_*.parquet</code>', '各品种每日 1 小时窗口波动率'], ['<code>output/h_estimates.parquet</code>', 'H 估计'],
        ['<code>output/dist.parquet、cov.parquet、forecast.parquet、longmem.parquet</code>', '各项检验结果']]))

    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>粗糙波动率复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
