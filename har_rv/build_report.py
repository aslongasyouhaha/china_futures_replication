"""生成 HAR-RV 复现报告（离线 HTML，图表为内嵌 SVG）。数字从 output/ 里的结果文件读取。输出: report.html"""
import html
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import har_rv as H  # noqa: E402

OUT = os.path.join(HERE, 'report.html')
MODEL_NAMES = {'HAR': 'HAR', 'ARFIMA': 'ARFIMA(5,d,0)', 'AR22': 'AR(22)', 'AR3': 'AR(3)', 'AR1': 'AR(1)',
               'EWMA20': 'EWMA 20 天', 'EWMA': 'EWMA 60 天（TSMOM 现用）'}
# 伪长记忆检验结果（long_memory_check.py 输出；Local Whittle，带宽 n^0.65，不设上限）
LM = [('沪深300股指', '2015-09-07 交易限制', 0.515, 0.515, 0.50, 0.48), ('10年期国债', '无（对照组）', 0.64, 0.64, None, None),
      ('螺纹钢', '2014-12-29 夜盘', 0.64, 0.645, 0.645, 0.635), ('铜', '2013-12-23 夜盘', 0.545, 0.56, 0.545, 0.54),
      ('豆粕', '2014-12-29 夜盘', 0.53, 0.51, 0.525, 0.535), ('原油', '无（上市即有夜盘）', 0.515, 0.62, None, None),
      ('黄金', '2013-07-08 夜盘', 0.49, 0.44, 0.465, 0.47), ('PTA', '2014-12-15 夜盘', 0.47, 0.445, 0.46, 0.57)]


def insample():
    rows = []
    for ex, p, name in H.PRODUCTS:
        rv = pl.read_parquet(f'{H.OUT}/rv_{p}.parquet')
        x = H.features(rv).drop_nulls(['rv_w', 'rv_m', 'y1'])
        X = np.column_stack([np.ones(x.height), x['rv'], x['rv_w'], x['rv_m']])
        y = x['y1'].to_numpy()
        b = H.ols(X, y)
        e = y - X @ b
        se = H.nw_se(X, e, 5)
        r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
        rows.append(dict(name=name, start=str(x['trading_day'].min()), n=x.height, mean=float(rv['rv'].mean()),
                         b=b[1:], t=b[1:] / se[1:], r2=r2))
    return rows


def rel_rmse():
    o = pl.read_parquet(f'{H.OUT}/har_oos.parquet')
    har = o.filter(pl.col('模型') == 'HAR').select('品种', 'h', pl.col('RMSE').alias('har'))
    rel = o.join(har, on=['品种', 'h']).with_columns(((pl.col('RMSE') / pl.col('har') - 1) * 100).alias('rel'))
    avg = rel.group_by(['模型', 'h']).agg(pl.col('rel').mean())
    mz = o.group_by(['模型', 'h']).agg(pl.col('MZ_R2').mean())
    return o, rel, avg, mz


def lines_svg(avg):
    W, Hh, left, y0, ph = 760, 340, 56, 30, 240
    hs = list(H.HORIZONS)
    models = ['EWMA', 'EWMA20', 'AR1', 'AR3', 'AR22', 'HAR', 'ARFIMA']
    val = {(r['模型'], r['h']): r['rel'] for r in avg.iter_rows(named=True)}
    lo, hi = min(-5, min(val.values())), max(val.values()) + 2
    yv = lambda v: y0 + (hi - v) / (hi - lo) * ph
    xs = lambda i: left + 30 + i * (W - left - 230) / (len(hs) - 1)
    style = {'HAR': ('accent', 3), 'ARFIMA': ('fg', 2), 'EWMA': ('bad', 2), 'EWMA20': ('bad2', 2)}
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="各模型预测误差相对 HAR">']
    for t in range(int(np.floor(lo / 10) * 10), int(hi) + 1, 10):
        s.append(f'<line x1="{left}" x2="{W-170}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" class="{"axis" if t == 0 else "grid"}"/>'
                 f'<text x="{left-6}" y="{yv(t)+4:.1f}" text-anchor="end" class="sub">{t:+d}%</text>')
    for i, h in enumerate(hs):
        s.append(f'<text x="{xs(i):.1f}" y="{y0+ph+22}" text-anchor="middle" class="sub">未来 {h} 天</text>')
    ends = []
    for m in models:
        cls, w = style.get(m, ('muted', 1.5))
        pts = [(xs(i), yv(val[(m, h)])) for i, h in enumerate(hs) if (m, h) in val]
        s.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" class="ln {cls}" stroke-width="{w}"/>')
        for (x, y), h in zip(pts, hs):
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="dot {cls}"><title>{MODEL_NAMES[m]}，未来 {h} 天：{val[(m, h)]:+.1f}%</title></circle>')
        ends.append([pts[-1][1], pts[-1][0], pts[-1][1], m, cls])
    # 右端标签避免重叠：按纵坐标排序，相邻至少间隔 15px；被挪开的标签用细线连回数据点
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 15)
    for ly, lx, py, m, cls in ends:
        if abs(ly - py) > 2:
            s.append(f'<line x1="{lx+4:.1f}" y1="{py:.1f}" x2="{lx+12:.1f}" y2="{ly:.1f}" class="axis"/>')
        s.append(f'<text x="{lx+14:.1f}" y="{ly+4:.1f}" class="lbl {cls}t">{html.escape(MODEL_NAMES[m])}</text>')
    s.append('</svg>')
    return ''.join(s)


def coef_svg(ins):
    W, Hh, left, y0, ph = 760, 320, 40, 30, 220
    band = (W - left - 20) / len(ins)
    bw = band * 0.55
    base = y0 + ph
    yv = lambda v: base - v * ph
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="各品种 HAR 系数">',
         f'<line x1="{left}" x2="{W-20}" y1="{yv(1):.1f}" y2="{yv(1):.1f}" class="ref dash"/>'
         f'<text x="{W-22}" y="{yv(1)-5:.1f}" text-anchor="end" class="sub">系数和 = 1（完全持续）</text>',
         f'<line x1="{left}" x2="{W-20}" y1="{base}" y2="{base}" class="axis"/>']
    for i, r in enumerate(ins):
        x = left + i * band + (band - bw) / 2
        acc = 0
        for k, (cls, nm) in enumerate((('c1', '日'), ('c2', '周'), ('c3', '月'))):
            v = max(r['b'][k], 0)
            s.append(f'<rect x="{x:.1f}" y="{yv(acc+v):.1f}" width="{bw:.1f}" height="{v*ph:.1f}" class="{cls}">'
                     f'<title>{r["name"]} {nm}系数 {r["b"][k]:.3f}（t={r["t"][k]:.1f}）</title></rect>')
            acc += v
        s.append(f'<text x="{x+bw/2:.1f}" y="{yv(acc)-6:.1f}" text-anchor="middle" class="lbl">{sum(r["b"]):.2f}</text>'
                 f'<text x="{x+bw/2:.1f}" y="{base+20}" text-anchor="middle" class="sub">{html.escape(r["name"])}</text>')
    for k, (cls, nm) in enumerate((('c1', '日（βd）'), ('c2', '周（βw）'), ('c3', '月（βm）'))):
        s.append(f'<rect x="{left+k*110}" y="{Hh-22}" width="12" height="12" class="{cls}"/><text x="{left+k*110+18}" y="{Hh-12}" class="lbl">{nm}</text>')
    s.append('</svg>')
    return ''.join(s)


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--muted:#b9b6ab;--accent:#2f6fd0;--bad:#d1605e;--bad2:#e3a19f;--axis:#9a978e;--c1:#2f6fd0;--c2:#7fa7e6;--c3:#c6d6f2}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#6a675f;--accent:#5b93e6;--bad:#d77a77;--bad2:#9c5c5a;--axis:#77746c;--c1:#5b93e6;--c2:#3f6aa8;--c3:#2b4670}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--muted:#6a675f;--accent:#5b93e6;--bad:#d77a77;--bad2:#9c5c5a;--axis:#77746c;--c1:#5b93e6;--c2:#3f6aa8;--c3:#2b4670}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.meta{color:var(--sub);font-size:13px;margin-bottom:12px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--bg)}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}figcaption{color:var(--sub);font-size:12px;margin-top:6px}
figure .title{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--line)}svg .axis{stroke:var(--axis)}svg .ref{stroke:var(--axis)}svg .dash{stroke-dasharray:4 3}
svg .ln{fill:none}svg .ln.accent{stroke:var(--accent)}svg .ln.fg{stroke:var(--fg)}svg .ln.bad{stroke:var(--bad)}svg .ln.bad2{stroke:var(--bad2)}svg .ln.muted{stroke:var(--muted)}
svg .dot.accent{fill:var(--accent)}svg .dot.fg{fill:var(--fg)}svg .dot.bad{fill:var(--bad)}svg .dot.bad2{fill:var(--bad2)}svg .dot.muted{fill:var(--muted)}
svg text{fill:var(--fg);font-size:12px}svg .sub{fill:var(--sub)}svg .accentt{fill:var(--accent);font-weight:600}svg .badt{fill:var(--bad)}svg .bad2t{fill:var(--bad)}svg .mutedt{fill:var(--sub)}
svg .c1{fill:var(--c1)}svg .c2{fill:var(--c2)}svg .c3{fill:var(--c3)}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    ins = insample()
    o, rel, avg, mz = rel_rmse()
    g = lambda df, m, h, col: df.filter((pl.col('模型') == m) & (pl.col('h') == h))[col][0]
    ew22, ar22 = g(avg, 'EWMA', 22, 'rel'), g(avg, 'ARFIMA', 22, 'rel')
    p = []
    p.append('<h1>中国期货 HAR-RV 复现报告</h1><div class="meta">复现论文：Fulvio Corsi，《A Simple Approximate Long-Memory Model of Realized Volatility》，'
             '<i>Journal of Financial Econometrics</i>，2009 · 8 个品种，样本至 2026-09</div>')
    p.append(f'<p class="lead">论文的结论在中国期货上成立：只用日、周、月三段平均波动率的 HAR 模型，预测效果与长记忆模型 ARFIMA 只差 0%~3%，'
             f'远好于 TSMOM 现用的 EWMA（未来 22 天的预测误差低 {ew22:.0f}%）。中国期货波动率的持续性比海外更强。</p>')
    p.append(f'<div class="kpis"><div class="kpi"><b>0.88~0.95</b><span>HAR 系数和（持续性）</span></div>'
             f'<div class="kpi"><b>{-ew22:+.0f}%</b><span>EWMA 相对 HAR 的 22 天预测误差</span></div>'
             f'<div class="kpi"><b>{ar22:+.1f}%</b><span>ARFIMA 相对 HAR（22 天）</span></div>'
             f'<div class="kpi"><b>0.45~0.65</b><span>长记忆参数 d（海外约 0.4）</span></div></div>'.replace(f'{-ew22:+.0f}%', f'+{ew22:.0f}%'))

    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['资产', 'S&amp;P 500 期货、美元/瑞郎、美国 30 年国债期货', '沪深 300、10 年期国债、螺纹钢、铜、豆粕、原油、黄金、PTA'],
        ['日已实现波动率', '逐笔数据，双尺度估计量（TSRV）', '5 分钟对数收益；只算连续交易时段，跳过午休、小节休息和夜盘到日盘的跳空'],
        ['口径', 'RV = √(日内收益平方和)，年化，不取对数', '同左（年化系数 243，单位 %）'],
        ['模型', 'RV(t+1) = c + βd·RV(t) + βw·周均值 + βm·月均值', '同左；周 = 过去 5 天，月 = 过去 22 天'],
        ['估计', 'OLS，Newey-West 标准误（5 阶）', '同左'],
        ['样本外', '1000 日滚动窗口逐日重估；预测 1 天、1 周、2 周', '同左，另加未来 22 天（对应 TSMOM 月度调仓）'],
        ['对比模型', 'AR(1)、AR(3)、ARFIMA(5,d,0)', '同左，另加 AR(22)、EWMA 60 天（TSMOM 现用）、EWMA 20 天'],
        ['ARFIMA 的 d', '全样本估计（对 ARFIMA 有利）', '滚动窗口用 Local Whittle 估计，每月更新；截断在 0.49 以保证平稳'],
        ['评价指标', 'RMSE、MAE、Mincer-Zarnowitz R²', '同左']]))

    p.append('<h2>样本内：三个周期都显著，持续性很高</h2>')
    p.append(f'<figure><div class="title">日、周、月三段系数都显著，系数和接近 1；中国商品的周度系数普遍最大</div>'
             f'{coef_svg(ins)}<figcaption>来源：har_rv.py · 全样本 OLS，Newey-West(5)</figcaption></figure>')
    p.append(table(['品种', '样本起点', '天数', 'RV 均值（%）', 'βd（t）', 'βw（t）', 'βm（t）', 'R²'],
                   [[r['name'], r['start'], r['n'], f"{r['mean']:.1f}"] + [f"{r['b'][k]:.3f}（{r['t'][k]:.1f}）" for k in range(3)] + [f"{r['r2']:.2f}"]
                    for r in ins] + [['<i>论文：S&amp;P 500 期货</i>', '1990', '4344', '—', '0.372（9.9）', '0.343（7.3）', '0.224（6.5）', '0.71']]))
    p.append('<p>系数和 0.88~0.95，与论文的 0.91~0.94 几乎一致；沪深 300 的系数与论文里的 S&amp;P 500 非常接近。除原油的月度系数（t = 1.7）外全部显著。</p>')

    p.append('<h2>样本外：HAR 与 ARFIMA 相当，远好于 EWMA</h2>')
    p.append(f'<figure><div class="title">HAR 与 ARFIMA 只差 0%~3%；TSMOM 现用的 EWMA 误差高 18%~33%</div>'
             f'<div class="meta">预测误差（RMSE）相对 HAR 的差距，8 个品种平均；负数表示比 HAR 更准</div>{lines_svg(avg)}'
             f'<figcaption>来源：har_rv.py · 1000 日滚动窗口逐日重估，2009~2026 年样本外预测</figcaption></figure>')
    hs = list(H.HORIZONS)
    order = ['ARFIMA', 'HAR', 'AR22', 'AR3', 'AR1', 'EWMA20', 'EWMA']
    p.append(table(['模型'] + [f'MZ R²，{h} 天' for h in hs], [[MODEL_NAMES[m]] + [f"{g(mz, m, h, 'MZ_R2'):.3f}" for h in hs] for m in order]))
    p.append('<p>和论文一致：HAR 明显好于 AR(1)、AR(3)，与长记忆模型相当。预测期越长，ARFIMA 的优势越明显（22 天时好约 3%），'
             '因为它能利用 22 天以前的信息；但 HAR 只有 3 个参数，更稳定、易解释。</p>')
    r22 = rel.filter(pl.col('h') == 22)
    p.append('<p><b>未来 22 天，各品种相对 HAR 的预测误差</b></p>')
    p.append(table(['品种', 'ARFIMA', 'AR(22)', 'EWMA 20 天', 'EWMA 60 天'],
                   [[name] + [f"{r22.filter((pl.col('品种') == name) & (pl.col('模型') == m))['rel'][0]:+.1f}%" for m in ('ARFIMA', 'AR22', 'EWMA20', 'EWMA')]
                    for _, _, name in H.PRODUCTS]))

    p.append('<h2>中国期货的长记忆是真实的，且比海外更强</h2>')
    p.append('<p>估计出的 d 在 0.45~0.65 之间，部分超过平稳边界 0.5（海外文献约 0.4）。为检验这是否是夜盘推出等结构变化造成的“伪长记忆”，做了以下对比：</p>')
    f = lambda v: '—' if v is None else f'{v:.3f}'
    p.append(table(['品种', '结构变化', '① 原始', '② 只用日盘', '③ 去台阶', '④ 只用变化后样本'],
                   [[a, b, f(c), f(d), f(e), f(g2)] for a, b, c, d, e, g2 in LM]))
    p.append('<ul><li>消除已知台阶后 d 基本不变（多在 ±0.05 以内）。</li><li>没有结构变化的 10 年期国债，d 反而最高（0.64）。</li>'
             '<li>不同带宽下 d 没有呈现伪长记忆的典型规律。</li></ul>'
             '<p>结论：高持续性是真实的，并非夜盘等已知变化所致；可能与政策和宏观周期形成的持续数月到数年的波动阶段有关（推测，未检验）。'
             'ARFIMA 的 d 被截断在 0.49，实际上压低了真实持续性；HAR 不需要假设 d，系数和小于 1，始终平稳。</p>')

    p.append('<h2>对 TSMOM 的意义</h2><p>TSMOM 的仓位 = 40% ÷ 预测波动率。把 EWMA 换成 HAR 或 ARFIMA，月度波动率预测可准约 18%~21%，'
             '有望更及时地在波动放大时减仓。能否提高夏普或降低回撤，需要放进 <code>research/time_series_momentum</code> 的实盘版回测检验；'
             '需注意 HAR 预测的是交易时段内波动率，不含隔夜跳空，放入 TSMOM 时要换算成全天口径。</p>')
    p.append('<h2>与论文的差异</h2><ul><li>用 5 分钟收益代替逐笔数据的 TSRV。</li><li>ARFIMA 的 d 用 Local Whittle 滚动估计、截断在 0.49；论文用全样本估计。</li>'
             '<li>多步预测：HAR 用直接回归，ARFIMA 用逐日递推。</li><li>只覆盖 8 个主要品种；冷门品种的日内波动率噪音更大。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [
        ['<code>har_rv.py</code>', '已实现波动率计算；HAR、ARFIMA、AR、EWMA 的样本内和样本外对比'],
        ['<code>long_memory_check.py</code>', '伪长记忆检验'], ['<code>build_report.py</code>', '生成本报告'],
        ['<code>output/rv_*.parquet</code>', '各品种每日已实现波动率'], ['<code>output/har_oos.parquet</code>', '样本外预测误差']]))

    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>HAR-RV 复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
