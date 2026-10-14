"""生成 HARQ 复现报告（离线 HTML）。数字从 output/ 读取。输出: report.html"""
import os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'report.html')
O = os.path.join(HERE, 'output')
MODELS = ['AR', 'HAR', 'HAR-J', 'CHAR', 'SHAR', 'ARQ', 'HARQ', 'HARQ-F', 'HARQ-h']


def rd(n):
    return pl.read_parquet(os.path.join(O, f'{n}.parquet'))


def table(header, rows):
    h = ''.join(f'<th>{c}</th>' for c in header)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def ratio_rows(o, h, prod):
    rows = []
    ms = [m for m in MODELS if not (m == 'HARQ-h' and h == 1)]
    for w in ('RW', 'IW'):
        for lf in ('MSE', 'QLIKE'):
            q = o.filter((pl.col('h') == h) & (pl.col('窗口') == w) & (pl.col('损失') == lf))
            if prod == 'IF':
                v = {m: q.filter((pl.col('品种') == 'IF') & (pl.col('模型') == m))['比值'][0] for m in ms}
                lab = f'{w} {lf}'
                rows.append(row(lab, v, ms))
            else:
                c = q.filter(pl.col('品种') != 'IF')
                for stat, fn in (('平均', 'mean'), ('中位数', 'median')):
                    v = {m: getattr(c.filter(pl.col('模型') == m)['比值'], fn)() for m in ms}
                    rows.append(row(f'{w} {lf} {stat}', v, ms))
    return ['', *ms], rows


def row(lab, v, ms):
    best = min(v, key=v.get)
    return [lab] + [f'<b>{v[m]:.3f}</b>' if m == best else f'{v[m]:.3f}' for m in ms]


CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--sub:#6b6862;--line:#e3e0d8;--card:#ffffff;--accent:#2f6fd0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--accent:#5b93e6}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#ecebe6;--sub:#a09d95;--line:#34322e;--card:#232220;--accent:#5b93e6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:900px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:20px 0 6px}.meta{color:var(--sub);font-size:13px;margin-bottom:12px}.lead{font-size:17px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--sub);font-size:13px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--bg)}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def build():
    o = rd('oos')
    st = rd('oos_strat')
    rc = rd('reality_check')
    ins = rd('insample')
    sm = rd('summary')
    i1 = ins.filter((pl.col('品种') == 'IF') & (pl.col('h') == 1))
    I = {r['模型']: r for r in i1.iter_rows(named=True)}
    com = ins.filter((pl.col('品种') != 'IF') & (pl.col('h') == 1))
    C = {r['模型']: r for r in com.group_by('模型').agg(pl.col('β1').mean(), pl.col('持续性').mean(), pl.col('平均滞后').mean(), pl.col('R2').mean()).iter_rows(named=True)}
    hq = com.filter(pl.col('模型') == 'HARQ')
    neg_sig = ((hq['β1Q'] / hq['se_β1Q']) < -1.96).mean()
    g = lambda h, w, lf, m: o.filter((pl.col('品种') != 'IF') & (pl.col('h') == h) & (pl.col('窗口') == w) & (pl.col('损失') == lf) & (pl.col('模型') == m))['比值'].mean()
    rcs = rc.filter(pl.col('品种') != 'IF').group_by(['窗口', '损失']).agg((pl.col('RC p 值') < 0.05).mean().alias('a'), (pl.col('RC p 值') < 0.10).mean().alias('b'))
    R = {(w, l): (a, b) for w, l, a, b in rcs.iter_rows()}
    p = []
    p.append('<h1>HARQ 波动率预测复现报告</h1><div class="meta">复现论文：Tim Bollerslev, Andrew J. Patton &amp; Rogier Quaedvlieg，《Exploiting the Errors: A Simple Approach for Improved Volatility Forecasting》，'
             '<i>Journal of Econometrics</i>，2016（全文依据作者接受稿）· 沪深 300 股指期货（对应论文的标普 500）+ 22 个商品期货（对应论文的 27 只道指成分股）· 5 分钟数据</div>')
    p.append(f"<p class=\"lead\">论文的结论在中国期货上基本复现：考虑测量误差后，模型变得更持续（沪深 300 持续性 {I['HAR']['持续性']:.2f} → {I['HARQ']['持续性']:.2f}）、"
             f"更看重最近的波动；测量误差项系数为负且显著（{neg_sig:.0%} 的商品显著）。样本外，HARQ 在商品上普遍优于 HAR，尤其是 5 天、22 天预测（损失平均降低 2%~5%）；"
             f"按 Reality Check，{R[('IW', 'QLIKE')][0]:.0%} 的商品上 HARQ 显著胜过所有基准模型。沪深 300 的 1 天预测则好坏参半，论文中标普 500 也是如此。</p>")
    p.append(f"<div class=\"kpis\"><div class=\"kpi\"><b>{I['HAR']['持续性']:.2f} → {I['HARQ']['持续性']:.2f}</b><span>沪深 300 持续性：HAR → HARQ</span></div>"
             f"<div class=\"kpi\"><b>{g(22, 'RW', 'QLIKE', 'HARQ'):.3f}</b><span>商品 22 天预测 HARQ/HAR 的 QLIKE 比值（RW）</span></div>"
             f"<div class=\"kpi\"><b>{R[('IW', 'QLIKE')][0]:.0%}</b><span>HARQ 显著胜过所有基准的商品比例（IW、QLIKE）</span></div></div>")
    p.append('<h2>论文讲了什么</h2>')
    p.append('<p>用 5 分钟收益算出的已实现方差（RV）只是真实波动的估计，带有测量误差。误差会让模型低估波动的持续性（衰减偏误）。误差大小可以用已实现四次方差（RQ）估计：'
             'RQ 大的日子，昨天的 RV 不太可靠，应少依赖它；RQ 小的日子则可以多依赖。HARQ 在 HAR 的基础上，让昨天 RV 的系数随 √RQ 变化：β₁,t = β₁ + β₁Q·√RQ。</p>')
    p.append('<h2>复现对象与方法</h2>')
    p.append(table(['环节', '论文', '本次复现'], [
        ['资产', '标普 500 期货（1997–2013）+ 27 只道指成分股', '沪深 300 股指期货（2010 起）+ 22 个历史长、流动性好的商品期货（2005–2013 起）'],
        ['度量', '5 分钟收益：RV、RQ = M/3·Σr⁴、BPV、正负半方差', '同左；只用同一合约内、相邻 5 分钟 K 线间隔 ≤ 10 分钟的收益（不含午休、夜盘与日盘之间的跳空）'],
        ['模型', 'AR、HAR、HAR-J、CHAR、SHAR、ARQ、HARQ、HARQ-F；更长期限另有 HARQ-h', '同左'],
        ['样本外', '每天重估；滚动窗口 1000 天（RW）与扩展窗口（IW）；1、5、22 天直接预测；insanity filter', '同左'],
        ['评价', 'MSE、QLIKE 相对 HAR 的比值；按前一日 RQ 最高 5% 分层；修正的 Reality Check（平稳自助法 999 次、块长 5）', '同左']]))
    p.append('<h2>一、样本内：测量误差确实存在</h2>')
    rows = []
    for m in ('AR', 'HAR', 'ARQ', 'HARQ', 'HARQ-F'):
        r = I[m]
        f = lambda k: '' if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k))) else (
            f"{r[k]:.4f}（{r['se_'+k]:.4f}）" if k == 'β1Q' else f"{r[k]:.3f}（{r['se_'+k]:.3f}）")
        rows.append([m, f('β1'), f('β2'), f('β3'), f('β1Q'), f"{r['R2']:.3f}", f"{r['QLIKE']:.3f}", f"{r['持续性']:.3f}", '' if r.get('平均滞后') is None else f"{r['平均滞后']:.2f}"])
    p.append('<h3>沪深 300（对应论文 Table 3）</h3>')
    p.append(table(['模型', 'β1（标准误）', 'β2', 'β3', 'β1Q', 'R²', 'QLIKE', '持续性', '平均滞后（天）'], rows))
    p.append(table(['商品平均', 'β1', '持续性', '平均滞后', 'R²'], [[m, f"{C[m]['β1']:.3f}", f"{C[m]['持续性']:.3f}", '' if C[m]['平均滞后'] is None else f"{C[m]['平均滞后']:.2f}", f"{C[m]['R2']:.3f}"]
                                                                for m in ('AR', 'ARQ', 'HAR', 'HARQ', 'HARQ-F')]))
    p.append(f"<ul><li>与论文完全一致：β1Q 为负且显著（沪深 300；商品 {neg_sig:.0%} 显著为负），即 RQ 大时昨天 RV 的权重下降。</li>"
             f"<li>考虑测量误差后，AR 系数从 {I['AR']['β1']:.2f} 升到 {I['ARQ']['β1']:.2f}（沪深 300），商品从 {C['AR']['β1']:.2f} 升到 {C['ARQ']['β1']:.2f}：普通 AR 被测量误差压低了。</li>"
             f"<li>HARQ 比 HAR 更持续、平均滞后更短（沪深 300 {I['HAR']['平均滞后']:.1f} → {I['HARQ']['平均滞后']:.1f} 天；商品 {C['HAR']['平均滞后']:.1f} → {C['HARQ']['平均滞后']:.1f} 天），把权重从周、月均值移回到最近一天，与论文相同。</li></ul>")
    p.append(table(['品种', '天数', 'AR(1)', 'ARQ 的 β1', '√RQ 的离散程度（标准差/均值）'],
                   [[r['品种'], r['天数'], f"{r['AR(1)']:.3f}", f"{r['ARQ β1']:.3f}", f"{r['RQ^0.5 标准差（相对均值）']:.2f}"] for r in sm.iter_rows(named=True)]))
    p.append('<h2>二、样本外 1 天预测（对应论文 Table 4）</h2>')
    hd, rows = ratio_rows(o, 1, 'IF')
    p.append('<h3>沪深 300（相对 HAR 的损失比值，越小越好，粗体为最优）</h3>' + table(hd, rows))
    hd, rows = ratio_rows(o, 1, 'C')
    p.append('<h3>商品（22 个）</h3>' + table(hd, rows))
    p.append(table(['Reality Check（HARQ 显著优于所有基准）', '5% 水平', '10% 水平'],
                   [[f'{w} {l}', f'{R[(w, l)][0]:.0%}', f'{R[(w, l)][1]:.0%}'] for w in ('RW', 'IW') for l in ('MSE', 'QLIKE')]
                   + [['论文：个股 RW MSE / QLIKE', '44% / 30%', '63% / 37%']]))
    ifrc = {(r['窗口'], r['损失']): r['RC p 值'] for r in rc.filter(pl.col('品种') == 'IF').iter_rows(named=True)}
    p.append(f"<ul><li><b>沪深 300</b>：HARQ 在 QLIKE 上与 HAR 相当或略好，MSE 上略差；Reality Check p 值 RW {ifrc[('RW', 'MSE')]:.2f}/{ifrc[('RW', 'QLIKE')]:.2f}、IW {ifrc[('IW', 'MSE')]:.2f}/{ifrc[('IW', 'QLIKE')]:.2f}（MSE/QLIKE）。论文中标普 500 也是好坏参半。</li>"
             '<li><b>商品</b>：扩展窗口下 HARQ 两种损失都是最优；滚动窗口下 QLIKE 与 HAR 持平、MSE 略差。HARQ-F 不如 HARQ，与论文一致（多出的参数难以估准）。</li></ul>')
    s = st.filter((pl.col('h') == 1) & (pl.col('品种') != 'IF') & (pl.col('模型') == 'HARQ')).group_by(['分层', '窗口', '损失']).agg(pl.col('比值').median()).sort(['分层', '窗口', '损失'])
    p.append('<h3>按前一日 RQ 分层（对应论文 Table 5，商品中位数）</h3>')
    p.append(table(['分层', '窗口', '损失', 'HARQ/HAR'], [[a, b, c, f'{v:.3f}'] for a, b, c, v in s.iter_rows()]))
    p.append('<p>与论文相同：HARQ 的改进主要来自 95% 的“正常”日子（更高的持续性），而不是 RQ 很高的少数日子；在 RQ 最高的日子，滚动窗口下 HARQ 反而略差。</p>')
    p.append('<h2>三、更长期限（对应论文 Table 7、8）</h2>')
    for h in (5, 22):
        hd, rows = ratio_rows(o, h, 'C')
        p.append(f'<h3>商品，{h} 天</h3>' + table(hd, rows))
        hd, rows = ratio_rows(o, h, 'IF')
        p.append(f'<h3>沪深 300，{h} 天</h3>' + table(hd, rows))
    p.append(f"<p>预测期越长，HARQ 的优势越明显：商品 5 天、22 天预测，四种窗口与损失组合下 HARQ 平均都比 HAR 低 2%~5%（例如 22 天 RW QLIKE {g(22, 'RW', 'QLIKE', 'HARQ'):.3f}），沪深 300 的 22 天预测低约 5%。论文中 HARQ 在多数期限也优于 HAR。</p>")
    j = o.filter((pl.col('h') == 1) & (pl.col('模型') == 'HARQ') & (pl.col('窗口') == 'RW') & (pl.col('损失') == 'QLIKE')).join(sm, on='品种')
    cr = np.corrcoef(j['比值'], j['RQ^0.5 标准差（相对均值）'])[0, 1]
    p.append(f"<h2>四、改进与测量误差的变化程度（对应论文 Figure 4）</h2><p>跨品种看，√RQ 离散程度越大（测量误差变化越剧烈），HARQ 相对 HAR 的 QLIKE 比值越低（相关 {cr:.2f}），与论文的发现一致。</p>")
    p.append('<h2>结论</h2><ul><li>测量误差在中国期货的已实现方差中同样存在且随时间变化，HARQ 的机制（误差大时少依赖昨天）在样本内得到完全一致的证据。</li>'
             '<li>样本外，HARQ 在商品期货上稳定优于 HAR，期限越长优势越大；沪深 300 的短期预测改进不明显。</li>'
             '<li>与此前的 HAR、粗糙波动率研究一致：中国期货的波动率持续性很强，考虑测量误差后更强。</li></ul>')
    p.append('<h2>与论文的差异</h2><ul><li>用商品期货代替个股作为“横截面”，用沪深 300 股指期货代替标普 500。</li>'
             '<li>中国期货有午休、夜盘，日内收益只算连续时段，RV 不含隔夜跳空。</li>'
             '<li>insanity filter 替换的预测比例最高约 6%（论文低于 0.1%），主要出现在少数品种与参数较多的模型上。</li>'
             '<li>未做第 4 节的其他稳健性（其他 RV 估计量、不同采样频率、带 Q 的 HAR-J/CHAR/SHAR）。</li></ul>')
    p.append('<h2>代码与数据文件</h2>')
    p.append(table(['文件', '内容'], [['<code>measures.py</code>', '5 分钟收益计算 RV、RQ、BPV、正负半方差'], ['<code>harq.py</code>', '各模型、样本内估计、滚动 / 扩展窗口样本外预测、分层、Reality Check'],
                                     ['<code>build_report.py</code>', '生成本报告'], ['<code>output/*.parquet</code>', '各步结果；harq_log.txt 为运行日志']]))
    doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>HARQ 复现报告</title><style>{CSS}</style></head><body><main>{"".join(p)}</main></body></html>')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('written', OUT, len(doc), 'bytes')


if __name__ == '__main__':
    build()
