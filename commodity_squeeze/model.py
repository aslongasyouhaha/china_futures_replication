"""《Negative Oil & Nickel Squeeze: A Feedback Model for Extreme Commodity Futures Prices》复现 —— Zimbidis & Sircar (2026, arXiv:2610.00951)

模型：观测近月价 F_obs = F_lic ∓ ε·V±，V± 为以近月/次月价差为标的的展期期权，其收益又依赖 F_obs（反馈）。
  Case 1（不能交割的一方净多，被迫卖出）：V+ = f2·G+(ξ)，(1−ε)G+ = P(ξ+εG+)
  Case 2（不能交割的一方净空，被迫买入）：V− = f2·G−(ξ)，(1−ε)G− = C(ξ−εG−)
  P、C 为 Black-Scholes-Margrabe 看跌/看涨（ξ = f1/f2，执行价 D，方差 σ_M²(T−t)）
  反推隐含失衡（式 57）：ε+ = (F_lic−F_obs) / (F_lic−F_obs + F2·P(F_lic/F2))，ε− 同理用 C
本文件：公式实现 + 复现论文第 3 节与第 4 节的数值（Figure 1、Monte Carlo、WTI 与镍的隐含失衡）。
"""
import numpy as np
from scipy.stats import norm


def margrabe(xi, tau, sig, D=1.0):
    """返回 (P, C, d1, d2)；xi > 0。"""
    s = sig * np.sqrt(tau)
    d1 = (np.log(xi / D) + 0.5 * s * s) / s
    d2 = d1 - s
    P = D * norm.cdf(-d2) - xi * norm.cdf(-d1)
    C = xi * norm.cdf(d1) - D * norm.cdf(d2)
    return P, C, d1, d2


def _bisect(fun, lo, hi, n=200):
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    lo, hi = lo.copy(), hi.copy()
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        pos = fun(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def z_case1(xi, eps, tau, sig, D=1.0):
    """Case 1：由观测比值 ξ 求 z = ξ + εG+（z > 0），利用 ξ = z − ε/(1−ε)·P(z) 单调递增（第 3.2 节），可处理 ξ ≤ 0。"""
    xi = np.asarray(xi, float)
    f = lambda z: z - eps / (1 - eps) * margrabe(z, tau, sig, D)[0] - xi
    return _bisect(f, np.full_like(xi, 1e-12), np.maximum(xi, 0) + 10 * D + 10)


def G_plus(xi, eps, tau, sig, D=1.0):
    z = z_case1(xi, eps, tau, sig, D)
    return margrabe(z, tau, sig, D)[0] / (1 - eps)


def G_minus(xi, eps, tau, sig, D=1.0):
    """Case 2：y = ξ − εG−，ξ = y + ε/(1−ε)·C(y) 单调递增（推论 2.5）。"""
    xi = np.asarray(xi, float)
    f = lambda y: y + eps / (1 - eps) * margrabe(y, tau, sig, D)[1] - xi
    y = _bisect(f, np.full_like(xi, 1e-12), xi)
    return margrabe(y, tau, sig, D)[1] / (1 - eps)


def implied_eps(f_obs, f2, f_lic, tau, sig, D=1.0):
    """式 (57)：近月低于基准 → Case 1（ε+），高于基准 → Case 2（ε−）。返回 (ε, case)。"""
    P, C, _, _ = margrabe(f_lic / f2, tau, sig, D)
    if f_obs <= f_lic:
        g = f_lic - f_obs
        return g / (g + f2 * P), 1
    g = f_obs - f_lic
    return g / (g + f2 * C), 2


def mc_case1(f1_0=10.0, f2_0=25.0, eps=0.8, D=1.0, T=1 / 3, s1=0.5, s2=0.3, rho=0.3, mus=(-0.3, 0.0, 0.3),
             n_paths=20000, steps_per_day=2, seed=0):
    """第 3.4 节：Case 1 观测价的扩散（β、γ 由显式解给出），统计曾跌破 0 的路径比例。"""
    sigM = np.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    n = int(round(T * 252 * steps_per_day))
    dt = T / n
    out = {}
    for mu in mus:
        rng = np.random.default_rng(seed)
        f1 = np.full(n_paths, f1_0)
        f2 = np.full(n_paths, f2_0)
        crossed = np.zeros(n_paths, bool)
        alive = np.ones(n_paths, bool)
        z = z_case1(f1 / f2, eps, T, sigM, D)
        kk = eps / (1 - eps)
        for k in range(n - 1):                                    # 最后一步 τ→0，停在 T 之前
            tau = T - k * dt
            xi = f1 / f2
            bound = -eps * D / (1 - eps)
            alive &= xi > bound + 1e-9
            xa = np.where(alive, xi, 1.0)
            for _ in range(30):                                   # 牛顿迭代：ξ = z − k·P(z)，dξ/dz = 1 + k·N(−d1) ≥ 1
                P_, _, d1_, _ = margrabe(z, tau, sigM, D)
                z = np.maximum(z - (z - kk * P_ - xa) / (1 + kk * norm.cdf(-d1_)), 1e-12)
            if np.abs(z - kk * margrabe(z, tau, sigM, D)[0] - xa).max() > 1e-8:
                z = z_case1(xa, eps, tau, sigM, D)
            d2 = margrabe(z, tau, sigM, D)[3]
            gam = -eps * s2 * f2 / (1 - eps) * D * norm.cdf(-d2)
            beta = s1 * f1 - s1 / s2 * gam
            w1 = rng.standard_normal(n_paths)
            w2 = rho * w1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
            sq = np.sqrt(dt)
            f1 = np.where(alive, f1 + mu * f1 * dt + beta * sq * w1 + gam * sq * w2, f1)
            f2 = f2 * np.exp(-0.5 * s2 ** 2 * dt + s2 * sq * w2)
            crossed |= alive & (f1 < 0)
        out[mu] = (crossed.mean(), (~alive).mean())
    return out


def main():
    import json
    res = {'fig1': {}, 'mc': {}}
    print('== Figure 1：D=0.90，σ_M=0.50，T−t=1/3，f2=30，在执行价 f1=Df2=27 处 ==')
    D, sig, tau, f2 = 0.9, 0.5, 1 / 3, 30.0
    xi = 27 / 30
    P, C, _, _ = margrabe(xi, tau, sig, D)
    print(f'  无反馈：V+ = {f2*P:.2f}，V− = {f2*C:.2f}（论文约 3.10）')
    res['fig1']['0'] = [float(f2 * P), float(f2 * C)]
    for e in (0.2, 0.4, 0.6, 0.8, 0.95):
        res['fig1'][str(e)] = [float(f2 * G_plus(np.array([xi]), e, tau, sig, D)[0]), float(f2 * G_minus(np.array([xi]), e, tau, sig, D)[0])]
        print(f'  ε={e:.2f}：V+ = {f2*G_plus(np.array([xi]), e, tau, sig, D)[0]:.2f}，V− = {f2*G_minus(np.array([xi]), e, tau, sig, D)[0]:.2f}'
              + ('（论文 11.25 / 8.06）' if e == 0.95 else ''))
    e = 0.8
    print(f'  Case 1 可定义的下界：ξ > −εD/(1−ε) = {-e*D/(1-e):.2f}，即近月价可跌到 {-e*D/(1-e)*f2:.1f}')
    print('\n== 第 4 节：反推隐含失衡 ==')
    eps, case = implied_eps(-37.63, 20.43, 13.14, 2 / 252, 0.52)
    res['wti'] = float(eps)
    print(f'  WTI 2020-04-20 结算：ε = {eps:.3f}（Case {case}；论文 0.87）')
    for lab, tau in (('表中所写 T−t = 1/4 年', 1 / 4), ('T−t = 1 天（1/365 年）', 1 / 365)):
        eps, case = implied_eps(101365, 60195, 62863.41, tau, 0.5)
        res.setdefault('nickel', {})[lab] = float(eps)
        print(f'  LME 镍 2022-03-08 高点，{lab}：ε = {eps:.4f}（Case {case}；论文 0.9346）')
    print('  → 论文的 0.9346 对应剩余 1 天，与 Table 2 所写的 1/4 年不一致')
    print('\n== 第 3.4 节 Monte Carlo：F0=10，F2=25，ε=0.8，D=1，T=1/3，σ1=0.5，σ2=0.3，ρ=0.3 ==')
    for spd, n in ((2, 20000), (12, 8000)):
        for mu, (cr, dead) in mc_case1(n_paths=n, steps_per_day=spd).items():
            res['mc'][f'{spd}|{mu}'] = float(cr)
            print(f'  每天 {spd} 步、{n} 条路径，μ={mu:+.1f}：曾跌破 0 的路径比例 {cr:.1%}')
    print('  论文：62.9%、61.0%、59.0%（μ = −0.3、0、0.3）；比例随步长变细而上升（离散时点才能观测到穿越）')
    import os
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'model_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
