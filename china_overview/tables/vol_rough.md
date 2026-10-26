| | (1)<br>H<br>1 hour window | (2)<br>H<br>Full Day RV | (3)<br>H<br>First half / second half sample | (4)<br>Cov vs. Δ^2H<br>R² | (5)<br>Incremental kurtosis<br>Δ = 1 | (6)<br>actual d | (7)<br>Simulating RFSV<br>d |
|---|---|---|---|---|---|---|---|
| CSI 300 stock index | 0.111 | 0.119 | 0.095 / 0.132 | 0.975 | 3.62 | 0.52 | 0.59 |
| 10-Year Treasury Bond | 0.088 | 0.108 | 0.084 / 0.098 | 0.965 | 4.60 | 0.64 | 0.58 |
| Rebar | 0.111 | 0.133 | 0.108 / 0.117 | 0.978 | 4.68 | 0.64 | 0.59 |
| Copper | 0.105 | 0.122 | 0.086 / 0.131 | 0.990 | 7.19 | 0.55 | 0.58 |
| soybean meal | 0.064 | 0.087 | 0.063 / 0.066 | 0.981 | 6.32 | 0.55 | 0.55 |
| crude oil | 0.127 | 0.142 | 0.115 / 0.149 | 0.979 | 4.89 | 0.52 | 0.60 |
| gold | 0.109 | 0.114 | 0.099 / 0.122 | 0.972 | 5.43 | 0.49 | 0.57 |
| PTA | 0.079 | 0.096 | 0.075 / 0.088 | 0.983 | 12.28 | 0.47 | 0.56 |

H is the Hurst exponent: the slope of the q-order moment of the logarithmic volatility increment versus the time interval Δ regression ÷ q (here q = 2); Brownian motion H = 0.5, the smaller it is, the "rougher" it is. 1 hour window = 1 minute return realized volatility from 10:30–11:29 each day; full day = 5 minute RV. Cov's R² against Δ^2H tests the covariance structure of the RFSV model; kurtosis is the kurtosis of Δ = 1-day log volatility increment (normal = 3); d is the ARFIMA long memory parameter, and "simulated" is d (averaged 20 times) after simulating the RFSV path at the estimated H. Source: Reproduction of Gatheral, Jaisson & Rosenbaum, Quantitative Finance, 2018.
