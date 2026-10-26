| | (1)<br>MSE<br>h = 1 | (2)<br>MSE<br>h = 5 | (3)<br>MSE<br>h = 22 | (4)<br>QLIKE<br>h = 1 | (5)<br>QLIKE<br>h = 5 | (6)<br>QLIKE<br>h = 22 |
|---|---|---|---|---|---|---|
| AR | 1.183 | 1.274 | 1.193 | 1.357 | 1.574 | 1.361 |
| ARQ | 1.102 | 1.172 | 1.089 | 1.223 | 1.301 | 1.187 |
| HAR-J | 1.007 | 1.002 | 1.001 | 1.003 | 0.998 | 1.001 |
| CHAR | 1.011 | 1.009 | 1.015 | 1.009 | 1.004 | 1.008 |
| SHAR | 1.007 | 1.004 | 1.001 | 1.007 | 0.999 | 1.000 |
| HARQ | 0.998 | 0.973 | 0.969 | 0.994 | 0.959 | 0.956 |
| HARQ-F | 1.007 | 1.032 | 1.004 | 1.015 | 1.030 | 1.035 |
| HARQ-h | — | 0.995 | 0.988 | — | 0.973 | 1.015 |

Each cell is the predicted loss of the model ÷ the loss of HAR, taking the median of 22 commodities; less than 1 means better than HAR. The 1000-day rolling window is revalued daily; h is the number of forecast days. QLIKE is a loss function more commonly used for volatility prediction, with a heavier penalty for underestimating volatility. Source: Reproduction of Bollerslev, Patton & Quaedvlieg, Journal of Econometrics, 2016.
