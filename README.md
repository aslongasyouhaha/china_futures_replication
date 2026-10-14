# 中国期货市场：金融论文复现

在中国期货数据（2005–2026，六家交易所约 90 个品种）上复现正式发表的金融论文，检验这些结论在中国是否成立。目的是理解中国期货市场，不是构建交易策略。

**总结入口：[china_overview/中国期货市场全貌.md](china_overview/中国期货市场全貌.md)**。它按"市场因子 → 截面溢价 → 时序择时 → 波动率 → 交割前挤仓"五部分，汇总了下列复现的结论，配有论文格式的回归表。

## 复现的论文

每个文件夹对应一篇论文，包含复现脚本、生成报告的 `build_report.py` 和离线报告 `report.html`。完整引用、DOI 和免费版本链接见 [papers.md](papers.md)。

### 市场因子与截面溢价

| 文件夹 | 论文 | 中国的主要结论 |
|---|---|---|
| [carry/](carry/) | Koijen, Moskowitz, Pedersen & Vrugt (2018), *Carry*, Journal of Financial Economics | 截面 carry 夏普 0.91，按实盘条件执行后 0.71，强于海外 |
| [commodity_three_factor/](commodity_three_factor/) | Bakshi, Gao & Rossi (2019), *Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns*, Management Science | 平均、carry、动量三因子都必要；主力合约上动量大部分与 carry 重叠 |
| [value_momentum/](value_momentum/) | Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, Journal of Finance（仅商品部分） | 动量与论文一样强，价值单独为负；价值+动量组合独立于 carry |
| [basis_momentum/](basis_momentum/) | Boons & Prado (2019), *Basis-Momentum*, Journal of Finance | 能预测收益，但被普通动量吸收，不是独立因子 |
| [commodity_skewness/](commodity_skewness/) | Fernandez-Perez, Frijns, Fuertes & Miffre (2018), *The Skewness of Commodity Futures Returns*, Journal of Banking & Finance | 没有偏度溢价；偏度被动量掩盖，2016–2018 年前后方向反转 |
| [kang_rouwenhorst_tang/](kang_rouwenhorst_tang/) | Kang, Rouwenhorst & Tang (2020), *A Tale of Two Premiums*, Journal of Finance | 前 20 名大户买入后价格继续上涨，与论文方向相反，更像知情交易 |
| [fan_zhang/](fan_zhang/) | Fan & Zhang (2020), *The Untold Story of Commodity Futures in China*, Journal of Futures Markets | 期限结构和动量在论文之后的 9 年仍显著；价值样本外显著为负 |
| [szymanowska/](szymanowska/) | Szymanowska, de Roon, Nijman & van den Goorbergh (2014), *An Anatomy of Commodity Futures Risk Premia*, Journal of Finance | 基差分组的现货溢价、期限溢价都更强；单一基差因子解释不了中国的现货溢价 |

### 时序择时

| 文件夹 | 论文 | 中国的主要结论 |
|---|---|---|
| [time_series_momentum/](time_series_momentum/) | Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, Journal of Financial Economics | 论文口径夏普 0.61，按实盘条件执行后 0.31，不显著 |
| [tsmom_huang/](tsmom_huang/) | Huang, Li, Wang & Zhou (2020), *Time Series Momentum: Is It There?*, Journal of Financial Economics | 控制品种固定效应后不显著；论文的自助法有缺陷，改为按月整块重抽 |
| [open_interest/](open_interest/) | Hong & Yogo (2012), *What Does Futures Market Interest Tell Us about the Macroeconomy and Asset Prices?*, Journal of Financial Economics | 基差预测商品整体收益；持仓增长方向一致但偏弱 |
| [intraday_momentum/](intraday_momentum/) | Gao, Han, Li & Zhou (2018), *Market Intraday Momentum*, Journal of Financial Economics | 股指靠倒数第二个半小时；商品扣一跳价差后大多不赚钱 |

### 波动率

| 文件夹 | 论文 | 中国的主要结论 |
|---|---|---|
| [har_rv/](har_rv/) | Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, Journal of Financial Econometrics | HAR 接近长记忆模型，比 EWMA 好约 18%；持续性强于海外 |
| [harq/](harq/) | Bollerslev, Patton & Quaedvlieg (2016), *Exploiting the Errors*, Journal of Econometrics | 测量误差项显著为负；HARQ 改进 5 天、22 天预测 |
| [rough_vol/](rough_vol/) | Gatheral, Jaisson & Rosenbaum (2018), *Volatility Is Rough*, Quantitative Finance | H ≈ 0.06~0.14；20 天预测 RFSV 最优；中国波动率增量明显厚尾 |
| [volatility_managed/](volatility_managed/) | Moreira & Muir (2017), *Volatility-Managed Portfolios*, Journal of Finance | 沪深 300 有效（α 6.4%）；商品有害，因为高波动时期多是牛市 |

### 其他

| 文件夹 | 内容 |
|---|---|
| [china_overview/](china_overview/) | 跨论文的汇总：全貌文档、市场因子、carry/动量/价值的重叠，以及文档中的表格（`tables/`）和图（`figures/`） |
| [common/](common/) | 共用的交易成本工具：合约乘数、最小变动价位、买卖价差估计、日内 VWAP |
| [commodity_squeeze/](commodity_squeeze/) | Zimbidis & Sircar (2026), *Negative Oil & Nickel Squeeze*, arXiv 工作论文（未经同行评审）：模型数值完整复现；中国个人客户交割月前须平仓，压力月近月偏离明显更大、以被压低为主，但幅度远小于海外极端事件 |
| [cta_combo/](cta_combo/) | carry 与时间序列动量组合、波动率预测用于仓位配平的研究（非论文复现，留作参考） |

## 数据

仓库**不包含数据**。原始数据包括第三方分钟行情、交易所官方日线与结算价、Tushare 的持仓排名与宏观数据，不能公开分发。

数据目录由环境变量 `CTA_RAW_DATA` 指定（见 [paths.py](paths.py)）：

```
# Windows（PowerShell，设置后需重开终端）
[Environment]::SetEnvironmentVariable('CTA_RAW_DATA', 'D:\data\raw_data', 'User')
# Linux / macOS
export CTA_RAW_DATA=/data/raw_data
```

未设置时退回到本仓库旁边的 `raw_data/`（即 `<上一级目录>/raw_data`），并给出提示。目录结构为：

```
raw_data/
├── trading_calendar.parquet
├── products_summary.parquet
├── 参考数据/                 南华指数、Shibor、CPI、PMI、汇率、合约信息等
└── {交易所}/{品种}/          如 SHFE/RB
    ├── daily_contracts.parquet   各合约日线（官方结算价、成交量、持仓量）
    ├── daily_main.parquet        主力连续（后复权）
    ├── daily_index.parquet       品种指数（持仓加权）
    ├── minute_main.parquet       主力连续 1 分钟
    └── holding_rank.parquet      前 20 名会员持仓
```

`output/` 文件夹（各步计算结果）由原始数据衍生，也不在仓库中，可用各文件夹的脚本重新生成。`report.html` 是自包含的离线报告，可以直接打开查看结果。

## 运行

```
pip install -r requirements.txt
python carry/carry.py          # 计算
python carry/build_report.py   # 生成 report.html
```

部分研究依赖 `common/` 的交易成本数据，需要先运行 `common/` 下的脚本。

## 说明

- 先按论文原方法复现，再做中国的调整（合约选择、流动性门槛、交易成本等），两者在报告中分开列出。
- 只复现正式发表在期刊上的论文（commodity_squeeze 除外，已注明）。
- 论文 PDF 不在仓库中（版权属于出版社或作者），可通过 [papers.md](papers.md) 中的链接获取。
