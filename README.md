# Chinese Futures Markets: Finance Paper Replications

Replicate published finance papers using Chinese futures data (2005–2026, about 90 instruments across six exchanges) to test whether these conclusions are valid in China. The purpose is to understand the Chinese futures market, rather than to develop a trading strategy.

**Summary entry: [china_overview/overview.md](china_overview/overview.md)**. It summarizes the following replication findings in five parts: "market factors → cross-section premium → timing → volatility → pre-delivery squeeze", with a regression table in the format of a paper.

## Replicated papers

Each folder corresponds to a paper and contains the reproduction script, report builder `build_report.py` and offline report `report.html`. Full citation, DOI and link to free version see [papers.md](papers.md).

### Market factors and cross-sectional premium

| folder | Paper | Main conclusions for China |
|---|---|---|
| [carry/](carry/) | Koijen, Moskowitz, Pedersen & Vrugt (2018), *Carry*, Journal of Financial Economics | Cross-sectional carry Sharpe 0.91, 0.71 after execution according to realistic execution assumptions, stronger than overseas |
| [commodity_three_factor/](commodity_three_factor/) | Bakshi, Gao & Rossi (2019), *Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns*, Management Science | The three factors of average, carry, and momentum are all necessary; most of the momentum on the dominant contract overlaps with carry. |
| [value_momentum/](value_momentum/) | Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, Journal of Finance (commodity section only) | Momentum is as strong as in the paper, value alone is negative; value+momentum combination is independent of carry |
| [basis_momentum/](basis_momentum/) | Boons & Prado (2019), *Basis-Momentum*, Journal of Finance | Can predict returns, but is absorbed by ordinary momentum and is not an independent factor |
| [commodity_skewness/](commodity_skewness/) | Fernandez-Perez, Frijns, Fuertes & Miffre (2018), *The Skewness of Commodity Futures Returns*, Journal of Banking & Finance | There is no skewness premium; skewness is masked by volume, and the direction reverses around 2016–2018 |
| [kang_rouwenhorst_tang/](kang_rouwenhorst_tang/) | Kang, Rouwenhorst & Tang (2020), *A Tale of Two Premiums*, Journal of Finance | The price continues to rise after the top 20 large traders buy, which is opposite to the direction of the paper and more like informed trading |
| [fan_zhang/](fan_zhang/) | Fan & Zhang (2020), *The Untold Story of Commodity Futures in China*, Journal of Futures Markets | Term structure and momentum remain significant 9 years after the paper; value out-of-sample is significantly negative |
| [szymanowska/](szymanowska/) | Szymanowska, de Roon, Nijman & van den Goorbergh (2014), *An Anatomy of Commodity Futures Risk Premia*, Journal of Finance | The spot premium and term premium of the basis group are stronger; a single basis factor cannot explain China’s spot premium. |

### Time-series timing

| folder | Paper | Main conclusions for China |
|---|---|---|
| [time_series_momentum/](time_series_momentum/) | Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, Journal of Financial Economics | The paper-style specification Sharpe is 0.61, and after execution according to the realistic execution assumptions, it is 0.31, not significant. |
| [tsmom_huang/](tsmom_huang/) | Huang, Li, Wang & Zhou (2020), *Time Series Momentum: Is It There?*, Journal of Financial Economics | It is not significant after controlling for instrument fixed effects; the bootstrap method of the paper is flawed, so it is changed to resampling complete monthly cross-sections in blocks. |
| [open_interest/](open_interest/) | Hong & Yogo (2012), *What Does Futures Market Interest Tell Us about the Macroeconomy and Asset Prices?*, Journal of Financial Economics | Basis forecasts overall commodity returns; position growth direction is consistent but weak |
| [intraday_momentum/](intraday_momentum/) | Gao, Han, Li & Zhou (2018), *Market Intraday Momentum*, Journal of Financial Economics | The stock index relies on the penultimate half hour; most commodities do not make money after deducting one-tick spread. |

### Volatility

| folder | Paper | Main conclusions for China |
|---|---|---|
| [har_rv/](har_rv/) | Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, Journal of Financial Econometrics | HAR is close to the long memory model, about 18% better than EWMA; persistence is stronger than overseas |
| [harq/](harq/) | Bollerslev, Patton & Quaedvlieg (2016), *Exploiting the Errors*, Journal of Econometrics | Measurement error term is significantly negative; HARQ improves 5-day, 22-day forecasts |
| [rough_vol/](rough_vol/) | Gatheral, Jaisson & Rosenbaum (2018), *Volatility Is Rough*, Quantitative Finance | H ≈ 0.06~0.14; 20-day forecast RFSV is optimal; China’s volatility increment is obviously heavy-tailed |
| [volatility_managed/](volatility_managed/) | Moreira & Muir (2017), *Volatility-Managed Portfolios*, Journal of Finance | Effective for CSI 300 (alpha 6.4%); volatility management hurts commodity performance because periods of high volatility are mostly bull markets |

### Others

| folder | content |
|---|---|
| [china_overview/](china_overview/) | Summary across papers: full picture document, market factors, carry/momentum/value overlap, and tables (`tables/`) and figures (`figures/`) in the document |
| [common/](common/) | Shared transaction cost tools: contract multiplier, minimum tick, bid-ask spread estimate, intraday VWAP |
| [commodity_squeeze/](commodity_squeeze/) | Zimbidis & Sircar (2026), *Negative Oil & Nickel Squeeze*, arXiv working paper (not peer-reviewed): The numerical model replication is complete; Chinese individual customers must close their positions before the delivery month, and the nearby-contract deviation in the pressure month is significantly larger, mainly being depressed, but the magnitude is much smaller than overseas extreme events |
| `cta_combo` (legacy exploration; not included in this repository) | Research on carry, time series momentum combination, and volatility prediction for position balancing (exploratory analysis rather than a paper replication) |

## Data

The reference-data directory is named `参考数据/` on disk; this identifier is retained to match the existing data-loading code.

The repository **does not contain data**. The original data includes third-party minute market prices, the exchange's official daily and settlement prices, Tushare's position rankings and macro data, and cannot be distributed publicly.

The data directory is specified by the environment variable `CTA_RAW_DATA` (see [paths.py](paths.py)):

```
# Windows (PowerShell, you need to restart the terminal after setting)
[Environment]::SetEnvironmentVariable('CTA_RAW_DATA', 'D:\data\raw_data', 'User')
# Linux / macOS
export CTA_RAW_DATA=/data/raw_data
```

If it is not set, the code falls back to `raw_data/` alongside this repository (i.e. `<upper-level directory>/raw_data`) and give a prompt. The directory structure is:

```
raw_data/
├── trading_calendar.parquet
├── products_summary.parquet
├── 参考数据/                  Nanhua indices, Shibor, CPI, PMI, exchange rates, contract metadata
└── {Exchange}/{instrument}/ such as SHFE/RB
    ├── daily_contracts.parquet Daily data for each contract (official settlement price, trading volume, open interest)
    ├── daily_main.parquet Roll-adjusted continuous dominant-contract prices
    ├── daily_index.parquet instrument index (open-interest weighted)
    ├── minute_main.parquet Continuous dominant-contract one-minute prices
    └── holding_rank.parquet Position rankings for the top 20 members
```

The `output/` folder (the calculation results of each step) is derived from the original data and is not in the repository. It can be regenerated by the script of each folder. `report.html` is a self-contained offline report that can be opened directly to view the results.

## Run

```
pip install -r requirements.txt
python carry/carry.py # Calculation
python carry/build_report.py # Generate report.html
```

Some studies rely on the transaction cost data of `common/` and need to run the script under `common/` first.

## Description

- First reproduce the original method of the paper, and then make adjustments for China (contract selection, liquidity threshold, transaction costs, etc.). The two are listed separately in the report.
- Only reproduce papers officially published in journals (except for commodity_squeeze, where noted).
- The paper PDF is not in the repository (copyright belongs to the publisher or author) and is available through the link in [papers.md](papers.md).
