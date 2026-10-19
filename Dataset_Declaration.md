# 📋 Dataset Declaration
> **HackInsight 2026 Submission**

## 1. Chosen Data Sources
* **Source 1 (Weather Data)**: Open-Meteo Historical Weather API
  * **Publisher**: Open-Meteo
  * **URL**: `https://archive-api.open-meteo.com/`
  * **Licence**: Creative Commons Attribution 4.0 (CC-BY 4.0)
  * **Update Frequency**: Hourly
  * **Location Focus**: Lahore, Pakistan (`31.5204° N, 74.3587° E`)

* **Source 2 (Demand Data)**: Regional Hourly Electricity Load Benchmarks
  * **Publisher**: Open Power System Data / Energy Benchmarks
  * **Licence**: Open Data Commons Attribution License (ODC-By)
  * **Update Frequency**: Hourly

## 2. Dataset Volume & Verification
* **Total Volume**: 17,520 rows (2 full years of continuous hourly observations: 2022-01-01 to 2023-12-31).
* **Integrity Check**: Every raw dataset fetch is saved in `data/raw/` and hashed via SHA-256 in `manifest.csv`.

## 3. Declared Baseline
* **Baseline Definition**: Naive 24-hour persistence lag model (`demand_lag_24`).
* **Rule**: Fixed prior to model training; predictions assume tomorrow's same-hour demand equals today's observed demand.