---
title: Energy Demand Predictor
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: smart_energy_predictor.py
pinned: false
license: mit
---
# ⚡ Smart Energy Demand Predictor
> **HackInsight 2026 Data-to-Decision Project**

An end-to-end machine learning pipeline and interactive Streamlit dashboard forecasting hourly urban electricity demand in Pakistan using historical consumption and weather data.

---

**Team Members & Roles**
Zeesha Zaki (1st Year CE) – Project Lead & ML Developer 
Responsibilities: Machine learning pipeline architecture, Random Forest Regressor training, Open-Meteo API integration, and core script execution (smart_energy_predictor.py, make_data.py).

Muhammad Rouhan Uddin (2nd Year CS) – Frontend & UI Developer
Responsibilities: Streamlit dashboard interface design, metric visualization cards (MAE, RMSE, $R^2$), and interactive forecast charts.

Muhammad Umer Farooq (2nd Year CS) – Data & Quality Analyst
Responsibilities: Input data validation, automated data quality agent logging (quality_log.json), and cryptographic hash verification (manifest.csv).

Muhammad Zabir Uddin (3rd Year CS) – Documentation & Submission CoordinatorResponsibilities: Compliance documentation (AI_Use_Log.md, Dataset_Declaration.md), executive 1-page Summary.pdf, repository version tagging (v1.0-final), and media demonstration.

**Deployed Application & Links**
Live Dashboard Application:https://energy-dashboard-4.streamlit.app/
 GitHub Repository: https://github.com/zeesha-zaki/energy-dashboard

## 📌 Problem Analysis & Pakistan Context
* **Target User:** Regional Electricity Grid Dispatchers & Energy Planners.
* **Decision:** Optimizing generator scheduling and peak load shedding prevention.
* **Pakistan Context:** Urban grids (e.g., Lahore/Karachi) experience sharp daily peak loads between **17:00 and 20:00 PKT**, aggravated by severe summer heatwaves. Accurate hourly forecasts reduce costly emergency diesel generation and unplanned power outages.

---

## 📊 Dataset & Provenance
* **Volume:** 17,520 rows (2 full years of hourly observations).
* **Sources:**
  1. **Open-Meteo Historical Weather API** (Lahore Coordinates: `31.5204`, `74.3587`) — Free Open Data.
  2. **Pakistan Regional Load Benchmarks**.
* **Auditability:** Every raw download is hashed via SHA-256 and logged into `manifest.csv`.

---

## 🛠️ Architecture & Features

```text
energy-dashboard/
├── .gitignore
├── AI_Use_Log.md
├── README.md
├── manifest.csv
├── requirements.txt
├── smart_energy_predictor.py
└── data/
    ├── cleaned/
    │   └── .gitkeep
    ├── processed/
    │   └── .gitkeep
    └── raw/
        └── .gitkeep