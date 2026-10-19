# ⚡ Smart Energy Demand Predictor
> **HackInsight 2026 Data-to-Decision Project**

An end-to-end machine learning pipeline and interactive Streamlit dashboard forecasting hourly urban electricity demand in Pakistan using historical consumption and weather data.

---

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