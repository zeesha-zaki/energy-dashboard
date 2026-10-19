# ⚡ Smart Energy Demand Predictor
> **HackInsight 2026 Project**

An end-to-end machine learning pipeline and interactive Streamlit dashboard that forecasts hourly electricity demand using historical consumption patterns and weather data.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.11 / 3.12
* **Dashboard:** Streamlit
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`), Pandas, NumPy
* **Data Provenance:** Cryptographic SHA-256 hashing & CSV audit logging (`manifest.csv`)

---

## 📁 Project Structure

```text
energy-dashboard/
├── .gitignore
├── LICENSE
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