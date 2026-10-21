# 🤖 AI Use Log
> **HackInsight 2026 Disclosure Log**

| Tool & Version | Task / Module | Code Kept or Modified | Responsible Team Member |
| :---           | :---          | :---                  | :---                    |
| **Gemini / ChatGPT** | Scaffolding `smart_energy_predictor.py` pipeline structure | Refactored data generation to 17,520 rows (2 years hourly) and added Open-Meteo API integration. | All Team Members |
| **Gemini / ChatGPT** | Feature engineering (lags, rolling averages) | Verified time-series split logic to prevent future data leakage. | All Team Members |
| **Gemini / ChatGPT** | Data Quality Agent decision logging (`fix`, `fill`, `flag`) | Modified to export structured JSON logs (`quality_log.json`) for auditing. | All Team Members |
| **Gemini / ChatGPT** | Insight Agent (RAG) implementation | Added explicit Gold Table Row ID citations and refusal fallback mechanism. | All Team Members |