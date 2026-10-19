"""
Rebuild Data Layer CLI Script
HackInsight 2026 - Deliverable 1
"""

from smart_energy_predictor import (
    collect_energy_data,
    save_raw_data,
    record_manifest,
    clean_data,
    prepare_processed_data,
)
from datetime import datetime, timezone

def main():
    print("🔄 Running 'make data': Rebuilding raw, cleaned, and processed layers...")
    
    # 1. Fetch Raw Data
    raw_df = collect_energy_data()
    fetch_time = datetime.now(timezone.utc).isoformat()
    sha256_hash = save_raw_data(raw_df, "raw_energy.csv")
    
    record_manifest(
        source="Open-Meteo & Load Benchmarks",
        query="hourly_demand_weather_pakistan",
        fetch_time=fetch_time,
        sha256=sha256_hash,
        licence="CC-BY 4.0 / ODC-By"
    )
    print(f"✅ Raw layer saved to data/raw/raw_energy.csv (SHA-256: {sha256_hash[:10]}...)")
    
    # 2. Clean Data & Apply Quality Agent Rules
    cleaned_df, quality_decisions = clean_data(raw_df)
    print(f"✅ Cleaned layer saved to data/cleaned/cleaned_energy.csv ({len(quality_decisions)} quality decisions logged)")
    
    # 3. Build Processed Layer
    processed_df = prepare_processed_data(cleaned_df)
    print(f"✅ Processed gold layer saved to data/processed/processed_dataset.csv ({len(processed_df):,} rows ready)")

if __name__ == "__main__":
    main()