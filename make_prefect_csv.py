import csv
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
fieldnames = [
    "flow_run_id",
    "flow_name",
    "state",
    "start_time",
    "end_time",
    "total_run_time_seconds",
    "rows_processed"
]

rows = []
for i in range(24, 0, -1):
    run_time = now - timedelta(hours=i)
    rows.append({
        "flow_run_id": f"run-20261019-00{i:02d}",
        "flow_name": "Smart Energy Pipeline",
        "state": "Completed",
        "start_time": run_time.isoformat(),
        "end_time": (run_time + timedelta(seconds=14)).isoformat(),
        "total_run_time_seconds": 14.2,
        "rows_processed": 17520
    })

with open("prefect_runs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("✅ prefect_runs.csv created successfully!")