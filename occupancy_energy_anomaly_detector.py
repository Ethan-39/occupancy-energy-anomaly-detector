"""Flag unusual building energy use after fitting occupancy-aware hourly baselines."""

import argparse
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
INPUT_PATH = DATA_DIR / "building_hourly.csv"
TRAIN_DAYS = 7
ROBUST_Z = 3.5
MINIMUM_RESIDUAL_THRESHOLD_KWH = 0.35


def write_example(path=INPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261012)
    start = datetime(2026, 9, 1)
    rows = []
    for day in range(14):
        for hour in range(24):
            if 8 <= hour <= 17:
                occupancy = rng.randint(7, 21)
            elif 18 <= hour <= 22:
                occupancy = rng.randint(0, 7)
            else:
                occupancy = rng.randint(0, 2)
            hourly_base = 0.9 + 0.16 * math.sin((hour - 6) * math.pi / 12) + 0.03 * math.cos(hour * math.pi / 6)
            energy = hourly_base + 0.095 * occupancy + rng.gauss(0, 0.075)
            truth = "normal"
            if day == 9 and hour in (2, 3, 4):
                energy += 2.6
                truth = "synthetic_spike"
            rows.append({
                "timestamp": (start + timedelta(days=day, hours=hour)).isoformat(timespec="minutes"),
                "occupancy_count": occupancy,
                "energy_kwh": f"{energy:.4f}",
                "scenario_truth": truth,
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "occupancy_count", "energy_kwh", "scenario_truth"])
        writer.writeheader()
        writer.writerows(rows)


def read_records(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for line_number, row in enumerate(rows, start=2):
        try:
            records.append({
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "occupancy_count": float(row["occupancy_count"]),
                "energy_kwh": float(row["energy_kwh"]),
                "scenario_truth": row.get("scenario_truth", "unknown"),
                "source_row": line_number,
            })
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"input line {line_number} has invalid timestamp, occupancy, or energy") from error
        if records[-1]["occupancy_count"] < 0 or records[-1]["energy_kwh"] < 0:
            raise ValueError(f"input line {line_number} has a negative occupancy or energy value")
    return records


def fit_hour_model(points):
    """Fit energy = intercept + slope * occupancy for one hour of day."""
    if not points:
        return 0.0, 0.0, 0.0
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator) if denominator else 0.0
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in points]
    center = statistics.median(residuals)
    mad = statistics.median(abs(value - center) for value in residuals)
    return intercept, slope, mad


def analyze_records(records, train_days=TRAIN_DAYS, robust_z=ROBUST_Z,
                    threshold_floor=MINIMUM_RESIDUAL_THRESHOLD_KWH):
    if len(records) < 48:
        raise ValueError("at least 48 hourly records are needed for a training and review period")
    ordered = sorted(records, key=lambda row: row["timestamp"])
    split_at = ordered[0]["timestamp"] + timedelta(days=train_days)
    train = [row for row in ordered if row["timestamp"] < split_at]
    test = [row for row in ordered if row["timestamp"] >= split_at]
    if not train or not test:
        raise ValueError("both chronological training and test periods must contain records")

    models = {}
    for hour in range(24):
        points = [(row["occupancy_count"], row["energy_kwh"])
                  for row in train if row["timestamp"].hour == hour]
        models[hour] = fit_hour_model(points)

    scored = []
    for row in ordered:
        hour = row["timestamp"].hour
        intercept, slope, mad = models[hour]
        predicted = intercept + slope * row["occupancy_count"]
        residual = row["energy_kwh"] - predicted
        threshold = max(robust_z * 1.4826 * mad, threshold_floor)
        scored.append({
            "timestamp": row["timestamp"].isoformat(timespec="minutes"),
            "partition": "train" if row["timestamp"] < split_at else "test",
            "occupancy_count": row["occupancy_count"],
            "energy_kwh": round(row["energy_kwh"], 4),
            "expected_kwh_for_occupancy": round(predicted, 4),
            "residual_kwh": round(residual, 4),
            "threshold_kwh": round(threshold, 4),
            "flagged": abs(residual) > threshold,
            "scenario_truth": row["scenario_truth"],
        })

    test_scores = [row for row in scored if row["partition"] == "test"]
    truth = [row["scenario_truth"] == "synthetic_spike" for row in test_scores]
    predicted = [row["flagged"] for row in test_scores]
    tp = sum(actual and flag for actual, flag in zip(truth, predicted))
    fp = sum(not actual and flag for actual, flag in zip(truth, predicted))
    fn = sum(actual and not flag for actual, flag in zip(truth, predicted))
    tn = sum(not actual and not flag for actual, flag in zip(truth, predicted))
    report = {
        "input_rows": len(ordered),
        "training_rows": len(train),
        "test_rows": len(test),
        "split": f"first {train_days} days train; later records test",
        "method": "hour-of-day ordinary least squares energy baseline using occupancy; robust MAD residual threshold",
        "threshold_rule": f"max({robust_z} * 1.4826 * training residual MAD, {threshold_floor} kWh)",
        "test_flags": sum(predicted),
        "synthetic_scenario_evaluation": {
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "note": "scenario_truth is used only after scoring to evaluate this generated example, never as a model feature",
        },
        "limitations": "Synthetic screening baseline only; flags do not establish waste, fault, or a cause.",
    }
    return report, scored


def run(input_path=INPUT_PATH, train_days=TRAIN_DAYS):
    if not input_path.exists():
        write_example(input_path)
    report, scored = analyze_records(read_records(input_path), train_days=train_days)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["timestamp", "partition", "occupancy_count", "energy_kwh", "expected_kwh_for_occupancy",
              "residual_kwh", "threshold_kwh", "flagged", "scenario_truth"]
    with (RESULTS_DIR / "scored_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
    with (RESULTS_DIR / "review_flags.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in scored if row["partition"] == "test" and row["flagged"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    args = parser.parse_args()
    if args.train_days < 1:
        parser.error("--train-days must be positive")
    run(args.input, args.train_days)


if __name__ == "__main__":
    main()
