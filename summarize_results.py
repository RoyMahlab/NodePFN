#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_dataset_name(filename: str) -> str:
    base = os.path.basename(filename)
    if base.endswith(".csv"):
        base = base[:-4]
    return base.split("_nodepfn_results_")[0]


def read_metrics(csv_path: str) -> Tuple[float, float]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
        if row is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        return float(row["test_accuracy_mean"]), float(row["test_accuracy_std"])


def collect_results(results_dir: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    baseline_dir = os.path.join(results_dir, "baseline")
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"Missing baseline folder: {baseline_dir}")

    baseline = {}
    for name in os.listdir(baseline_dir):
        if not name.endswith(".csv"):
            continue
        csv_path = os.path.join(baseline_dir, name)
        dataset = parse_dataset_name(name)
        baseline[dataset] = read_metrics(csv_path)

    other_values: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for folder in os.listdir(results_dir):
        if folder == "baseline":
            continue
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for name in os.listdir(folder_path):
            if not name.endswith(".csv"):
                continue
            csv_path = os.path.join(folder_path, name)
            dataset = parse_dataset_name(name)
            other_values[dataset].append(read_metrics(csv_path))

    other: Dict[str, Tuple[float, float]] = {}
    for dataset, values in other_values.items():
        means = [value[0] for value in values]
        stds = [value[1] for value in values]
        other[dataset] = (sum(means) / len(means), sum(stds) / len(stds))

    return baseline, other


def format_metric(value: Tuple[float, float]) -> str:
    mean, std = value
    return f"{mean:.2f} +- {std:.2f}"


def compute_overall_average(values: Dict[str, Tuple[float, float]]) -> float:
    if not values:
        return float("nan")
    means = [value[0] for value in values.values()]
    return sum(means) / len(means)


def build_summary(baseline: Dict[str, Tuple[float, float]], other: Dict[str, Tuple[float, float]]) -> Tuple[List[str], List[str], List[str]]:
    datasets = sorted(set(baseline.keys()) & set(other.keys()))
    header = ["row"] + datasets + ["overall_average"]

    baseline_row = ["baseline"]
    other_row = ["other"]

    for dataset in datasets:
        baseline_row.append(format_metric(baseline[dataset]))
        other_row.append(format_metric(other[dataset]))

    baseline_row.append(f"{compute_overall_average({k: baseline[k] for k in datasets}):.2f}")
    other_row.append(f"{compute_overall_average({k: other[k] for k in datasets}):.2f}")

    return header, baseline_row, other_row


def write_summary(output_path: str, header: List[str], baseline_row: List[str], other_row: List[str]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(baseline_row)
        writer.writerow(other_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NodePFN results into a two-row CSV.")
    parser.add_argument("--results-dir", default="results", help="Path to results folder.")
    parser.add_argument("--output", default="results/summary.csv", help="Output CSV path.")
    args = parser.parse_args()

    baseline, other = collect_results(args.results_dir)
    header, baseline_row, other_row = build_summary(baseline, other)
    write_summary(args.output, header, baseline_row, other_row)


if __name__ == "__main__":
    main()
