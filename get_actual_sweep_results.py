import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np



if __name__ == '__main__':
    # initial_results = Path("sweep_results/20260908_113431")
    # with open(initial_results / "raw_results.json", "r") as f:
    #         initial_results = json.load(f)
    # initial_results
    results_folder = Path("sweep_results/20260918_141216")
    with open(results_folder / "raw_results.json", "r") as f:
        results = json.load(f)
    aggregated = []
    aggregated_mean = []
    datasets = ["tolokers-2", "city-reviews", "artnet-exp", "hm-categories"]
    for r in tqdm(results, desc="Aggregating results"):
        if r["result"] is None:
            continue
        metric = "test_ap_mean" if r["result"].get("test_ap_mean", None) is not None else "test_acc_mean"
        aggregated.append({
            "type": r["checkpoint_label"],
            "cfg": r["combo_key"],
            "dataset": r["dataset"],
            "metric": metric,
            "metric_value": r["result"].get(metric, None)
        })
    df = pd.DataFrame(aggregated).sort_values(by="cfg")
    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]
        dataset_df = dataset_df.sort_values(["cfg", "type"]).reset_index(drop=True)
        full_value = (
            dataset_df
            .assign(
                full_metric_value=lambda df: df["metric_value"].where(df["type"] == "full")
            )
            .groupby(["cfg", "dataset"])["full_metric_value"]
            .transform("max")
        )
        dataset_df["diffs"] = dataset_df["metric_value"] - full_value
        dataset_df = dataset_df.sort_values(by="diffs", ascending=False).reset_index(drop=True)
        dataset_df.to_csv(results_folder / f"aggregated_results_{dataset}.csv", index=False)

    df.to_csv(results_folder / "aggregated_results.csv", index=False)
    print(f"Saved aggregated results to {results_folder / 'aggregated_results.csv'}")
    mean_df = (
        df[df["metric"] == "test_acc_mean"]
        .groupby(["type", "cfg"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "mean_test_acc"})
        .sort_values(by="mean_test_acc", ascending=False)
    )
    mean_df.to_csv(results_folder / "mean_results.csv", index=False)
    print(f"Saved mean results to {results_folder / 'mean_results.csv'}")
