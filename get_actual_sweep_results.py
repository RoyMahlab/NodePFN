import json
import pandas as pd
from tqdm import tqdm
import numpy as np



if __name__ == '__main__':
    with open("sweep_results/20260908_113431/raw_results.json", "r") as f:
        results = json.load(f)
    aggregated = []
    aggregated_mean = []
    datasets = ["cora-full", "ogbn-arxiv", "dblp"]
    for r in tqdm(results, desc="Aggregating results"):
        aggregated.append({
            "type": r["checkpoint_label"],
            "cfg": r["combo_key"],
            "dataset": r["dataset"],
            "metric": "test_acc_mean",
            "metric_value": r["result"].get("test_acc_mean", None)
        })
    df = pd.DataFrame(aggregated).sort_values(by="cfg")
    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]
        dataset_df = dataset_df.sort_values(["cfg", "type"]).reset_index(drop=True)
        diffs = dataset_df["metric_value"].iloc[::2].values - dataset_df["metric_value"].iloc[1::2].values
        neg_diff = -1 * diffs
        dataset_df["diffs"] = np.column_stack([diffs, neg_diff]).ravel()
        dataset_df = dataset_df.sort_values(by="diffs", ascending=False).reset_index(drop=True)
        dataset_df.to_csv(f"sweep_results/20260908_113431/aggregated_results_{dataset}.csv", index=False)

    df.to_csv("sweep_results/20260908_113431/aggregated_results.csv", index=False)
    print(f"Saved aggregated results to sweep_results/20260908_113431/aggregated_results.csv")
    mean_df = (
        df[df["metric"] == "test_acc_mean"]
        .groupby(["type", "cfg"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "mean_test_acc"})
        .sort_values(by="mean_test_acc", ascending=False)
    )
    mean_df.to_csv("sweep_results/20260908_113431/mean_results.csv", index=False)
    print(f"Saved mean results to sweep_results/20260908_113431/mean_results.csv")
