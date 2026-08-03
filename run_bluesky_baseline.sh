# !/bin/bash
# ---------------------------------------------------------------------------
# BlueSky social-network regression baselines (quotes / replies / reposts).
#
# Unlike GraphLand, BlueSky is inductive: train/val/test are three separate
# graphs from disjoint time windows (own nodes, edges, features), loaded via
# nodepfn.dataset.load_bluesky_splits(). The target is continuous (log1p
# median engagement count), so nodepfn.node_regression solves it by
# discretizing into quantile bins, running NodePFNClassifier on the bins, and
# decoding the predicted class distribution back into a continuous value via
# a probability-weighted average of per-bin training means.
#
# NodePFN's train tokens self-attend to each other (O(train_size^2)), so the
# in-context example set is subsampled from the (much larger) train graph via
# --train_sample_size, independent of GPU memory. The query side (val/test)
# is chunked via --query_batch_size to cap the (train+query) attention/GCN
# context -- these are large, hub-heavy social graphs (hundreds of thousands
# of nodes), so --precision bf16 and --batch_size_inference 1 keep peak
# memory down, same as the largest GraphLand datasets.
#
# One flat command per dataset (no shell loop): log_regression_to_wandb.py
# parses this file line-by-line, same convention as run_graphland_baseline.sh.
# ---------------------------------------------------------------------------

python -m nodepfn.node_regression --dataset bluesky_quotes --target all --base_model_path models_ckpts/baseline_100_classes --n_bins 100 --train_sample_size 1000 --n_ensemble 8 --runs 5 --query_batch_size 2000 --batch_size_inference 1 --precision bf16

python -m nodepfn.node_regression --dataset bluesky_replies --target all --base_model_path models_ckpts/baseline_100_classes --n_bins 100 --train_sample_size 1000 --n_ensemble 8 --runs 5 --query_batch_size 2000 --batch_size_inference 1 --precision bf16

python -m nodepfn.node_regression --dataset bluesky_reposts --target all --base_model_path models_ckpts/baseline_100_classes --n_bins 100 --train_sample_size 1000 --n_ensemble 8 --runs 5 --query_batch_size 2000 --batch_size_inference 1 --precision bf16