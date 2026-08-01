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
# We use the geo_100_classes checkpoint (geo_similarity prior) rather than
# the plain baseline: it's pretrained on graph-topology-aware synthetic data,
# a closer match to a real social network than the generic prior_bag model,
# and its max_num_classes=100 gives finer quantile resolution than the
# 20-class plain baseline.
#
# NodePFN's train tokens self-attend to each other (O(train_size^2)), so the
# in-context example set is subsampled from the (much larger) train graph via
# --train_sample_size, independent of GPU memory. The query side (val/test)
# is chunked via --query_batch_size to cap the (train+query) attention/GCN
# context -- these are large, hub-heavy social graphs (hundreds of thousands
# of nodes), so --precision bf16 and --batch_size_inference 1 keep peak
# memory down, same as the largest GraphLand datasets.
# ---------------------------------------------------------------------------

for dataset in bluesky_quotes bluesky_replies bluesky_reposts; do
  python -m nodepfn.node_regression --dataset "$dataset" --target all \
    --base_model_path models_ckpts/baseline_100_classes --e 50 --n_bins 100 \
    --train_sample_size 1000 --n_ensemble 8 --runs 5 \
    --query_batch_size 2000 --batch_size_inference 1 --precision bf16
done
