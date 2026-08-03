# !/bin/bash
# ---------------------------------------------------------------------------
# Standalone classification datasets requested alongside run_graphland_baseline.sh
# and run_bluesky_baseline.sh (not part of run_baseline.sh's default sweep).
# ---------------------------------------------------------------------------

# ogbn-arxiv (169k nodes, 40 classes, OGB's fixed split has ~91k train nodes --
# NodePFN's train tokens self-attend to each other (O(train_size^2)), so the
# in-context set is capped via --train_sample_size regardless of GPU memory.
# 20000 is the largest value confirmed to fit on a single 47GB A6000 with these
# ensemble/precision settings; 24000 also fit in isolation, 32000 OOM'd.
python -m nodepfn.node_classification --dataset ogbn-arxiv --base_model_path=models_ckpts/baseline_100_classes --dim_reduction none --runs=5 --smoothing_steps 2 --n_ensemble 8 --train_sample_size 20000 --query_batch_size 5000 --precision bf16 --batch_size_inference 1

# dblp (full DBLP citation network, 17716 nodes, 4 classes) -- same tuning as
# run_baseline.sh's entry. Auto-downloads via torch_geometric's CitationFull.
python -m nodepfn.node_classification --dataset dblp --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25 --runs=5 --smoothing_steps 3


# cora-full (19793 nodes, 70 classes, 8710 raw bag-of-words features -- needs
# reduction). Many classes -> baseline_100_classes checkpoint + randomized SVD
# for the high raw feature count, matching the graphland many-class precedent.
python -m nodepfn.node_classification --dataset cora-full --base_model_path=models_ckpts/baseline_100_classes --dim_reduction tsvd --n_components 25 --svd_algorithm randomized --runs=5 --smoothing_steps 3 --n_ensemble 8 --train_sample_size 1000 --precision bf16 --batch_size_inference 1
