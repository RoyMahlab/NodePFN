# !/bin/bash
# ---------------------------------------------------------------------------
# GraphLand classification baselines (multiclass + binary), excluding the
# `web-*` datasets (web-topics, web-fraud).
#
# Each dataset is matched to its most-similar dataset in run_baseline.sh and
# its hyperparameters are adapted from it. The main knob, --smoothing_steps,
# is driven by graph homophily (see benchmarks/graphland_stats.csv):
#   high homophily -> more smoothing,  low/heterophilous -> little or none.
#
# GraphLand is wired into load_dataset() in nodepfn/dataset.py (loaded via its
# PyGDataset). Data is resolved from the GraphLand repo itself, so --data_dir is
# not needed. Default split is RL (10/10/80); append ":RH"/":TH" to the dataset
# name to use another transductive split, e.g. --dataset artnet-exp:RH
# ---------------------------------------------------------------------------

# tolokers-2  (binary, deg 88, homophily LOW 0.09, 16 feats)
#   match: tolokers (it IS the GraphLand variant of tolokers)
#   -> inherit dim_reduction none + n_ensemble 4
python -m nodepfn.node_classification --dataset tolokers-2 --base_model_path=models_ckpts/baseline --dim_reduction none --n_components 25 --runs=5 --smoothing_steps 2 --n_ensemble 4

# city-reviews  (binary, 148k nodes, homophily HIGH 0.59, 37 feats)
#   match: amazon-computer (homophilous) -> heavy smoothing
#   -> randomized svd + modest ensemble for the larger graph
python -m nodepfn.node_classification --dataset city-reviews --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 3 --n_ensemble 8 --svd_algorithm randomized --pipeline_gpus 2 --precision bf16

# artnet-exp  (binary, 50k nodes, homophily MEDIUM 0.16, 75 feats)
#   match: chameleon (low-med homophily, feature-rich) -> light smoothing,
#   many components, n_ensemble 16
python -m nodepfn.node_classification --dataset artnet-exp --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25 --runs=5 --smoothing_steps 1 --n_ensemble 16

# hm-categories  (multiclass 21cls, deg 461 very dense, homophily LOW 0.08, 35 feats)
#   match: coauthor-cs (many classes) -> n_components 25
#   -> low homophily + huge density: cut smoothing to 1, randomized svd, smaller ensemble
python -m nodepfn.node_classification --dataset hm-categories --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25 --runs=5 --smoothing_steps 1 --n_ensemble 8 --svd_algorithm randomized

# pokec-regions  (multiclass 183cls, 1.6M nodes, homophily MEDIUM 0.42, 56 feats)
#   match: coauthor-cs (many classes, homophilous) -> smoothing 2
#   WARNING: very large (1.6M nodes / 22M edges) -> likely OOM on GPU.
#   Scale-safe: randomized svd, n_ensemble 4, --cpu. Drop --cpu if it fits on GPU.
python -m nodepfn.node_classification --dataset pokec-regions --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25 --runs=5 --smoothing_steps 2 --n_ensemble 4 --svd_algorithm randomized --cpu
