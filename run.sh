# !/bin/bash
# Cora
python -m nodepfn.node_classification  --dataset cora --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 4
# Citeseer
python -m nodepfn.node_classification  --dataset citeseer --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 2
# Pubmed
python -m nodepfn.node_classification  --dataset pubmed --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 2  
# Air-USA
python -m nodepfn.node_classification  --dataset air-usa --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --label_num_per_class 20 --n_ensemble 8
#Air-Europe
python -m nodepfn.node_classification  --dataset air-europe --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 1 --label_num_per_class 20 --n_ensemble 32
# Air-Brazil
python -m nodepfn.node_classification  --dataset air-brazil --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --label_num_per_class 20 --n_ensemble 32
# WikiCS
python -m nodepfn.node_classification  --dataset wikics --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 2 --n_ensemble 32 --svd_algorithm arpack --cpu
# Computers
python -m nodepfn.node_classification  --dataset amazon-computer --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 3 --svd_algorithm arpack
# Photo
python -m nodepfn.node_classification  --dataset amazon-photo --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 3 --svd_algorithm arpack
# DBLP
python -m nodepfn.node_classification  --dataset dblp --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3
# Coauthor-CS
python -m nodepfn.node_classification  --dataset coauthor-cs  --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 2 --n_ensemble 32  --svd_algorithm arpack
# Coauthor-Physics
python -m nodepfn.node_classification  --dataset coauthor-physics --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 4  --n_ensemble 4 --svd_algorithm randomized
# Deezer
python -m nodepfn.node_classification  --dataset deezer --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 50  --runs=5 --smoothing_steps 0 --n_ensemble 32
# Cornell
python -m nodepfn.node_classification  --dataset cornell --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 0 --n_ensemble=16 --svd_algorithm randomized
# Texas
python -m nodepfn.node_classification  --dataset texas --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 20  --runs=10  --smoothing_steps 0
# Wisconsin
python -m nodepfn.node_classification  --dataset wisconsin --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5  --smoothing_steps 0 --n_ensemble=32 --svd_algorithm randomized
# Chameleon
python -m nodepfn.node_classification  --dataset chameleon --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5  --smoothing_steps 0 --n_ensemble=16
# Actor
python -m nodepfn.node_classification  --dataset actor --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 10  --runs=5  --smoothing_steps 0 --n_ensemble=32
# Minesweeper
python -m nodepfn.node_classification  --dataset minesweeper --base_model_path=models_ckpts/baseline --dim_reduction none --n_components 15  --runs=5 --smoothing_steps 1 --n_ensemble 32
# Tolokers
python -m nodepfn.node_classification  --dataset tolokers --base_model_path=models_ckpts/baseline --dim_reduction none --n_components 25  --runs=5 --smoothing_steps 2 --n_ensemble 4
# Amazon-Ratings
python -m nodepfn.node_classification  --dataset amazon-ratings --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 20  --runs=3 --smoothing_steps 3 --n_ensemble 8
# Questions
python -m nodepfn.node_classification  --dataset questions --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --n_ensemble 1
# Squirrel
python -m nodepfn.node_classification  --dataset squirrel --base_model_path=models_ckpts/baseline --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 2 --svd_algorithm arpack