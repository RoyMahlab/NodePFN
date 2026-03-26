# !/bin/bash
# Cora
python nodepfn/node_classification.py  --dataset cora --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 4
# Citeseer
python nodepfn/node_classification.py  --dataset citeseer --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 2
# Pubmed
python nodepfn/node_classification.py  --dataset pubmed --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 2  
# Air-USA
python nodepfn/node_classification.py  --dataset air-usa --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --label_num_per_class 20 --n_ensemble 8
#Air-Europe
python nodepfn/node_classification.py  --dataset air-europe --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 1 --label_num_per_class 20 --n_ensemble 32
# Air-Brazil
python nodepfn/node_classification.py  --dataset air-brazil --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --label_num_per_class 20 --n_ensemble 32
# WikiCS
python nodepfn/node_classification.py  --dataset wikics --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 2 --n_ensemble 32 --svd_algorithm arpack --cpu
# Computers
python nodepfn/node_classification.py  --dataset amazon-computer --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 3 --svd_algorithm arpack
# Photo
python nodepfn/node_classification.py  --dataset amazon-photo --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 3 --svd_algorithm arpack
# DBLP
python nodepfn/node_classification.py  --dataset dblp --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3
# Coauthor-CS
python nodepfn/node_classification.py  --dataset coauthor-cs  --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 2 --n_ensemble 32  --svd_algorithm arpack
# Coauthor-Physics
python nodepfn/node_classification.py  --dataset coauthor-physics --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 4  --n_ensemble 4 --svd_algorithm randomized
# Deezer
python nodepfn/node_classification.py  --dataset deezer --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 50  --runs=5 --smoothing_steps 0 --n_ensemble 32
# Cornell
python nodepfn/node_classification.py  --dataset cornell --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 0 --n_ensemble=16 --svd_algorithm randomized
# Texas
python nodepfn/node_classification.py  --dataset texas --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 20  --runs=10  --smoothing_steps 0
# Wisconsin
python nodepfn/node_classification.py  --dataset wisconsin --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5  --smoothing_steps 0 --n_ensemble=32 --svd_algorithm randomized
# Chameleon
python nodepfn/node_classification.py  --dataset chameleon --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5  --smoothing_steps 0 --n_ensemble=16
# Actor
python nodepfn/node_classification.py  --dataset actor --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 10  --runs=5  --smoothing_steps 0 --n_ensemble=32
# Minesweeper
python nodepfn/node_classification.py  --dataset minesweeper --base_model_path=models_ckpts/nodepfn --dim_reduction none --n_components 15  --runs=5 --smoothing_steps 1 --n_ensemble 32
# Tolokers
python nodepfn/node_classification.py  --dataset tolokers --base_model_path=models_ckpts/nodepfn --dim_reduction none --n_components 25  --runs=5 --smoothing_steps 2 --n_ensemble 4
# Amazon-Ratings
python nodepfn/node_classification.py  --dataset amazon-ratings --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 20  --runs=3 --smoothing_steps 3 --n_ensemble 8
# Questions
python nodepfn/node_classification.py  --dataset questions --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --n_ensemble 1
# Squirrel
python nodepfn/node_classification.py  --dataset squirrel --base_model_path=models_ckpts/nodepfn --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 2 --svd_algorithm arpack