# !/bin/bash
# # Actor - MemoryError
# python -m nodepfn.node_classification  --dataset actor --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 10  --runs=5  --smoothing_steps 0 --n_ensemble=32 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Air-Brazil
# python -m nodepfn.node_classification  --dataset air-brazil --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --label_num_per_class 20 --n_ensemble 32 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=transportation_descriptions.json
# #Air-Europe
# python -m nodepfn.node_classification  --dataset air-europe --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 1 --label_num_per_class 20 --n_ensemble 32\
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=transportation_descriptions.json
# Air-USA
# python -m nodepfn.node_classification  --dataset air-usa --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --label_num_per_class 20 --n_ensemble 8 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=transportation_descriptions.json
# Computers
python -m nodepfn.node_classification  --dataset amazon-computer --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 3 --svd_algorithm arpack \
    --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=products_descriptions.json
# # Photo
# python -m nodepfn.node_classification  --dataset amazon-photo --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 3 --svd_algorithm arpack \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=products_descriptions.json
# # Chameleon
# python -m nodepfn.node_classification  --dataset chameleon --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5  --smoothing_steps 0 --n_ensemble=16 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Citeseer
# python -m nodepfn.node_classification  --dataset citeseer --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 2 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Coauthor-CS
# python -m nodepfn.node_classification  --dataset coauthor-cs  --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 2 --n_ensemble 32  --svd_algorithm arpack \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Coauthor-Physics
# python -m nodepfn.node_classification  --dataset coauthor-physics --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 4  --n_ensemble 4 --svd_algorithm randomized \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# Cora
# python -m nodepfn.node_classification  --dataset cora --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 4 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Cornell
# python -m nodepfn.node_classification  --dataset cornell --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 0 --n_ensemble=16 --svd_algorithm randomized \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # DBLP 
# python -m nodepfn.node_classification  --dataset dblp --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Minesweeper 
# python -m nodepfn.node_classification  --dataset minesweeper --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction none --n_components 15  --runs=5 --smoothing_steps 1 --n_ensemble 32 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=minesweeper_descriptions.json
# # Pubmed
# python -m nodepfn.node_classification  --dataset pubmed --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15 --runs=5 --smoothing_steps 2  \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Questions - MemoryError
# python -m nodepfn.node_classification  --dataset questions --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5 --smoothing_steps 3 --n_ensemble 1 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=questions_descriptions.json
# # Squirrel
# python -m nodepfn.node_classification  --dataset squirrel --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 2 --svd_algorithm arpack \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Texas
# python -m nodepfn.node_classification  --dataset texas --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 20  --runs=10  --smoothing_steps 0 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Tolokers
# python -m nodepfn.node_classification  --dataset tolokers --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction none --n_components 25  --runs=5 --smoothing_steps 2 --n_ensemble 4 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=tolokers_descriptions.json
# # WikiCS
# python -m nodepfn.node_classification  --dataset wikics --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 15  --runs=5 --smoothing_steps 2 --n_ensemble 32 --svd_algorithm arpack --cpu \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Wisconsin
# python -m nodepfn.node_classification  --dataset wisconsin --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 25  --runs=5  --smoothing_steps 0 --n_ensemble=32 --svd_algorithm randomized \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=citation_descriptions.json
# # Amazon-Ratings - MemoryError
# python -m nodepfn.node_classification  --dataset amazon-ratings --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 20  --runs=3 --smoothing_steps 3 --n_ensemble 8 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=amazon_descriptions.json
# # Deezer - MemoryError
# python -m nodepfn.node_classification  --dataset deezer --base_model_path=models_ckpts/ginat_right_place_injection --dim_reduction tsvd --n_components 50  --runs=5 --smoothing_steps 0 --n_ensemble 32 \
#     --prompt_cache_dir=/gfs/shared/public/datasets/ginat_roy/ --hf_model=Qwen/Qwen3-8B-Base --prompts_file=reddit_descriptions.json 