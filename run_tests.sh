# !/bin/bash

# baseline
python -m nodepfn.pretrain --model_name=baseline --is_baseline

# ginat
python -m nodepfn.pretrain --prompt_cache_dir=/gfs/shared/public/datasets/ginat/ --hf_model=Qwen_Qwen3-8B-Base --prompts_file=citation_descriptions.json --model_name=ginat

# baseline - not gps
python -m nodepfn.pretrain --model_name=baseline --is_baseline