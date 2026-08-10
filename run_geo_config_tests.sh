#!/bin/bash

CONFIG_PATH=configs
NUM_STEPS=2048

# uv run python pretrain_and_node_classification.py \
#     --model_name full_baseline \
#     --prior causal \
#     --gpus 1 \
#     --epochs 50 \
#     --num_steps $NUM_STEPS \
#     --batch_size 1 \
#     --max_num_classes 100

configs=(
  configs/geo_baseline.json
  configs/geo_baseline_less_features.json
  # configs/geo_baseline_less_geo_features.json
  # configs/geo_baseline_low_gamma_dist.json
  # configs/geo_baseline_mid_gamma_dist.json
)

for config_file in "${configs[@]}"; do
  echo "$config_file"
  uv run python pretrain_and_node_classification.py \
    --model_name $config_file \
    --prior geo \
    --gpus 1 \
    --epochs 30 \
    --num_steps $NUM_STEPS \
    --batch_size 1 \
    --max_num_classes 100 \
    --geo_config_path $config_file
done