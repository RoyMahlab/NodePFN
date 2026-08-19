#!/bin/bash

CONFIG_PATH=configs
NUM_STEPS=8096
EPOCHS=30
GPUS=8

uv run python pretrain_and_node_classification.py \
    --model_name full_baseline_${GPUS}_gpus \
    --prior causal \
    --gpus $GPUS \
    --epochs $EPOCHS \
    --num_steps $NUM_STEPS \
    --batch_size 1 \
    --max_num_classes 100

configs=(
  configs/geo_baseline.json
  configs/geo_baseline_less_features.json
  # configs/geo_baseline_less_geo_features.json
  # configs/geo_baseline_low_gamma_dist.json
  # configs/geo_baseline_mid_gamma_dist.json
)

for config_file in "${configs[@]}"; do
  echo "$config_file"
  CONF_NAME=$(basename "$config_file" .json)
  RUN_NAME="${CONF_NAME}_${GPUS}_gpus"
  uv run python pretrain_and_node_classification.py \
    --model_name $RUN_NAME \
    --prior geo \
    --gpus $GPUS \
    --epochs $EPOCHS \
    --num_steps $NUM_STEPS \
    --batch_size 1 \
    --max_num_classes 100 \
    --geo_config_path $config_file
done