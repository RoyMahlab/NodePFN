#!/bin/bash

CONFIG_PATH=configs
NUM_STEPS=2048

uv run python pretrain_and_node_classification.py \
    --model_name full_baseline \
    --prior causal \
    --gpus 1 \
    --epochs 50 \
    --num_steps $NUM_STEPS \
    --batch_size 1 \
    --max_num_classes 100

for config_file in "$CONFIG_PATH/geo_baseline"*.json; do
  echo "$config_file"
  uv run python pretrain_and_node_classification.py \
    --model_name $config_file \
    --prior geo \
    --gpus 1 \
    --epochs 50 \
    --num_steps $NUM_STEPS \
    --batch_size 1 \
    --max_num_classes 100 \
    --geo_config_path $config_file
done