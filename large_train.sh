#!/bin/bash


# baseline
python pretrain_and_node_classification.py --model_name baseline_100_classes --gpus 8 --num_steps 2048 --epochs 50 --batch_size 1 --prior causal

# geo_sampling
python pretrain_and_node_classification.py --model_name geo_100_classes --gpus 8 --num_steps 2048 --epochs 50 --batch_size 1 --prior geo --geo_similarity cosine