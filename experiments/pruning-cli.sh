#!/bin/bash

# 只有4个位置参数：显卡ID、模型路径、方法、比例
CUDA_VISIBLE_DEVICES=${1:-"0"}
export CUDA_VISIBLE_DEVICES
model_name=${2:-"/root/Qwen3-Next-80B-A3B-Thinking"}
pruning_method=${3:-"reap"}
compression_ratio=${4:-0.5}

# 移除这4个，剩下的透传
shift 4

echo "Running pruning on GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $model_name"

python src/reap/prune.py \
    --model-name "$model_name" \
    --prune-method "$pruning_method" \
    --compression-ratio "$compression_ratio" \
    --profile false \
    --do-eval false \
    --record_pruning_metrics_only true \
    "$@"
