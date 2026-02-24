#!/bin/bash

# Ensure we are in the root directory
if [ ! -d "dllm" ]; then
    echo "Please run this script from the root of the repository."
    exit 1
fi

# 1. Convert the model (if not already done)
if [ ! -d "models/a2d/Qwen3-0.6B" ]; then
    echo "Converting Qwen/Qwen3-0.6B to non-causal attention..."
    python dllm/pipelines/a2d/convert.py \
        --model_name_or_path "Qwen/Qwen3-0.6B" \
        --output_dir "models/a2d/Qwen3-0.6B"
else
    echo "Model already converted at models/a2d/Qwen3-0.6B"
fi

# 2. Train BD3LM
# Adjusted gradient_accumulation_steps to 16 for 8 GPUs to match global batch size of 2048
# (8 GPUs * 16 batch size * 16 grad acc = 2048)
echo "Starting BD3LM training..."
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bd3lm/sft.py \
    --model_name_or_path "models/a2d/Qwen3-0.6B" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk+OpenCoder-LLM/opc-sft-stage1[lang:python]+OpenCoder-LLM/opc-sft-stage2[lang:python]" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --block_size 32 \
    --output_dir "models/a2d/Qwen3-0.6B/tulu-3-sft-mixture+smoltalk+opc-sft-stage1&2/epochs-10-bs-2048-len-512-bls-32"
