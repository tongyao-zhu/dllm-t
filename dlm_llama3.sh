mode=$1
ds_name=$2
ds_path=/home/aiops/zhuty/cont_data/$ds_name/train
eval_path=/home/aiops/zhuty/cont_data/cc_valid

export WANDB_PROJECT="dllm"
export WANDB_MODE="online"

# Determine output directory based on mode
if [ "$mode" == "mdlm" ]; then
    output_dir="models/a2d/llama3-1b/mdlm/$ds_name"
elif [ "$mode" == "block" ]; then
    output_dir="models/a2d/llama3-1b/bd3lm_new/$ds_name"
else
    echo "Error: Invalid mode '$mode'. Must be 'mdlm' or 'block'."
    exit 1
fi

# Automatically find the latest checkpoint in output_dir
resume_arg=""
if [ -d "$output_dir" ]; then
    # Find all checkpoint directories (checkpoint-*), extract numbers, sort numerically, and get latest
    latest_checkpoint=$(find "$output_dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | \
        sed 's|.*/checkpoint-||' | \
        sort -n | \
        tail -1)
    
    if [ -n "$latest_checkpoint" ] && [ -d "$output_dir/checkpoint-$latest_checkpoint" ]; then
        checkpoint_path="$output_dir/checkpoint-$latest_checkpoint"
        resume_arg="--resume_from_checkpoint $checkpoint_path"
        echo "Found latest checkpoint: $checkpoint_path"
        echo "Resuming training from checkpoint: $checkpoint_path"
    else
        echo "No checkpoint found in $output_dir. Starting new training."
    fi
else
    echo "Output directory $output_dir does not exist. Starting new training."
fi

if [ "$mode" == "mdlm" ]; then

accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path "models/a2d/Llama-3.2-1B" \
    --num_proc 32 \
    --dataset_args $ds_path \
    --text_field "text" \
    --insert_eos True \
    --max_length 2048 \
    --max_steps 5000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --output_dir "$output_dir" \
    --eval_strategy "no" \
    --save_steps 500 \
    --save_only_model False \
    --ddp_timeout 7200 \
    --loss_norm_type "token" \
    --lr_scheduler_type "constant" \
    $resume_arg

elif [ "$mode" == "block" ]; then
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bd3lm/pt.py \
    --model_name_or_path "models/a2d/Llama-3.2-1B" \
    --num_proc 32 \
    --dataset_args $ds_path \
    --text_field "text" \
    --insert_eos True \
    --max_length 2048 \
    --max_steps 5000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --output_dir "$output_dir" \
    --eval_strategy "no" \
    --save_steps 500 \
    --save_only_model False \
    --ddp_timeout 7200 \
    --loss_norm_type "token" \
    --lr_scheduler_type "constant" \
    --attn_implementation "flex_attention" \
    $resume_arg
fi