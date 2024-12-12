source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

# =========== policy argument ===========
policy_name=zephyr-7b-sft-full
policy_dir=/workspace/jihaozhe/models/$policy_name
N=128
in_temp="src/template/zephyr.in"
policy_output_dir=./data/uf-${policy_name}_n${N}

# =========== rm argument ===========
rm_dir=/workspace/jihaozhe/rm_hacking/




# stage-1: rollout
CUDA_VISIBLE_DEVICES=0,1,2,3 \
deepspeed src/batch_inference.py \
    --eval_task generate \
    --zero_stage 3 \
    --flash_attn \
    --bf16 \
    --max_num_seqs 2 \
    --pretrain $policy_dir \
    --dataset /workspace/jihaozhe/data/ultrafeedback-binarized-preferences \
    --dataset_split "train[-1024:]" \
    --tp_size 1 \
    --seed 42 \
    --max_new_tokens 1024 \
    --prompt_max_len 1024 \
    --micro_batch_size 64 \
    --input_template $in_temp \
    --best_of_n $N \
    --output_path $policy_output_dir \
    --input_key "instruction" \







# stage-2: argmax