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


ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

CUDA_VISIBLE_DEVICES=0,1,2,3 \
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/workspace/jihaozhe/rm_hacking/src", "conda": "exo"}' \
    -- python ray_inference.py \
    --zero_stage 3 \
    --pretrain $policy_dir \
    --seed 42 \
    --dataset /workspace/jihaozhe/data/ultrafeedback-binarized-preferences \
    --dataset_split "train[-1024:]" \
    --input_key "instruction" \
    --micro_batch_size 64 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \

