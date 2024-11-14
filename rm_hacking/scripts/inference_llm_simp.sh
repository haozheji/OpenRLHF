source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

model_name=PPO_ray_kl0_Llama-3.2-1B-sft-full_uf-ArmoRM-Llama3-8B-v0.1_bsz128-8_rollout1024-16_alr5e-7_clr9e-6_eval1024-orc-step2
#zephyr-7b-sft-full
model_type=Llama
N=1
num_sample=1024
devices=0,1,2,3,4,5,6,7

model_dir=/workspace/jihaozhe/rm_hacking/checkpoint/$model_name
save_dir=./data/uf${num_sample}-N${N}-${model_name}
in_temp=src/template/${model_type}.in

CUDA_VISIBLE_DEVICES=$devices python src/simple_inference.py \
    --model_dir $model_dir \
    --dataset /workspace/jihaozhe/data/ultrafeedback-binarized-preferences \
    --split train[-$num_sample:] \
    --input_key instruction \
    --input_template $in_temp \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --devices $devices \
    --top_p 1.0 \
    --temperature 1.0 \
    --micro_batch_size 64 \
    --N $N \
    --save_dir $save_dir \
    --save_split eval \