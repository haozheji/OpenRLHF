source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

model_type=zephyr
model_name=RM_zephyr_uf-ArmoRM-Llama3-8B-v0.1
model_dir=/workspace/jihaozhe/rm_hacking/checkpoint/$model_name
value_head=value_head

IN_TEMPLATE="src/template/${model_type}.in" 
OUT_TEMPLATE="src/template/${model_type}.out"


data_name=uf1024-N128-zephyr-7b-sft-full
dataset=./data/$data_name

save_dir=./data/$data_name-BoN-$model_name


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
deepspeed src/batch_inference.py \
    --eval_task rm \
    --post_processor rs \
    --pretrain $model_dir \
    --zero_stage 3 \
    --flash_attn \
    --bf16 \
    --train_batch_size 8 \
    --micro_batch_size 64 \
    --value_head_prefix $value_head \
    --dataset $dataset \
    --dataset_split eval \
    --input_key instruction \
    --output_key response \
    --max_len 2048 \
    --prompt_max_len 1024 \
    --input_template $IN_TEMPLATE \
    --output_template $OUT_TEMPLATE \
    --output_path $save_dir \