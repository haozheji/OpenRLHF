source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

DATA_NAME=uf-UltraRM-13b
BASE_MODEL_NAME=mistral

exp_name=PPO_${BASE_MODEL_NAME}_${DATA_NAME}

SFT_DIR=/workspace/jihaozhe/models/Mistral-7B-Instruct-v0.3
RM_DIR=checkpoint/RM_mistral_uf-UltraRM-13b
DATA_DIR=/workspace/jihaozhe/rm_hacking/data/$DATA_NAME


OUTPUT=./checkpoint/$exp_name

mkdir -p $OUTPUT

IN_TEMPLATE="template/mistral.in" 
OUT_TEMPLATE="template/mistral.out"

# 128
# 1024

#
#deepspeed --include localhost:0,1,2,3,4,5 src/train_ppo.py \
deepspeed src/train_ppo.py \
  --pretrain $SFT_DIR \
  --reward_pretrain $RM_DIR \
  --save_path $OUTPUT \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data $DATA_DIR \
  --prompt_split "train[:-1000]" \
  --input_template $IN_TEMPLATE \
  --input_key instruction \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb 9b9195bb8719975e33f9f0ee70971bd51fe0d331 \
  --wandb_run_name $exp_name \
  --wandb_project rm_hacking \
  2>&1 | tee -a $OUTPUT/training.log