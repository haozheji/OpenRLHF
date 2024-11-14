source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo


# =========== args ===========
kl=0.01
freezing_steps=-1
bsz=128
mbsz=8
rollout_bsz=1024
rollout_mbsz=16
eval_mbsz=16

actor_lr=5e-7
critic_lr=9e-6

eval_steps=2
eval_number=1024

# =========== names ===========

base_model_name=Llama
base_model=Llama-3.2-1B-sft-full
orc_rm_name=ArmoRM
orc_rm=ArmoRM-Llama3-8B-v0.1

data_name=uf-${orc_rm}


rm_name=RM_${base_model_name}_${data_name}
exp_name=PPO_ray_adp-clip-kl${kl}-a19-a299_${base_model}_${data_name}_bsz${bsz}-${mbsz}_rollout${rollout_bsz}-${rollout_mbsz}_alr${actor_lr}_clr${critic_lr}_eval${eval_number}-orc-step${eval_steps} #_eval-kl-step${eval_steps}_RUN2

# ============ dirs =============

working_dir=/workspace/jihaozhe/rm_hacking
pretrain_dir=/workspace/jihaozhe/models

sft_dir=${pretrain_dir}/${base_model}
rm_dir=${working_dir}/checkpoint/${rm_name}
orc_rm_dir=${pretrain_dir}/${orc_rm}

data_dir=${working_dir}/data/${data_name}
output=${working_dir}/checkpoint/${exp_name}

#oracle_data_dir=None #${working_dir}/data/uf1024-N128-zephyr-7b-sft-full-BoN-RM_zephyr_uf-ArmoRM-Llama3-8B-v0.1

oracle_data_dir=${working_dir}/data/uf1024-N1-PPO_ray_kl0.01_Llama-3.2-1B-sft-full_uf-ArmoRM-Llama3-8B-v0.1_bsz128-8_rollout1024-16_alr5e-7_clr9e-6_eval1024-orc-step2

mkdir -p $output

# ============= templates ==============

in_temp="template/${base_model_name}.in" 
out_temp="template/${base_model_name}.out"

orc_in_temp="template/${orc_rm_name}.in"
orc_out_temp="template/${orc_rm_name}.out"

# ============== commands ===============


# --runtime-env-json='{"working_dir": "/openrlhf"}' \

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8


#--oracle_data $oracle_data_dir \
RAY_DEDUP_LOGS=0 \
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/workspace/jihaozhe/rm_hacking/src", "conda": "exo"}' \
   -- python train_ppo_ray.py \
   --alpha1 0.9 \
   --alpha2 0.99 \
   --freezing_actor_steps $freezing_steps \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --oracle_num_nodes 0 \
   --oracle_num_gpus_per_node 1 \
   --oracle_eval_steps $eval_steps \
   --constraint_eval_steps $eval_steps \
   --remote_oracle_url "http://172.18.195.143:6006/get_reward" \
   --oracle_template $orc_in_temp $orc_out_temp \
   --oracle_pretrain $orc_rm_dir \
   --oracle_value_head_prefix "regression_head" \
   --oracle_split "eval" \
   --output_key "response" \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --ref_reward_offload \
   --pretrain $sft_dir \
   --reward_pretrain $rm_dir \
   --save_path $output \
   --micro_train_batch_size $mbsz \
   --train_batch_size $bsz \
   --micro_rollout_batch_size $rollout_mbsz \
   --rollout_batch_size $rollout_bsz \
   --micro_eval_batch_size $eval_mbsz \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $actor_lr \
   --critic_learning_rate $critic_lr \
   --init_kl_coef $kl \
   --prompt_data $data_dir \
   --prompt_split "train[:-${eval_number}]" \
   --eval_split "train[-${eval_number}:]" \
   --input_key "instruction" \
   --input_template $in_temp \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb 9b9195bb8719975e33f9f0ee70971bd51fe0d331 \
   --wandb_run_name $exp_name \
   --wandb_project rm_hacking \
   2>&1 | tee -a $output/training.log