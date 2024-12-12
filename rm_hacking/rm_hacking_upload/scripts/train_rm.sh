source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

DATA_NAME=uf-ArmoRM-Llama3-8B-v0.1
#DATA_NAME=uf-UltraRM-13b
BASE_MODEL_NAME=Llama
BASE_MODEL=Llama-3.2-1B-sft-full


IN_TEMPLATE="src/template/${BASE_MODEL_NAME}.in" 
OUT_TEMPLATE="src/template/${BASE_MODEL_NAME}.out"

exp_name=RM_${BASE_MODEL_NAME}_${DATA_NAME}



#DATA_DIR=/workspace/jihaozhe/data/ultrafeedback-binarized-preferences
DATA_DIR=/workspace/jihaozhe/rm_hacking/data/$DATA_NAME
SFT_DIR=/workspace/jihaozhe/models/${BASE_MODEL}
OUTPUT=./checkpoint/$exp_name

mkdir -p $OUTPUT

#deepspeed src/train_rm.py \
#deepspeed --include localhost:0,1,2,3,4,5 src/train_rm.py \
deepspeed src/train_rm.py \
   --save_path $OUTPUT \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 50 \
   --train_batch_size 256 \
   --micro_train_batch_size 16 \
   --pretrain $SFT_DIR \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 3 \
   --input_template $IN_TEMPLATE \
   --output_template $OUT_TEMPLATE \
   --learning_rate 9e-6 \
   --dataset $DATA_DIR \
   --prompt_key instruction \
   --chosen_key chosen_response \
   --rejected_key rejected_response \
   --train_split "train[:-1000]" \
   --eval_split "train[-1000:]" \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb 9b9195bb8719975e33f9f0ee70971bd51fe0d331 \
   --wandb_run_name $exp_name \
   --wandb_project rm_hacking \
   2>&1 | tee -a $OUTPUT/training.log
