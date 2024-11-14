source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

MODEL_NAME=ArmoRM-Llama3-8B-v0.1

#MODEL_DIR=./checkpoint/mistral_instruct-rm-uf
MODEL_DIR=/workspace/jihaozhe/models/$MODEL_NAME
VALUE_HEAD="regression_head"

# template from https://huggingface.co/openbmb/UltraRM-13b
IN_TEMPLATE="src/template/ArmoRM.in" 
OUT_TEMPLATE="src/template/ArmoRM.out"

SAVE_DATA_DIR=./data/uf-${MODEL_NAME}

mkdir -p $SAVE_DATA_DIR

#deepspeed --include localhost:0,1,2,3,4,5 src/batch_inference.py \
deepspeed src/batch_inference.py \
    --eval_task rm_pair \
    --zero_stage 3 \
    --flash_attn \
    --bf16 \
    --train_batch_size 8 \
    --micro_batch_size 32 \
    --value_head_prefix $VALUE_HEAD \
    --pretrain $MODEL_DIR \
    --dataset /workspace/jihaozhe/data/ultrafeedback-binarized-preferences \
    --dataset_split "train" \
    --chosen_key chosen_response \
    --rejected_key rejected_response \
    --prompt_key instruction \
    --max_len 8192 \
    --output_path $SAVE_DATA_DIR \
    --input_template $IN_TEMPLATE \
    --output_template $OUT_TEMPLATE \
    2>&1 | tee -a $SAVE_DATA_DIR/eval.log


    



