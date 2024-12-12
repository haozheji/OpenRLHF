source /workspace/jihaozhe/miniconda3/etc/profile.d/conda.sh
conda activate exo

SAVE_DATA_DIR=/workspace/jihaozhe/models/ArmoRM-Llama3-8B-v0.1

CUDA_VISIBLE_DEVICES=7 python src/serve_rm.py \
    --reward_pretrain /workspace/jihaozhe/models/ArmoRM-Llama3-8B-v0.1 \
    --output_key "score" \
    --port 6006 \
    --bf16 \
    --flash_attn \
    --max_len 2048 \
    --batch_size 128 \
    2>&1 | tee -a $SAVE_DATA_DIR/eval.log
