BATCH_SIZE_PER_GPU=4
NUM_GPUS=8
GRADIENT_ACC_STEPS=1

MODEL_NAME="yentinglin/zh_TW_LLAMA2"
TOKENIZER_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="data/processed/code_alpaca/code_alpaca_data.jsonl"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME},${DATASET_NAME},stage3,no-offload, multi-node"

echo "Training llama2 model: ${MODEL_NAME}"
echo "using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


deepspeed \
  --hostfile hostfile.txt \
  --num_nodes 4 \
  --num_gpus 8 \
  open_instruct/finetune_trainer.py \
  --deepspeed ds_configs/stage3_no_offloading.conf \
  --model_name_or_path $MODEL_NAME \
  --tokenizer_name $TOKENIZER_NAME \
  --use_fast_tokenizer False \
  --train_file $DATASET_NAME \
  --max_seq_length 4096 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --num_train_epochs 3 \
  --do_train \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --evaluation_strategy "no" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 3 \
  --output_dir /output/test_run/ \
  --bf16 True \
  --tf32 True \
  --save_on_each_node True \
  --overwrite_output_dir
