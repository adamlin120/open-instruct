BATCH_SIZE_PER_GPU=${1:-1}
DEBUG=${2:-0}

NUM_GPUS=8

MODEL_NAME="yentinglin/zh_llama2_13b"
DATASET_NAME="yentinglin/zh_instruction"
MODEL_NAME_STEM="zh_llama2_13b-sft-0730"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME_STEM},${DATASET_NAME},instruction-tuning,stage3,no-offload,plus_evol"

export WANDB_API_KEY="94f8e06129c90551b50bfc5556e389fc574c2fa3"
export HUGGING_FACE_HUB_TOKEN="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"
export OMP_NUM_THREADS=1

echo "Training ${MODEL_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU"

python -m torch.distributed.run \
  --nproc_per_node=$NUM_GPUS \
  sft_trainer.py \
  --model_name $MODEL_NAME \
  --output_dir "output/${MODEL_NAME_STEM}/" \
  --debug $DEBUG \
  --learning_rate 1e-5 \
  --batch_size $BATCH_SIZE_PER_GPU \
  --num_train_epochs 3