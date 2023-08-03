export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE_ARG=${1:-13}
BATCH_SIZE_PER_GPU=${2:-1}
DEBUG=${3:-0}

MODEL_SIZE="${MODEL_SIZE_ARG}b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_SIZE}-hf"
OUTPUT_DIR="zh_llama2_${MODEL_SIZE}_-0802"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME},zh_c4,zh_wiki,tw_news,magazine,dcard,ptt,${MODEL_SIZE},fsdp,trl"

export WANDB_API_KEY="94f8e06129c90551b50bfc5556e389fc574c2fa3"
export HUGGING_FACE_HUB_TOKEN="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"

echo "Training llama2 model: ${MODEL_NAME}"

python -m torch.distributed.run \
  --nproc_per_node=$NUM_GPUS \
  pretrain.py \
  --model_name $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --debug $DEBUG \
  --learning_rate 1e-5 \
  --batch_size $BATCH_SIZE_PER_GPU
