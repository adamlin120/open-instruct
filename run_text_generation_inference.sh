#!/bin/sh

# If $1 is a path to a local model, use it; otherwise default to the specified or a default model from Hugging Face Model Hub
if [ -d "$1" ]; then
    model="/model"
    model_volume="-v $1:$model"
else
    model=${1:-"meta-llama/Llama-2-70b-chat-hf"}
    model_volume=""
fi

num_shard=${2:-8}  # Use the second command-line argument if provided, otherwise default to 8
volume=${3:-"$PWD/data"}  # Use the third command-line argument if provided, otherwise default to $PWD/data
port=${4:-8080}  # Use the fourth command-line argument if provided, otherwise default to 8080
max_input_length=${5:-2000}  # set to max prompt length (should be < max_length)
max_length=${6:-4000}  # set to max length in tokenizer_config

docker run --gpus all --shm-size 1g -p $port:80 $model_volume -v $volume:/data -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN ghcr.io/huggingface/text-generation-inference:latest --model-id $model --num-shard $num_shard --max-input-length $max_input_length --max-total-tokens $max_length
