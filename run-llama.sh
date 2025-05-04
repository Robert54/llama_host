#!/bin/bash

# Path to the llama.cpp executable
LLAMA_CLI="/home/ubuntu/llama-hosting/llama.cpp/build/bin/llama-cli"

# Path to the model file
MODEL_PATH="/home/ubuntu/llama-hosting/Llama-4-Scout-Q4_K_XL/UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf"

# Default parameters
CONTEXT_SIZE="4096"
TEMPERATURE="0.7"
N_THREADS="4"

# Run the model
$LLAMA_CLI \
  -m $MODEL_PATH \
  -c $CONTEXT_SIZE \
  -t $N_THREADS \
  --temp $TEMPERATURE \
  --color \
  $@

# Add additional parameters by passing them as arguments to this script
# For example: ./run-llama.sh -p "Tell me a joke"
