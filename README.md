# Llama Hosting

A toolkit for hosting, fine-tuning, and converting Llama models.

## Overview

This repository contains utilities for working with Llama models, specifically focused on:
- Running Llama models using the llama.cpp framework
- Fine-tuning models using QLoRA and other techniques
- Converting models to GGUF format for efficient inference
- Creating and processing fine-tuning datasets

## Prerequisites

- Python 3.8+
- llama.cpp (included as a submodule)
- Required Python packages (to be installed via requirements)
- CUDA-compatible GPU (recommended for training)

## Repository Structure

- `run-llama.sh` - Shell script to run inference with llama.cpp
- `convert_to_gguf.py` - Convert models to GGUF format
- `finetune_qlora.py` - Fine-tune Llama models using QLoRA
- `finetune_simple.py` - Simplified fine-tuning script
- `make_ft_dataset.py` - Prepare datasets for fine-tuning
- `make_ft_ds.ipynb` - Jupyter notebook for dataset preparation
- `merge_and_convert.py` - Merge model weights and convert formats

## Quick Start

### Running a Llama Model

```bash
./run-llama.sh -p "Your prompt here"
```

The script uses default parameters which can be modified in the script or passed as arguments.

### Fine-tuning a Model

1. Prepare your dataset using `make_ft_dataset.py` or the Jupyter notebook
2. Run fine-tuning:

```bash
python finetune_qlora.py --model_name_or_path <base_model> --dataset_path <your_dataset> --output_dir <output_directory>
```

### Converting Models

To convert a model to GGUF format:

```bash
python convert_to_gguf.py --model_path <input_model> --output_path <output_directory>
```

## Advanced Usage

See individual script files for detailed usage instructions and available parameters.

## License

This project is open-source and available under [appropriate license].

## Acknowledgements

- This project uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient Llama model inference
- Based on research and models from Meta AI
