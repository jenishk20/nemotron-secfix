#!/bin/bash
# LoRA SFT on the security set, then merge the adapter into full weights for vLLM.
#   sbatch cluster/train.sbatch            (run name defaults to lora_v1)
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=secfix-train
#SBATCH --mem=160GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/%u/nemotron-secfix/logs/train_%j.out
#SBATCH --error=/scratch/%u/nemotron-secfix/logs/train_%j.err
set -eo pipefail
cd /scratch/$USER/nemotron-secfix
RUN="${RUN:-lora_v1}"
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
env_train/bin/python cluster/sft_lora.py --model models/lightning-bf16 \
  --train data/secpy_train.jsonl --val data/secpy_val.jsonl --out runs/$RUN ${TRAIN_ARGS:-}
env_train/bin/python cluster/sft_lora.py --merge --model models/lightning-bf16 --out runs/$RUN
ls -la runs/$RUN/merged | head
