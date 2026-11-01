#!/bin/bash
# 2-minute check that a plain sbatch job on an H200 node can use env_train (no container).
#   sbatch cluster/check.sbatch   then   cat logs/check_<jobid>.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:10:00
#SBATCH --job-name=secfix-check
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/%u/nemotron-secfix/logs/check_%j.out
#SBATCH --error=/scratch/%u/nemotron-secfix/logs/check_%j.out
cd /scratch/$USER/nemotron-secfix
echo "node: $(hostname)"; head -2 /etc/os-release; ldd --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
env_train/bin/python -c "
import torch, transformers, peft
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); print('matmul ok:', (x @ x).float().abs().mean().item() > 0)
print('transformers', transformers.__version__, 'peft', peft.__version__)
from transformers import AutoConfig; print('model config:', AutoConfig.from_pretrained('models/lightning-bf16').model_type)
"
echo "CHECK DONE"
