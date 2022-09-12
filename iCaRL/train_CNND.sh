#!/bin/bash
#SBATCH  --gres=gpu:1
#SBATCH  --mem=60G
source /scratch_net/kringel/chuqli/conda/etc/profile.d/conda.sh
conda activate icarl_pytorch
REPEAT=1
############################################
python main_icarl_CNND.py "$@"