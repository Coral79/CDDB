#!/bin/bash
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G
source /scratch_net/kringel/chuqli/conda/etc/profile.d/conda.sh
conda activate icarl_pytorch
REPEAT=1
############################################

python main_icarl_CNND.py --name icarl_ganfake_binary_256_ep60 --checkpoints_dir /srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL/checkpoints  --batch_size 32 --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/DoGAN_data/new/GanFake --task_name cyclegan,progan256,progan1024,glow,stargan  --multiclass 1 1 0 1 1 --init_lr 0.001 --num_epochs 60 --schedule 20 40 60 --binary --nb_protos 256