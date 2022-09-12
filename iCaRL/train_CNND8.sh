#!/bin/bash
#SBATCH  --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G
source /scratch_net/kringel/chuqli/conda/etc/profile.d/conda.sh
conda activate icarl_pytorch
############################################

python main_icarl_CNND.py --name icarl_df_7_mixup_13 --checkpoints_dir /srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --mixup 

python main_icarl_CNND.py --name icarl_df_7_ls_15 --checkpoints_dir /srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --label-smoothing
