#!/bin/bash
#SBATCH  --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G
source /scratch_net/kringel/chuqli/conda/etc/profile.d/conda.sh
conda activate icarl_pytorch
############################################

python main_icarl_CNND.py --name icarl_df_12_binary --checkpoints_dir /srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san --multiclass  0 0 1 0 0 0 0 1 1 1 0 0 --batch_size 32 --num_epochs 30 --schedule 10 20 30 --binary

python main_icarl_CNND.py --name icarl_df_12_sum_a_sig03 --checkpoints_dir /srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san --multiclass  0 0 1 0 0 0 0 1 1 1 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.3 --binary_loss sum_a_sig


