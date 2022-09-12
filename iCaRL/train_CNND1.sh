#!/bin/bash
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G
source /scratch_net/kringel/chuqli/conda/etc/profile.d/conda.sh
conda activate icarl_pytorch
REPEAT=1
############################################

python main_icarl_CNND.py --name MT_7_bm_sum_b_sig01_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.1 --binary_loss sum_b_sig

python main_icarl_CNND.py --name MT_7_bm_sum_b_sig09_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.9 --binary_loss sum_b_sig

python main_icarl_CNND.py --name MT_7_bm_sum_b_log01_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.1 --binary_loss sum_b_log

python main_icarl_CNND.py --name MT_7_bm_sum_b_log07_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.7 --binary_loss sum_b_log

