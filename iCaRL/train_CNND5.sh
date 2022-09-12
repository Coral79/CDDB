#!/bin/bash
#SBATCH  --constraint='titan_xp|geforce_gtx_titan_x'
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G
source /scratch_net/kringel/chuqli/conda/etc/profile.d/conda.sh
conda activate icarl_pytorch
############################################

python main_icarl_CNND.py --name MT_7_bm_sum_a_sig07_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.7 --nb_protos 1536 --binary_loss sum_a_sig

python main_icarl_CNND.py --name MT_7_bm_sum_a_sig01_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.1 --nb_protos 1536 --binary_loss sum_a_sig

python main_icarl_CNND.py --name MT_7_bm_sum_a_sig09_15 --checkpoints_dir /scratch_net/kringel/chuqli/CNNDetection/checkpoints --model_weights /home/chuqli/scratch/CNNDetection/checkpoints/no_aug/model_epoch_best.pth --dataroot /srv/beegfs02/scratch/generative_modeling/data/Deepfake/test --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.9 --nb_protos 1536 --binary_loss sum_a_sig
