import os
import time
import wandb
import argparse
import numpy as np

from configs import TrainOptions
from methods.lucir.trainer import Trainer

"""
python lucir_binary_main.py --name icarl_df --checkpoints_dir ./checkpoints  --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass 0 0 1 0 0 0 0 --batch_size 32 --num_epochs 30 > lucir_binary
"""

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt

args = TrainOptions().parse()
val_opt = get_val_opt()


# Print the parameters
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('Using gpu:', args.gpu)


time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
# wandb.init(project="GANIL", entity="iamwangyabin", config=the_args, save_code=False, tags=[time], )
# wandb.run.log_code(os.getcwd())
# wandb.run.log_code(".")
# wandb.run.name = wandb.run.id
# wandb.run.save()

# 保存每一次试验代码的
# import shutil
# shutil.copytree(os.path.abspath('.'), '../aalog/'+wandb.run.name,
#               ignore=lambda directory, contents: ['data', 'logs', 'runs', 'wandb'] if directory == os.path.abspath('.') else [])

if __name__ == '__main__':
    trainer = Trainer(args, val_opt, time)
    trainer.train()
