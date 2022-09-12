""" Main function for this project. """
import os
import argparse
import numpy as np
from trainer import Trainer
"""
python arpmain.py --lr-scheduler cosineannealinglr --lr-warmup-epoch 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 160 --random-erase 0.1 --weight-decay 0.0002 --norm-weight-decay 0.0 --label-smoothing 0.1 --train_batch_size 128 --base_lr 0.1"""
import wandb
import time

parser = argparse.ArgumentParser()

### Basic parameters
parser.add_argument('--gpu', default='0', help='the index of GPU')
parser.add_argument('--num_phase', default=3, type=int)
parser.add_argument('--rootpath', default='/home/yabin/workspace/data/new', type=str)
parser.add_argument('--ckpt_dir_fg', type=str,
                    default='/home/yabin/workspace/Incremental/adaptive-aggregation-networks/logs/cifar100_nfg50_ncls10_nproto20_lucir_oyonudg1/iter_4.pth',  # 这个是LUCIR+反正学
                    help='the checkpoint file for the 0-th phase')
parser.add_argument('--resume_fg', action='store_true', help='resume 0-th phase model from the checkpoint')
parser.add_argument('--resume', action='store_true', help='resume from the checkpoints')
parser.add_argument('--num_workers', default=16, type=int, help='the number of workers for loading data')
parser.add_argument('--random_seed', default=1993, type=int, help='random seed')
parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size for train loader')
parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test loader')
parser.add_argument('--eval_batch_size', default=128, type=int, help='the batch size for validation loader')
parser.add_argument('--disable_gpu_occupancy', action='store_false', help='disable GPU occupancy')

### Incremental learning parameters
parser.add_argument('--epochs', default=600, type=int, help='the number of epochs')  # 160
parser.add_argument('--dynamic_budget', action='store_true', help='using dynamic budget setting')

### LUCIR parameters
parser.add_argument('--the_lambda', default=5, type=float, help='lamda for LF')
parser.add_argument('--dist', default=0.5, type=float, help='dist for margin ranking losses')
parser.add_argument('--K', default=2, type=int, help='K for margin ranking losses')
parser.add_argument('--lw_mr', default=1, type=float, help='loss weight for margin ranking losses')

### iCaRL parameters
parser.add_argument('--icarl_beta', default=0.25, type=float, help='beta for iCaRL')
parser.add_argument('--icarl_T', default=2, type=int, help='T for iCaRL')
parser.add_argument('--less_forget', action='store_true', help='Less forgetful')

### Enhance Classification parameters
parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
# distributed training parameters
parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument(
    "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
parser.add_argument("--model-ema-steps", type=int, default=32,
    help="the number of iterations that controls how often to update the EMA model (default: 32)",)
parser.add_argument("--model-ema-decay", type=float, default=0.99998,
    help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)
parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument( "--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay",)
parser.add_argument("--norm-weight-decay", default=None, type=float,
    help="weight decay for Normalization layers (default: None, same value as --wd)",)
parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
### General learning parameters
parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--custom_weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
parser.add_argument('--base_lr', default=0.1, type=float, help='learning rate for the 0-th phase')


the_args = parser.parse_args()


# Print the parameters
print(the_args)
# Set GPU index
os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
print('Using gpu:', the_args.gpu)

time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
wandb.init(project="GANIL", entity="iamwangyabin", config=the_args, save_code=False, tags=[time], )
wandb.run.log_code(os.getcwd())
wandb.run.log_code(".")
wandb.run.name = wandb.run.id
wandb.run.save()
#
# ###########保存每一次试验代码的
# import shutil
# shutil.copytree(os.path.abspath('.'), '../aalog/'+wandb.run.name,
#               ignore=lambda directory, contents: ['data', 'logs', 'runs', 'wandb'] if directory == os.path.abspath('.') else [])

if __name__ == '__main__':
    trainer = Trainer(the_args, wandb.run.name)
    trainer.train()
