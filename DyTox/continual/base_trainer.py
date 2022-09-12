import argparse
import copy
import datetime
import json
import os
import statistics
import time
import warnings
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from continuum.metrics import Logger
from continuum.tasks import split_train_val
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from utils.mixup import Mixup
import utils.utils as utils
from utils import factory, scaler
from models.classifier import Classifier
from utils.incremental_utils.rehearsal import Memory, get_finetuning_dataset
from continual.sam import SAM
from utils.datasets import build_dataset
from continual.engine import eval_and_log, train_one_epoch
from continual.losses import bce_with_logits

warnings.filterwarnings("ignore")


class BaseLearner(object):
    def __init__(self, args):
        logger = Logger(list_subsets=['train', 'test'])

        use_distillation = args.auto_kd
        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        cudnn.benchmark = True

        scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
        scenario_val, _ = build_dataset(is_train=False, args=args)

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

        model = factory.get_backbone(args)
        model.head = Classifier(model.embed_dim, args.nb_classes, args.initial_increment,args.increment)
        model.to(device)
        # model will be on multiple GPUs, while model_without_ddp on a single GPU, but
        # it's actually the same model.
        model_without_ddp = model
        teacher_model = None

        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # Start the logging process on disk ----------------------------------------
        if args.name:
            log_path = os.path.join(args.log_dir, f"logs_{args.trial_id}.json")
            long_log_path = os.path.join(args.log_dir, f"long_logs_{args.trial_id}.json")

            if utils.is_main_process():
                os.system("echo '\ek{}\e\\'".format(args.name))
                os.makedirs(args.log_dir, exist_ok=True)
                with open(os.path.join(args.log_dir, f"config_{args.trial_id}.json"), 'w+') as f:
                    config = vars(args)
                    config["nb_parameters"] = n_parameters
                    json.dump(config, f, indent=2)
                with open(log_path, 'w+') as f:
                    pass  # touch
                with open(long_log_path, 'w+') as f:
                    pass  # touch
            log_store = {'results': {}}

            args.output_dir = os.path.join(args.output_basedir,
                                           f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.name}_{args.trial_id}")
        else:
            log_store = None
            log_path = long_log_path = None
        if args.output_dir and utils.is_main_process():
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        if args.distributed:
            torch.distributed.barrier()

        output_dir = Path(args.output_dir)

        pass



