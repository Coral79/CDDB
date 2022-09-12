# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified for DyTox by Arthur Douillard
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
from continual.losses import bce_with_logits, soft_bce_with_logits

warnings.filterwarnings("ignore")


from configs.arg_parser import get_args_parser

def get_criterion(args):
    if args.mixup > 0. or args.cutmix > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.bce_loss:
        criterion = bce_with_logits
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def main(args):
    print(args)
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
    model.head = Classifier(
        model.embed_dim, args.nb_classes, args.initial_increment,
        args.increment, len(scenario_train)
    )
    model.to(device)
    # model will be on multiple GPUs, while model_without_ddp on a single GPU, but
    # it's actually the same model.
    model_without_ddp = model
    teacher_model = None

    # change backbone
    # model_ckpt = torch.load("/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-09_dytox_ganfake_zero-shot_1/checkpoint_0.pth", map_location='cpu')['model']
    # model_without_ddp.load_state_dict(model_ckpt, strict=False)


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

        args.output_dir = os.path.join(args.output_basedir, f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.name}_{args.trial_id}")
    else:
        log_store = None
        log_path = long_log_path = None
    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        torch.distributed.barrier()

    output_dir = Path(args.output_dir)


    loss_scaler = scaler.ContinualScaler(args.no_amp)
    criterion = get_criterion(args)


    if args.memory_size > 0:
        memory = Memory(
            args.memory_size, scenario_train.nb_classes, args.rehearsal, args.fixed_memory
        )
    else:
        memory = None

    nb_classes = args.initial_increment
    base_lr = args.lr
    accuracy_list = []
    start_time = time.time()

    if args.debug:
        args.base_epochs = 1
        args.epochs = 1

    args.increment_per_task = [args.initial_increment] + [args.increment for _ in range(len(scenario_train) - 1)]

    # --------------------------------------------------------------------------
    #
    # Begin of the task loop
    #
    # --------------------------------------------------------------------------
    dataset_true_val = None

    for task_id, dataset_train in enumerate(scenario_train):
        if args.max_task == task_id:
            print(f"Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")

        # ----------------------------------------------------------------------
        # Data
        dataset_val = scenario_val[:task_id + 1]
        if args.validation > 0.:  # use validation split instead of test
            if task_id == 0:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val = dataset_val
            else:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val.concat(dataset_val)
            dataset_val = dataset_true_val

        for i in range(3):  # Quick check to ensure same preprocessing between train/test
            assert abs(dataset_train.trsf.transforms[-1].mean[i] - dataset_val.trsf.transforms[-1].mean[i]) < 0.0001
            assert abs(dataset_train.trsf.transforms[-1].std[i] - dataset_val.trsf.transforms[-1].std[i]) < 0.0001

        loader_memory = None
        if task_id > 0 and memory is not None:
            dataset_memory = memory.get_dataset(dataset_train)
            loader_memory = factory.InfiniteLoader(factory.get_train_loaders(
                dataset_memory, args,
                args.replay_memory if args.replay_memory > 0 else args.batch_size
            ))
            if not args.sep_memory:
                previous_size = len(dataset_train)
                for _ in range(args.oversample_memory):
                    dataset_train.add_samples(*memory.get())
                print(f"{len(dataset_train) - previous_size} samples added from memory.")

            if args.only_ft:
                dataset_train = get_finetuning_dataset(dataset_train, memory, 'balanced')
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Initializing teacher model from previous task
        if use_distillation and task_id > 0:
            teacher_model = copy.deepcopy(model_without_ddp)
            teacher_model.freeze(['all'])
            teacher_model.eval()
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Ensembling
        if args.dytox:
            model_without_ddp = factory.update_dytox(model_without_ddp, task_id, args)
        elif args.dytox_pretrain:
            model_without_ddp = factory.update_dytox_pretrain(model_without_ddp, task_id, args)
        elif args.dytox_prompt:
            model_without_ddp = factory.update_dytox_prompt(model_without_ddp, task_id, args)
        elif args.dytox_ptconvit:
            model_without_ddp = factory.update_dytox_ptconvit(model_without_ddp, task_id, args)

        #####
        # if task_id == 0:
        #     # import pdb;pdb.set_trace()
        #     pretrained_model = torch.load("/home/wangyabin/workspace/dytox/logs/22-03-22_dytox_1/checkpoint_0.pth")['model']
        #     pretrained_model = {k: v for k, v in pretrained_model.items() if ('head' not in k)}
        #     pretrained_model = {k: v for k, v in pretrained_model.items() if ('tabs' not in k)}
        #
        #     model_without_ddp.load_state_dict(pretrained_model, strict=False)

        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Adding new parameters to handle the new classes
        print("Adding new parameters")
        if task_id > 0 and not args.dytox and not args.dytox_pretrain and not args.dytox_prompt and not args.dytox_ptconvit:
            model_without_ddp.head.add_classes()


        if task_id > 0:
            model_without_ddp.freeze(args.freeze_task)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Data
        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Learning rate and optimizer
        if task_id > 0 and args.incremental_batch_size:
            args.batch_size = args.incremental_batch_size

        if args.incremental_lr is not None and task_id > 0:
            linear_scaled_lr = args.incremental_lr * args.batch_size * utils.get_world_size() / 512.0
        else:
            linear_scaled_lr = base_lr * args.batch_size * utils.get_world_size() / 512.0

        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        # ----------------------------------------------------------------------

        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=nb_classes,
                loader_memory=loader_memory
            )

        skipped_task = False
        initial_epoch = epoch = 0
        if args.resume and args.start_task > task_id:
            utils.load_first_task_model(model_without_ddp, loss_scaler, task_id, args)
            print("Skipping first task")
            epochs = 0
            train_stats = {"task_skipped": str(task_id)}
            skipped_task = True
        elif args.base_epochs is not None and task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
            torch.distributed.barrier()
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs
        model_without_ddp.nb_batch_per_epoch = len(loader_train)

        # Init SAM, for DyTox++ (see appendix) ---------------------------------
        sam = None
        if args.sam_rho > 0. and 'tr' in args.sam_mode and ((task_id > 0 and args.sam_skip_first) or not args.sam_skip_first):
            if args.sam_final is not None:
                sam_step = (args.sam_final - args.sam_rho) / scenario_train.nb_tasks
                sam_rho = args.sam_rho + task_id * sam_step
            else:
                sam_rho = args.sam_rho

            print(f'Initialize SAM with rho={sam_rho}')
            sam = SAM(
                optimizer, model_without_ddp,
                rho=sam_rho, adaptive=args.sam_adaptive,
                div=args.sam_div,
                use_look_sam=args.look_sam_k > 0, look_sam_alpha=args.look_sam_alpha
            )
        # ----------------------------------------------------------------------

        print(f"Start training for {epochs-initial_epoch} epochs")
        max_accuracy = 0.0
        for epoch in range(initial_epoch, epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, loader_train,
                optimizer, device, epoch, task_id, loss_scaler,
                args.clip_grad, mixup_fn,
                debug=args.debug,
                args=args,
                teacher_model=teacher_model,
                model_without_ddp=model_without_ddp,
                sam=sam,
                loader_memory=loader_memory
            )

            lr_scheduler.step(epoch)

            if args.save_every_epoch is not None and epoch % args.save_every_epoch == 0:
                if os.path.isdir(args.resume):
                    with open(os.path.join(args.resume, 'save_log.txt'), 'w+') as f:
                        f.write(f'task={task_id}, epoch={epoch}\n')

                    checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
                    for checkpoint_path in checkpoint_paths:
                        if (task_id < args.start_task and args.start_task > 0) and os.path.isdir(args.resume) and os.path.exists(checkpoint_path):
                            continue

                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'task_id': task_id,
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)

            if args.eval_every and (epoch % args.eval_every  == 0 or (args.finetuning and epoch == epochs - 1)):
                eval_and_log(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, max_accuracy,
                    [], n_parameters, device, loader_val, train_stats, None, long_log_path,
                    logger, model_without_ddp.epoch_log()
                )
                logger.end_epoch()

        # print(f"+---------------------Zero Final Accuracy-------------------------+")
        # from continual.engine import evaluate
        # for task_id, dataset_train in enumerate(scenario_train):
        #     dataset_val = scenario_val[task_id]
        #     loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        #     print(str(task_id))
        #     print("num images: " + str(len(loader_val.dataset)))
        #     test_stats = evaluate(loader_val, model, device, logger)
        #     # print(str(task_id)+":\t"+test_stats['acc1'])
        # print(f"+------------------------------------------------------------+")
        # exit()


        if memory is not None and args.distributed_memory:
            task_memory_path = os.path.join(args.resume, f'dist_memory_{task_id}-{utils.get_rank()}.npz')
            if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                # Resuming this task step, thus reloading saved memory samples
                # without needing to re-compute them
                memory.load(task_memory_path)
            else:
                task_set_to_rehearse = scenario_train[task_id]
                if args.rehearsal_test_trsf:
                    task_set_to_rehearse.trsf = scenario_val[task_id].trsf

                memory.add(task_set_to_rehearse, model, args.initial_increment if task_id == 0 else args.increment)
                #memory.add(scenario_train[task_id], model, args.initial_increment if task_id == 0 else args.increment)

                if args.resume != '':
                    memory.save(task_memory_path)
                else:
                    memory.save(os.path.join(args.output_dir, f'dist_memory_{task_id}-{utils.get_rank()}.npz'))

        if memory is not None and not args.distributed_memory:
            task_memory_path = os.path.join(args.resume, f'memory_{task_id}.npz')
            if utils.is_main_process():
                if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                    # Resuming this task step, thus reloading saved memory samples
                    # without needing to re-compute them
                    memory.load(task_memory_path)
                else:
                    task_set_to_rehearse = scenario_train[task_id]
                    if args.rehearsal_test_trsf:
                        task_set_to_rehearse.trsf = scenario_val[task_id].trsf

                    memory.add(task_set_to_rehearse, model, args.initial_increment if task_id == 0 else args.increment)

                    if args.resume != '':
                        memory.save(task_memory_path)
                    else:
                        memory.save(os.path.join(args.output_dir, f'memory_{task_id}-{utils.get_rank()}.npz'))

            assert len(memory) <= args.memory_size, (len(memory), args.memory_size)
            torch.distributed.barrier()

            if not utils.is_main_process():
                if args.resume != '':
                    memory.load(task_memory_path)
                else:
                    memory.load(os.path.join(args.output_dir, f'memory_{task_id}-0.npz'))
                    memory.save(os.path.join(args.output_dir, f'memory_{task_id}-{utils.get_rank()}.npz'))

            torch.distributed.barrier()
        # ----------------------------------------------------------------------
        # FINETUNING
        # ----------------------------------------------------------------------

        # Init SAM, for DyTox++ (see appendix) ---------------------------------
        sam = None
        if args.sam_rho > 0. and 'ft' in args.sam_mode and ((task_id > 0 and args.sam_skip_first) or not args.sam_skip_first):
            if args.sam_final is not None:
                sam_step = (args.sam_final - args.sam_rho) / scenario_train.nb_tasks
                sam_rho = args.sam_rho + task_id * sam_step
            else:
                sam_rho = args.sam_rho

            print(f'Initialize SAM with rho={sam_rho}')
            sam = SAM(
                optimizer, model_without_ddp,
                rho=sam_rho, adaptive=args.sam_adaptive,
                div=args.sam_div,
                use_look_sam=args.look_sam_k > 0, look_sam_alpha=args.look_sam_alpha
            )
        # ----------------------------------------------------------------------

        if args.finetuning and memory and (task_id > 0 or scenario_train.nb_classes == args.initial_increment) and not skipped_task:
            dataset_finetune = get_finetuning_dataset(dataset_train, memory, args.finetuning, args.oversample_memory_ft, task_id)
            print(f'Finetuning phase of type {args.finetuning} with {len(dataset_finetune)} samples.')

            loader_finetune, loader_val = factory.get_loaders(dataset_finetune, dataset_val, args, finetuning=True)
            print(f'Train-ft and val loaders of lengths: {len(loader_finetune)} and {len(loader_val)}.')
            if args.finetuning_resetclf:
                model_without_ddp.reset_classifier()

            model_without_ddp.freeze(args.freeze_ft)

            if args.distributed:
                del model
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
                torch.distributed.barrier()
            else:
                model = model_without_ddp

            model_without_ddp.begin_finetuning()

            args.lr  = args.finetuning_lr * args.batch_size * utils.get_world_size() / 512.0
            optimizer = create_optimizer(args, model_without_ddp)
            for epoch in range(args.finetuning_epochs):
                if args.distributed and hasattr(loader_finetune.sampler, 'set_epoch'):
                    loader_finetune.sampler.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model, criterion, loader_finetune,
                    optimizer, device, epoch, task_id, loss_scaler,
                    args.clip_grad, mixup_fn,
                    debug=args.debug,
                    args=args,
                    teacher_model=teacher_model if args.finetuning_teacher else None,
                    model_without_ddp=model_without_ddp
                )

                if epoch % 10 == 0 or epoch == args.finetuning_epochs - 1:
                    eval_and_log(
                        args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                        epoch, task_id, loss_scaler, max_accuracy,
                        [], n_parameters, device, loader_val, train_stats, None, long_log_path,
                        logger, model_without_ddp.epoch_log()
                    )
                    logger.end_epoch()

            model_without_ddp.end_finetuning()

        eval_and_log(
            args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
            epoch, task_id, loss_scaler, max_accuracy,
            accuracy_list, n_parameters, device, loader_val, train_stats, log_store, log_path,
            logger, model_without_ddp.epoch_log(), skipped_task
        )
        logger.end_task()

        nb_classes += args.increment

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'Setting {args.data_set} with {args.initial_increment}-{args.increment}')
    print(f"All accuracies: {accuracy_list}")
    print(f"Average Incremental Accuracy: {statistics.mean(accuracy_list)}")

    print(f"+---------------------Final Accuracy-------------------------+")
    from continual.engine import evaluate
    for task_id, dataset_train in enumerate(scenario_train):
        dataset_val = scenario_val[task_id]
        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        print(str(task_id))
        print("num images: " + str(len(loader_val.dataset)))
        test_stats = evaluate(loader_val, model, device, logger)
    print(f"+------------------------------------------------------------+")



    if args.name:
        print(f"Experiment name: {args.name}")
        log_store['summary'] = {"avg": statistics.mean(accuracy_list)}
        if log_path is not None and utils.is_main_process():
            with open(log_path, 'a+') as f:
                f.write(json.dumps(log_store['summary']) + '\n')



def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)
        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DyTox training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)



    if args.options:
        name = load_options(args, args.options)
        if not args.name:
            args.name = name

    args.log_dir = os.path.join(
        args.log_path, args.data_set.lower(), args.log_category,
        datetime.datetime.now().strftime('%y-%m'),
        f"week-{int(datetime.datetime.now().strftime('%d')) // 7 + 1}",
        f"{int(datetime.datetime.now().strftime('%d'))}_{args.name}"
    )

    import wandb
    wandb.init(
        project="dytox-deepfake",
        group='{}'.format(args.name),
        name='{}'.format(args.name),
        config=args)

    if isinstance(args.class_order, list) and isinstance(args.class_order[0], list):
        print(f'Running {len(args.class_order)} different class orders.')
        class_orders = copy.deepcopy(args.class_order)

        for i, order in enumerate(class_orders, start=1):
            print(f'Running class ordering {i}/{len(class_orders)}.')
            args.trial_id = i
            args.class_order = order
            main(args)
    else:
        args.trial_id = 1
        main(args)
