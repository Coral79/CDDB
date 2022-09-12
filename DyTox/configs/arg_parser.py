
import argparse




def get_args_parser():
    parser = argparse.ArgumentParser('DyTox training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--incremental-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--base-epochs', default=500, type=int,
                        help='Number of epochs for base task')
    parser.add_argument('--no-amp', default=False, action='store_true',
                        help='Disable mixed precision')

    # Model parameters
    parser.add_argument('--model', default='')
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--embed-dim', default=768, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num-heads', default=12, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--norm', default='layer', choices=['layer', 'scale'],
                        help='Normalization layer type')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--incremental-lr", default=None, type=float,
                        help="LR to use for incremental task (t > 0)")
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--incremental-warmup-lr', type=float, default=None, metavar='LR',
                        help='warmup learning rate (default: 1e-6) for task T > 0')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem", "old"')

    # Distillation parameters
    parser.add_argument('--auto-kd', default=False, action='store_true',
                        help='Balance kd factor as WA https://arxiv.org/abs/1911.07053')
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output-dir', default='',
                        help='Dont use that')
    parser.add_argument('--output-basedir', default='./checkponts/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Continual Learning parameters
    parser.add_argument("--initial-increment", default=50, type=int,
                        help="Base number of classes")
    parser.add_argument("--increment", default=10, type=int,
                        help="Number of new classes per incremental task")
    parser.add_argument('--class-order', default=None, type=int, nargs='+',
                        help='Class ordering, a list of class ids.')

    parser.add_argument("--eval-every", default=50, type=int,
                        help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Only do one batch per epoch')
    parser.add_argument('--max-task', default=None, type=int,
                        help='Max task id to train on')
    parser.add_argument('--name', default='', help='Name to display for screen')
    parser.add_argument('--options', default=[], nargs='*')

    # DyTox related
    parser.add_argument('--dytox', action='store_true', default=False, help='Enable super DyTox god mode.')
    parser.add_argument('--dytox_pretrain', action='store_true', default=False, help='Enable super DyTox god mode.')
    parser.add_argument('--dytox_prompt', action='store_true', default=False, help='Enable super DyTox god mode.')
    parser.add_argument('--dytox_ptconvit', action='store_true', default=False, help='Enable super DyTox god mode.')
    parser.add_argument('--ind-clf', default='', choices=['1-1', '1-n', 'n-n', 'n-1'],
                        help='Independent classifier per task but predicting all seen classes')
    parser.add_argument('--joint-tokens', default=False, action='store_true',
                        help='Forward w/ all task tokens alltogether [Faster but not working as well, not sure why')

    # Diversity
    parser.add_argument('--head-div', default=0., type=float,
                        help='Use a divergent head to predict among new classes + 1 using last token')
    parser.add_argument('--head-div-mode', default=['tr', 'ft'], nargs='+', type=str,
                        help='Only do divergence during training (tr) and/or finetuning (ft).')

    # SAM-related parameters
    # SAM fails with Mixed Precision, so use --no-amp
    parser.add_argument('--sam-rho', default=0., type=float,
                        help='Rho parameters for Sharpness-Aware Minimization. Disabled if == 0.')
    parser.add_argument('--sam-adaptive', default=False, action='store_true',
                        help='Adaptive version of SAM (more robust to rho)')
    parser.add_argument('--sam-first', default='main', choices=['main', 'memory'],
                        help='Apply SAM first step on main or memory loader (need --sep-memory for the latter)')
    parser.add_argument('--sam-second', default='main', choices=['main', 'memory'],
                        help='Apply SAM second step on main or memory loader (need --sep-memory for the latter)')
    parser.add_argument('--sam-skip-first', default=False, action='store_true',
                        help='Dont use SAM for first task')
    parser.add_argument('--sam-final', default=None, type=float,
                        help='Final value of rho is it is changed linearly per task.')
    parser.add_argument('--sam-div', default='', type=str,
                        choices=['old_no_upd'],
                        help='SAM for diversity')
    parser.add_argument('--sam-mode', default=['tr', 'ft'], nargs='+', type=str,
                        help='Only do SAM during training (tr) and/or finetuning (ft).')
    parser.add_argument('--look-sam-k', default=0, type=int,
                        help='Apply look sam every K updates (see under review ICLR22)')
    parser.add_argument('--look-sam-alpha', default=0.7, type=float,
                        help='Alpha factor of look sam to weight gradient reuse, 0 < alpha <= 1')

    # Rehearsal memory
    parser.add_argument('--memory-size', default=2000, type=int,
                        help='Total memory size in number of stored (image, label).')
    parser.add_argument('--distributed-memory', default=False, action='store_true',
                        help='Use different rehearsal memory per process.')
    parser.add_argument('--global-memory', default=False, action='store_false', dest='distributed_memory',
                        help='Use same rehearsal memory for all process.')
    parser.set_defaults(distributed_memory=True)
    parser.add_argument('--oversample-memory', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal.')
    parser.add_argument('--oversample-memory-ft', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal for finetuning, only for old classes not new classes.')
    parser.add_argument('--rehearsal-test-trsf', default=False, action='store_true',
                        help='Extract features without data augmentation.')
    parser.add_argument('--rehearsal-modes', default=1, type=int,
                        help='Select N on a single gpu, but with mem_size/N.')
    parser.add_argument('--fixed-memory', default=False, action='store_true',
                        help='Dont fully use memory when no all classes are seen as in Hou et al. 2019')
    parser.add_argument('--rehearsal', default="random",
                        choices=[
                            'random',
                            'closest_token', 'closest_all',
                            'icarl_token', 'icarl_all',
                            'furthest_token', 'furthest_all'
                        ],
                        help='Method to herd sample for rehearsal.')
    parser.add_argument('--sep-memory', default=False, action='store_true',
                        help='Dont merge memory w/ task dataset but keep it alongside')
    parser.add_argument('--replay-memory', default=0, type=int,
                        help='Replay memory according to Guido rule [NEED DOC]')

    # Finetuning
    parser.add_argument('--finetuning', default='', choices=['balanced'],
                        help='Whether to do a finetuning after each incremental task. Backbone are frozen.')
    parser.add_argument('--finetuning-epochs', default=30, type=int,
                        help='Number of epochs to spend in finetuning.')
    parser.add_argument('--finetuning-lr', default=5e-5, type=float,
                        help='LR during finetuning, will be kept constant.')
    parser.add_argument('--finetuning-teacher', default=False, action='store_true',
                        help='Use teacher/old model during finetuning for all kd related.')
    parser.add_argument('--finetuning-resetclf', default=False, action='store_true',
                        help='Reset classifier before finetuning phase (similar to GDumb/DER).')
    parser.add_argument('--only-ft', default=False, action='store_true',
                        help='Only train on FT data')
    parser.add_argument('--ft-no-sampling', default=False, action='store_true',
                        help='Dont use particular sampling for the finetuning phase.')

    # What to freeze
    parser.add_argument('--freeze-task', default=[], nargs="*", type=str,
                        help='What to freeze before every incremental task (t > 0).')
    parser.add_argument('--freeze-ft', default=[], nargs="*", type=str,
                        help='What to freeze before every finetuning (t > 0).')
    parser.add_argument('--freeze-eval', default=False, action='store_true',
                        help='Frozen layers are put in eval. Important for stoch depth')

    # Convit - CaiT
    parser.add_argument('--local-up-to-layer', default=10, type=int,
                        help='number of GPSA layers')
    parser.add_argument('--locality-strength', default=1., type=float,
                        help='Determines how focused each head is around its attention center')
    parser.add_argument('--class-attention', default=False, action='store_true',
                        help='Freeeze and Process the class token as done in CaiT')

    # Logs
    parser.add_argument('--log-path', default="logs")
    parser.add_argument('--log-category', default="misc")

    # Classification
    parser.add_argument('--bce-loss', default=False, action='store_true')

    # distributed training parameters
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Resuming
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-task', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--save-every-epoch', default=None, type=int)

    parser.add_argument('--validation', default=0.0, type=float,
                        help='Use % of the training set as val, replacing the test.')

    # ganfake task
    parser.add_argument('--task-name', default=None, type=str, nargs='+', help='Task name.')
    parser.add_argument('--multiclass', default=None, type=int, nargs='+', help='Multiclass.')
    parser.add_argument('--binary_loss', default='None', type=str, help='binary loss type.')
    parser.add_argument('--binary_weight', default=0.1, type=float, help='binary loss weight.')


    return parser
