import argparse
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--model_type', type=str, default='resnet',
                            help="The type (mlp|lenet|vgg|resnet) of backbone network")
        parser.add_argument('--model_name', type=str, default='resnet18',
                            help="The name of actual model for the backbone")
        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default='0.5')
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default='cv2')
        parser.add_argument('--jpg_qual', default='75')

        parser.add_argument('--dataroot', type=str,
                            default='/srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL',
                            help="The root folder of dataset or downloaded data")
        parser.add_argument('--task_name', default='', help='tasks to train on')
        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpuid', nargs="+", type=int, default=[1],
                            help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

        parser.add_argument("--fine_tun", action="store_true", help="fine tunning mode")
        parser.add_argument("--pretrain_name", type=str, default='', help="pre-train model name for fine tunning")
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # additional
        opt.task_name = opt.task_name.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt




class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        ### Basic parameters
        parser.add_argument('--gpu', default='0', help='the index of GPU')
        parser.add_argument('--num_phase', default=3, type=int)
        parser.add_argument('--rootpath', default='/home/yabin/workspace/data/new', type=str)
        parser.add_argument('--ckpt_dir_fg', type=str,
                            default='/home/yabin/workspace/GANIL/logs/_wandb.run.name/iter_0.pth',
                            help='the checkpoint file for the 0-th phase')
        parser.add_argument('--resume_fg', action='store_true', help='resume 0-th phase model from the checkpoint')
        parser.add_argument('--resume', action='store_true', help='resume from the checkpoints')
        parser.add_argument('--num_workers', default=16, type=int, help='the number of workers for loading data')
        parser.add_argument('--random_seed', default=1993, type=int, help='random seed')
        parser.add_argument('--train_batch_size', default=64, type=int, help='the batch size for train loader')
        parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test loader')
        parser.add_argument('--eval_batch_size', default=128, type=int, help='the batch size for validation loader')
        parser.add_argument('--exampler_size', default=128, type=int, help='the batch size for validation loader')
        parser.add_argument('--memory_size', default=1536, type=int, help='the batch size for validation loader')

        parser.add_argument('--data_aug', action='store_true',
                            help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--multiclass', nargs="+", type=int, default=[1],
                            help="whether task have mutilclass dataset")
        parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time', type=int)

        ### Incremental learning parameters
        parser.add_argument('--num_epochs', default=2, type=int, help='Number of epochs')
        parser.add_argument('--add_binary', action='store_true', help='if specified, binary output')
        parser.add_argument('--init_lr', default=0.0005, type=float, help='Init learning rate')

        ### LUCIR parameters
        parser.add_argument('--the_lambda', default=0.5, type=float, help='lamda for LF')
        parser.add_argument('--dist', default=0.2, type=float, help='dist for margin ranking losses')
        parser.add_argument('--K', default=2, type=int, help='K for margin ranking losses')
        parser.add_argument('--lw_mr', default=0.1, type=float, help='loss weight for margin ranking losses')

        ### iCaRL parameters
        parser.add_argument('--icarl_beta', default=0.25, type=float, help='beta for iCaRL')
        parser.add_argument('--icarl_T', default=2, type=int, help='T for iCaRL')
        parser.add_argument('--less_forget', action='store_true', help='Less forgetful')

        ### Enhance Classification parameters
        parser.add_argument("--interpolation", default="bilinear", type=str,
                            help="the interpolation method (default: bilinear)")
        # distributed training parameters
        parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
        parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
        parser.add_argument(
            "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
        parser.add_argument("--model-ema-steps", type=int, default=32,
                            help="the number of iterations that controls how often to update the EMA model (default: 32)", )
        parser.add_argument("--model-ema-decay", type=float, default=0.99998,
                            help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)", )
        parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
        parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
        parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
        parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
                            help="weight decay (default: 1e-4)", dest="weight_decay", )
        parser.add_argument("--norm-weight-decay", default=None, type=float,
                            help="weight decay for Normalization layers (default: None, same value as --wd)", )
        parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)",
                            dest="label_smoothing")
        parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
        parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
        parser.add_argument("--lr-scheduler", default="multisteplr", type=str, help="the lr scheduler (default: steplr)")
        parser.add_argument("--lr-warmup-epochs", default=0, type=int,
                            help="the number of epochs to warmup (default: 0)")
        parser.add_argument("--lr-warmup-method", default="constant", type=str,
                            help="the warmup method (default: constant)")
        parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
        parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
        parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
        ### General learning parameters
        parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
        parser.add_argument('--custom_weight_decay', default=5e-4, type=float,
                            help='weight decay parameter for the optimizer')
        parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
        parser.add_argument('--base_lr', default=0.0001, type=float, help='learning rate for the 0-th phase')
        # parser.add_argument('--binary_loss', default='sum_b_sig', type=str, help='what type of multitask loss')
        parser.add_argument('--binary_loss', default='sum_a_sig', type=str, help='what type of multitask loss')
        parser.add_argument('--binary_weight', default=0.3, type=float, help='what type of multitask loss')

        self.isTrain = True
        return parser



# /data/yabin/data/DeepFake_Data/CL_data
