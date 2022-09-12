from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        # parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        # parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        # parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        # parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        # parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        # parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

        parser.add_argument('--model_weights', type=str, default=None,
                            help="The path to the file for the model weights (*.pth).")
        parser.add_argument('--multiclass', nargs="+", type=int, default=[1],
                            help="whether task have mutilclass dataset")

        parser.add_argument('--outfile', default='temp_0.1.csv', type=str, help='Output file name')
        # parser.add_argument('--matr', default='results/acc_matr.npz', help='Accuracy matrix file name')
        parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time', type=int)
        parser.add_argument('--init_lr', default=0.0005, type=float, help='Init learning rate')

        parser.add_argument('--num_epochs', default=12, type=int, help='Number of epochs')
        parser.add_argument('--add_binary', action='store_true', help='if specified, add binary output')
        parser.add_argument('--binary_weight', type=float, default=0.5, help='binary loss term weight')
        parser.add_argument('--schedule', nargs="+", type=int, default=[10],
                            help="epoch ")
        parser.add_argument('--binary', action='store_true', help='if specified, binary class')
        # mixup
        parser.add_argument('--mixup', action='store_true', help='use mixup augmentation')
        parser.add_argument('--mixup-alpha', type=float, default=0.1, help='mixup alpha value')

        # label smoothing
        parser.add_argument('--label-smoothing', action='store_true', help='use label smoothing')
        parser.add_argument('--smoothing-alpha', type=float, default=0.1, help='label smoothing alpha value')
        parser.add_argument('--binary_loss', default='sum_b_sig', type=str, help='sum_b_sig, sum_a_sig, sum_b_log, max')
        parser.add_argument('--nb_protos', type=int, default=1536,help="memory buffer")
        # parser.add_argument('--model_path')




        self.isTrain = True
        return parser
