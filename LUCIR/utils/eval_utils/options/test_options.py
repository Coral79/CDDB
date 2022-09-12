from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--multiclass', nargs="+", type=int, default=[1],
                            help="whether task have mutilclass dataset")
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--binary', action='store_true', help='if specified, binary class')
        parser.add_argument('--model_weights', type=str, default=None,
                            help="The path to the file for the model weights (*.pth).")
        self.isTrain = False
        return parser
