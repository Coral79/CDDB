import time
from functools import partial
from typing import Callable, Tuple, List
from collections import OrderedDict

import numpy as np
import torch
from math import ceil
from torch import Tensor
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100
from cl_dataset_tools import NCProtocol, NCProtocolIterator, TransformationDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

from cl_strategies import icarl_accuracy_measure, icarl_cifar100_augment_data
from cl_strategies.icarl import icarl_accuracy_measure_to_binary, icarl_accuracy_measure_binary
from models import make_icarl_net
from cl_metrics_tools import get_accuracy, get_accuracy_binary
from models.icarl_net import IcarlNet, initialize_icarl_net
from utils import get_dataset_per_pixel_mean, make_theano_training_function, make_theano_validation_function,  \
    make_theano_feature_extraction_function, make_theano_inference_function, make_batch_one_hot

from options.test_options import TestOptions
from data import *
from CNND_model import Model_CNND
from options.train_options import TrainOptions
from utils.theano_utils import make_theano_training_function_add_binary,make_theano_training_function_binary,make_theano_validation_function_binary,  \
    make_theano_training_function_mixup,make_theano_training_function_ls,make_theano_validation_function_to_binary

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score,roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from models.resnet_CNND import resnet50

def get_val_opt():
    val_opt = TestOptions().parse(print_options=False)
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

def save_path(opt, save_filename):
    opt.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_p = os.path.join(opt.save_dir, save_filename)
    # serialize model and optimizer to dict

    return save_p
# def save_networks(model,opt, epoch):
#     save_filename = 'model_epoch_%s.pth' % epoch
#     opt.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
#     save_path = os.path.join(opt.save_dir, save_filename)
#     # serialize model and optimizer to dict
#     state_dict = {
#         'model': model.state_dict(),
#     }

#     torch.save(state_dict, save_path)


def main():
    # This script tries to reproduce results of official iCaRL code
    # https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/main_cifar_100_theano.py

    args = TestOptions().parse()
    ######### Modifiable Settings ##########
    batch_size = args.batch_size # Batch size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # train_dataset_splits = get_total_Data_multi(args)
    val_opt = get_val_opt()
    val_dataset_splits = get_Data_multi(val_opt)
    # val_dataset_splits = get_Data(val_opt)
    # task_output_space = get_tos_multi(args)
    # print(task_output_space)
    # dataset_name = args.task_name

    task_names = args.task_name
    print('Task order:', task_names)
    task_num = len(task_names)
    class_num = task_num*2

    if args.binary:
        model = Model_CNND(1, args)
    else:
        # model = Model_CNND(class_num, args)
        model = resnet50(class_num)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, class_num, bias=False)

    model = model.to(device)

    if args.model_path is not None:
        print('=> Load model weights:', args.model_path)
        model_state = torch.load(args.model_path,
                                    map_location=lambda storage, loc: storage)  # Load to CPU.
        model.load_state_dict(model_state)
        print('=> Load Done')
    plt.figure(1) # 创建图表1
     # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    model.eval()
    acc_total = []
    ap_total = []
    for i in range(task_num):
        task_name = task_names[i]
        print(task_name)
        val_data = val_dataset_splits[task_name]        
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=4)
        target_total = []
        pred_total = []
        stat_hb1: List[bool] = []
        
        for i, (inputs, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                with torch.no_grad():
                    inputs = inputs.cuda()
                    # target = target.cuda()
                    output = model.forward(inputs)
                    ####sum####
                    # output_binary = torch.zeros(output.size(0), 2).to(device)
                    # task_num = int(output.size(1)/2)
                    # for i in range(task_num):
                    #     output_binary[:, 0] += output[:, i*2] 
                    #     output_binary[:, 1] += output[:, i*2+1]
                    # output_binary = torch.sigmoid(output_binary)
                    # output_binary = output_binary/torch.sum(output_binary, 1, keepdim = True)
                    # print(output_binary,y_binary)
                    ####max####
                    if args.binary:
                        y_pred = output.flatten().tolist()
                        y_true = (target%2).flatten().tolist()
                        y_true, y_pred = np.array(y_true), np.array(y_pred)
                        # acc = accuracy_score(y_true, y_pred > 0.5)
                        # print(acc)
                        # pred_total.append(output.detach().cpu())
                        # target_total.append(target%2)
                        # acc_total.append(acc)
                        pred_total.extend(y_pred)
                        target_total.extend(y_true)
                    else:
                        output_norm = output/torch.sum(output, 1, keepdim = True)
                        output_norm = output_norm.detach().cpu()
                        # output_max,_ = torch.max(output_norm, 1)
                        # # print(output_max)
                        # pred_total.append(output_max)
                        # # print(pred_total)
                        # target_total.append(target%2)
                        # # print(target_total)
                        output = output.detach().cpu() 
                        task = int(output.size(1)/2)
                        output_real = torch.zeros(output.size(0), task)
                        output_fake = torch.zeros(output.size(0), task)
                        stat_hb1 += (
                            [(ll%2) in (best%2) for ll, best in zip(target, torch.argsort(output, dim=1)[:, -1:])])
                        # print(stat_hb1)
                        for i in range(task):
                            output_real[:, i] = output[:, i*2]
                            output_fake[:, i] = output[:, i*2+1]
                        output_max_real,_ = torch.max(output_real, 1)
                        output_max_fake,_ = torch.max(output_fake, 1)
                        pred = torch.div(output_max_fake, (output_max_real+output_max_fake))
                        # print(output_max_fake, (output_max_real+output_max_fake), pred, target)
                        pred_total.append(pred)

                        # print(pred_total)
                        target_total.append(target%2)
                        # print(target_total)

        if args.binary:
            # acc_total = np.array(acc_total)
            # print(acc_total)
            # print('Avg ACC:', np.mean(acc_total))
            pred_total_np, target_total_np = np.array(pred_total), np.array(target_total)
            # r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
            # f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
            acc = accuracy_score(target_total_np, pred_total_np > 0.5)
            # ap = average_precision_score(target_total_np, pred_total_np)
            print('Avg ACC:', acc)
            acc_total.append(acc)
            # print("AP:",ap)

        else:
            stat_hb1_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_hb1])
            print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(torch.mean(stat_hb1_numerical) * 100))
            pred_total = torch.cat(pred_total, 0)
            # print(pred_total)
            target_total = torch.cat(target_total,0)        
            min, min_index = torch.min(pred_total,0)
            print(min, target_total[min_index])
            print(pred_total.size())
            print(target_total.size())

        precision, recall, thresholds = precision_recall_curve(target_total, pred_total)
        # fpr, tpr, _ = roc_curve(target_total, pred_total, pos_label=clf.classes_[1])
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot() 
        # plt.figure(1)
        # # plt.plot(precision)
        # # plt.figure(2)
        # plt.plot(recall, precision)
        # min = pred_total_np.min()
        # print(recall[recall == 0.9][0], precision[recall == 0.9][0], thresholds[(recall == 0.9)-1])
        # r_acc = accuracy_score(target_total_np[target_total_np==0], pred_total_np[target_total_np==0] >= min)
        # f_acc = accuracy_score(target_total_np[target_total_np==1], pred_total_np[target_total_np==1] >= min)
        # print(r_acc,f_acc)
        # print(recall[1], precision[1])

        # acc1 = accuracy_score(target_total_np, pred_total_np > thresholds[(recall == 0.9)][0])
        # print('racll=0 ACC=:', acc1)

        # plt.plot(recall)
        map = average_precision_score(target_total, pred_total)
        ap_total.append(map)
        print("AP:",map)
        auc = roc_auc_score(target_total, pred_total)
        print("AUC:",auc)
    
    if args.binary:
        acc_total = np.array(acc_total)
        print(acc_total)
        print('Avg ACC:', np.mean(acc_total))
    ap_total = np.array(ap_total)
    print(ap_total)
    print('Avg AP:', np.mean(ap_total))
        # plt.savefig('joint_p-r_{}.png'.format(task_name))
        # print('Saved!!!')
        # print(thresholds)







    

if __name__ == '__main__':
    main()
