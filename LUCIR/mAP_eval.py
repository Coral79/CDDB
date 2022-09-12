# /home/ubuntu/efs/L2/Lifelonglearning/Null_space/eval_new.py

import os
import csv
import torch
import torch.nn as nn
from types import MethodType
import numpy as np
from utils.eval_utils.data import *

from utils.eval_utils.options.test_options import TestOptions
from utils.eval_utils.eval_config import *
from models.modified_resnet import resnet50

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score,roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

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

# Running tests
args = TestOptions().parse(print_options=False)
batch_size = args.batch_size # Batch size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

val_opt = get_val_opt()
val_dataset_splits = get_Data_multi(val_opt)

task_names = args.task_name
print('Task order:', task_names)
task_num = len(task_names)
class_num = task_num*2


# tg_model = resnet50(num_classes=2*task_num)
"""
python mAP_eval.py --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san --multiclass  0 0 1 0 0 0 0 1 1 1 0 0
CUDA_VISIBLE_DEVICES=1 python mAP_eval.py --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name gaugan,biggan,wild,whichfaceisreal,san --multiclass  0 0 0 0 0
python mAP_eval.py --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0
 
python mAP_eval.py --dataroot /home/ubuntu/efs/DeepFake_Data/CL_data/ --task_name gaugan,biggan,wild,whichfaceisreal,san --multiclass  0 0 0 0 0
python mAP_eval.py --dataroot /home/ubuntu/efs/DeepFake_Data/CL_data/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0
python mAP_eval.py --dataroot /home/ubuntu/efs/DeepFake_Data/CL_data/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san --multiclass  0 0 1 0 0 0 0 1 1 1 0 0
 
# 在洪老师服务器做
long sumlogit ./logs/_2022-03-06-00_57_51/iter_0.pth
 ./table_long_sum_a_sig_0.3_1500_ep40  
long sumfeat ./logs/_2022-03-05-11_05_36/iter_0.pth
./table_long_sum_b_sig_0.3_1500_ep40  
long multiclass ./logs/_2022-03-03-22_47_47/iter_0.pth
exp_t5_long_mulclass_1500   
long binary ./logs/_2022-03-03-11_32_33/iter_0.pth
exp_t5_long_binary_1500  

hard binary 1500 ./logs/_2022-03-03-01_56_47/iter_0.pth
./exp_t5_hard_binary_1500  
hard sumlogit 1000 ./logs/_2022-03-04-04_12_35/iter_0.pth
table_hard_sumlogit_0.3_1000 
hard sumfeature 1000 ./logs/_2022-03-04-02_54_04/iter_0.pth
table_hard_sum_b_sig_0.3_1000   
hard sumlogit 500  ./logs/_2022-03-04-08_05_25/iter_0.pth
./table_hard_sumlogit_0.3_500   
hard sumfeature 500  ./logs/_2022-03-04-05_38_45/iter_0.pth
./table_hard_sum_b_sig_0.3_500   

easy ganfake ./logs/_2022-02-26-09_47_26/iter_0.pth
./ep50_fix_sumfeature_0.3 
easy multclass ./logs/_2022-02-22-18_09_28/iter_0.pth
./ep30_fixmem  
easy binary ./logs/_2022-02-21-23_29_17/iter_0.pth
./lucir_binary 

# 在aws做
hard sumlogit 1500  ./logs/_2022-03-05-05_44_23/iter_4.pth
./table_hard_sum_a_sig_0.3_1500_ep40
hard sumfeature 1500  ./logs/_2022-03-05-05_55_01/iter_4.pth
./table_hard_sum_b_sig_0.3_1500_ep40
hard multclass 1500  ./logs/_2022-03-03-04_36_34/iter_4.pth
 ./exp_t5_hard_mulclass_1500 
hard multclass 1000 ./logs/_2022-03-03-06_46_58/iter_4.pth
 ./exp_t5_hard_mulclass_1000 
hard multclass 500 ./logs/_2022-03-03-08_52_03/iter_4.pth
./exp_t5_hard_mulclass_500
hard binary 500 ./logs/_2022-03-03-06_34_56/iter_4.pth
./exp_t5_hard_binary_500 
hard binary 1000    ./logs/_2022-03-03-04_31_25/iter_4.pth
./exp_t5_hard_binary_1000 
 
easy max  ./logs/_2022-02-26-20_11_35/iter_4.pth
./ep50_fix_max_0.3
easy sumlogit ./logs/_2022-02-26-20_21_42/iter_4.pth
./ep50_fix_sum_a_sig_0.3 
easy sumfeature  ./logs/_2022-02-26-20_27_59/iter_4.pth
./ep50_fix_sum_b_sig_0.3

"""
strings = {
    'long sumlogit': './logs/_2022-03-06-00_57_51/iter_11.pth',
    # 'table_hard_sum_a_sig_0.3_1500_ep40': './logs/_2022-03-05-05_44_23/iter_4.pth',
    # 'table_hard_sum_a_sig_0.3_1000_ep40': './logs/_2022-03-05-07_31_21/iter_4.pth',
    # 'table_hard_sum_a_sig_0.3_500_ep40': './logs/_2022-03-05-09_15_22/iter_4.pth',
    # 'table_hard_sum_b_sig_0.3_1500_ep40': './logs/_2022-03-05-05_55_01/iter_4.pth',
    # 'table_hard_sum_b_sig_0.3_1000_ep40': './logs/_2022-03-05-07_41_48/iter_4.pth',
    # 'table_hard_sum_b_sig_0.3_500_ep40': './logs/_2022-03-05-09_23_21/iter_4.pth',
    # 'lucir-max': './logs/_2022-02-26-20_11_35/iter_6.pth',
    # 'lucir-sumfeat': './logs/_2022-02-26-20_27_59/iter_6.pth',
    # 'lucir-sumlogit': './logs/_2022-02-26-20_21_42/iter_6.pth',
    # 'lucir-sumlog': './logs/_2022-03-05-09_23_21/iter_4.pth',
    # 'lucir-linfc': './logs/_2022-03-05-09_23_21/iter_4.pth',

    # 'hard binary 1500': 0.8972381430388652, 'hard sumlogit 1000': 0.8814136605965001, 'hard sumfeature 1000': 0.8893155539201398, 'hard sumlogit 500': 0.8808131661353251, 'hard sumfeature 500': 0.8784138595478529
    # 'hard binary 1500': './logs/_2022-03-03-01_56_47/iter_4.pth',
    # 'hard sumlogit 1000': './logs/_2022-03-04-04_12_35/iter_4.pth',
    # 'hard sumfeature 1000': './logs/_2022-03-04-02_54_04/iter_4.pth',
    # 'hard sumlogit 500':  './logs/_2022-03-04-08_05_25/iter_4.pth',
    # 'hard sumfeature 500':  './logs/_2022-03-04-05_38_45/iter_4.pth',
    # {'easy ganfake': 0.9546685176415854, 'easy multclass': 0.9475019922365722, 'easy binary': 0.959358084045376}
    # 'easy ganfake': './logs/_2022-02-26-09_47_26/iter_6.pth',
    # 'easy multclass': './logs/_2022-02-22-18_09_28/iter_6.pth',
    # 'easy binary': './logs/_2022-02-21-23_29_17/iter_6.pth',

    # {'hard sumlogit 1500': 0.8787873826633226, 'hard sumfeature 1500': 0.8739373160332388, 'hard multclass 1500': 0.8673166068136979, 'hard multclass 1000': 0.8596580103080864, 'hard multclass 500': 0.8486898793847903, 'hard binary 500': 0.8581326366570995, 'hard binary 1000': 0.8846308484465748}
    # 'hard sumlogit 1500':  './logs/_2022-03-05-05_44_23/iter_4.pth',
    # 'hard sumfeature 1500':  './logs/_2022-03-05-05_55_01/iter_4.pth',
    # 'hard multclass 1500':  './logs/_2022-03-03-04_36_34/iter_4.pth',
    # 'hard multclass 1000': './logs/_2022-03-03-06_46_58/iter_4.pth',
    # 'hard multclass 500': './logs/_2022-03-03-08_52_03/iter_4.pth',
    # 'hard binary 500': './logs/_2022-03-03-06_34_56/iter_4.pth',
    # 'hard binary 1000': './logs/_2022-03-03-04_31_25/iter_4.pth',

    # 'easy max': 0.9540717182777536, 'easy sumlogit': 0.9526403546271569, 'easy sumfeature': 0.9566151972120929}
    # 'easy max':  './logs/_2022-02-26-20_11_35/iter_6.pth',
    # 'easy sumlogit': './logs/_2022-02-26-20_21_42/iter_6.pth',
    # 'easy sumfeature':  './logs/_2022-02-26-20_27_59/iter_6.pth',

    # supplementary data hard
# python mAP_eval.py --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name whichfaceisreal,biggan,wild,san,gaugan --multiclass  0 0 0 0 0 #2sq
# python mAP_eval.py --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name wild,san,gaugan,whichfaceisreal,biggan --multiclass  0 0 0 0 0 #3sq
#     'hard binary 2sq': './logs/_2022-03-13-00_56_17/iter_4.pth', # hardtask_binary_1500_2sq
#     'hard multiclass 2sq': './logs/_2022-03-11-21_49_46/iter_4.pth', # hardtask_multiclass_1500_2sq
#     'hard sumlogit 2sq': "./logs/_2022-03-11-14_25_46/iter_4.pth",  #hardtask_sum_a_sig_1500_2sq
# {'hard binary 2sq': 0.8897723795783923, 'hard multiclass 2sq': 0.912972533162461, 'hard sumlogit 2sq': 0.8979287680259553}

    # 'hard binary 3sq': './logs/_2022-03-13-04_19_13/iter_4.pth', #  hardtask_binary_1500_3sq
    # 'hard multiclass 3sq': './logs/_2022-03-11-19_07_59/iter_4.pth', #  hardtask_multiclass_1500_3sq
    # 'hard sumlogit 3sq': './logs/_2022-03-11-16_55_34/iter_4.pth' #  hardtask_sum_a_sig_1500_3sq
# {'hard binary 3sq': 0.8742020116625788, 'hard multiclass 3sq': 0.8957602234084323, 'hard sumlogit 3sq': 0.8860994488463412}

}

_res_={}
for k,v in strings.items():
    model = torch.load(v)

    model.to(device)
    plt.figure(1) # 创建图表1
    # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    acc_total = []
    ap_total = []
    model.eval()
    for i in range(task_num):
        task_name = task_names[i]
        print(task_name)
        val_data = val_dataset_splits[task_name]
        val_loader = torch.utils.data.DataLoader(val_data,
                                                    batch_size=args.batch_size, shuffle=False, num_workers=4)
        target_total = []
        pred_total = []
        stat_hb1 = []

        for i, (inputs, target, task) in enumerate(val_loader):
            if torch.cuda.is_available():
                with torch.no_grad():
                    inputs = inputs.cuda()
                    output = model(inputs)
                    if args.binary:
                        output = output['logits']
                        y_pred = output.sigmoid().flatten().tolist()
                        y_true = (target%2).flatten().tolist()
                        y_true, y_pred = np.array(y_true), np.array(y_pred)
                        pred_total.extend(y_pred)
                        target_total.extend(y_true)
                    else:
                        m = torch.nn.Softmax()
                        output = m(output['logits'])
                        # import pdb;pdb.set_trace()

                        output_norm = output/torch.sum(output, 1, keepdim = True)
                        output_norm = output_norm.detach().cpu()
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
                        # print((target%2),pred)
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
            pred_total_np, target_total_np = np.array(pred_total), np.array(target_total)

            # print(pred_total)
            target_total = torch.cat(target_total,0)
            min, min_index = torch.min(pred_total,0)
            print(min, target_total[min_index])
            print(pred_total.size())
            print(target_total.size())

        precision, recall, thresholds = precision_recall_curve(target_total, pred_total)
    #     fpr, tpr, _ = roc_curve(target_total, pred_total, pos_label=clf.classes_[1])
    #     roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.figure(1)
    #     # plt.plot(precision)
    #     # plt.figure(2)
        plt.plot(recall, precision)
    #     min = pred_total_np.min()
    #     print(recall[recall == 0.9][0], precision[recall == 0.9][0], thresholds[(recall == 0.9)-1])
    #     r_acc = accuracy_score(target_total_np[target_total_np==0], pred_total_np[target_total_np==0] >= min)
    #     f_acc = accuracy_score(target_total_np[target_total_np==1], pred_total_np[target_total_np==1] >= min)
    #     print(r_acc,f_acc)
    #     print(recall[1], precision[1])
    #
    #     acc1 = accuracy_score(target_total_np, pred_total_np > thresholds[(recall == 0.9)][0])
    #     print('racll=0 ACC=:', acc1)
    #
    #     plt.plot(recall)
        map = average_precision_score(target_total, pred_total)
        ap_total.append(map)
    #     print("AP:",map)
    #     auc = roc_auc_score(target_total, pred_total)
    #     print("AUC:",auc)
        plt.savefig('long_1500-r_{}.png'.format(task_name))
    # print('Saved!!!')

    if args.binary:
        acc_total = np.array(acc_total)
        print(acc_total)
        print('Avg ACC:', np.mean(acc_total))


    ap_total = np.array(ap_total)
    print(ap_total)
    print('Avg AP:', np.mean(ap_total))
    _res_[k]=np.mean(ap_total)

print(_res_)