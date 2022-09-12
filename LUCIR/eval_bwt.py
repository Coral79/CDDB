import torch
import argparse
import os
import numpy as np

import re
"""
table5 
long sumlogit 
python eval_bwt.py --filepath ./table_long_sum_a_sig_0.3_1500_ep40  --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san
long sumfeat
python eval_bwt.py --filepath ./table_long_sum_b_sig_0.3_1500_ep40  --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san
long multiclass
python eval_bwt.py --filepath ./exp_t5_long_mulclass_1500   --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san
long binary
python eval_bwt.py --filepath ./exp_t5_long_binary_1500   --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild,glow,stargan_gf,stylegan,whichfaceisreal,san

hard sumlogit
python eval_bwt.py --filepath ./table_hard_sumlogit_0.5_1500 --task_name gaugan,biggan,wild,whichfaceisreal,san

v
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--task_name', default='', help='tasks to train on')
parser.add_argument('--filepath', default='', help='log file')
opt, _ = parser.parse_known_args()
task_name = opt.task_name.split(',')

num_tasks = len(task_name)

acc_table = np.zeros([num_tasks, num_tasks])
acc_table_multi = np.zeros([num_tasks, num_tasks])

avg_final_acc = np.zeros(1)
final_bwt = np.zeros(1)
avg_final_multi_acc = np.zeros(1)


for i in range(num_tasks):
    results = []
    results_binar = []
    results_multi = []
    file = open(opt.filepath, 'r')
    for eachline in file:
        if eachline.find('FC accuracy on '+task_name[i]) >= 0:
            results.append(float(re.findall(r"\d+\.?\d*", eachline)[0]))
    results_binar = np.array(results)[1::2]
    results_multi = np.array(results)[::2]
    for j in range(len(results_binar)):
        acc_table[i, j+i] = results_binar[j]
        acc_table_multi[i, j+i] = results_multi[j]

import pdb;pdb.set_trace()

avg_acc_history = [0] * num_tasks
avg_acc_history_multi = [0] * num_tasks
bwt_history = [0] * num_tasks
for i in range(num_tasks):
    train_name = i
    cls_acc_sum = 0
    cls_multi_acc_sum = 0
    backward_transfer = 0
    for j in range(i + 1):
        val_name = j
        cls_acc_sum += acc_table[val_name][train_name]
        cls_multi_acc_sum += acc_table_multi[val_name][train_name]
        backward_transfer += acc_table[val_name][train_name] - acc_table[val_name][val_name]
    avg_acc_history[i] = cls_acc_sum / (i + 1)
    avg_acc_history_multi[i] = cls_multi_acc_sum / (i + 1)
    bwt_history[i] = backward_transfer / i if i > 0 else 0
    print('Task', train_name, 'average acc:', avg_acc_history[i])
    print('Task', train_name, 'average multi acc:', avg_acc_history_multi[i])
    print('Task', train_name, 'backward transfer:', bwt_history[i])


r =0
# Gather the final avg accuracy
avg_final_acc[r] = avg_acc_history[-1]
avg_final_multi_acc[r] = avg_acc_history_multi[-1]
final_bwt[r] = bwt_history[-1]

# Print the summary so far
print('===Summary of experiment repeats:',
         '===')
print('The last avg acc of all repeats:', avg_final_acc)
print('The last avg multi acc of all repeats:', avg_final_multi_acc)
print('The last bwt of all repeats:', final_bwt)
print('acc mean:', avg_final_acc.mean(), 'acc std:', avg_final_acc.std())
print('multi acc mean:', avg_final_multi_acc.mean(), 'multi acc std:', avg_final_multi_acc.std())
