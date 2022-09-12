import torch
import argparse
import os
import numpy as np

# def save_path(opt, save_filename):
#     opt.save_dir = os.path.join('/srv/beegfs02/scratch/generative_modeling/data/Deepfake/Adam-NSCL/checkpoints/', opt.name)
#     save_p = os.path.join(opt.save_dir, save_filename)
    
#     return save_p

# def parse_option():
#     parser = argparse.ArgumentParser('argument for training')


#     # root folders
#     parser.add_argument('--name', type=str, default='icarl_df_5_binary', help='root directory of dataset')
#     args = parser.parse_args()
#     return args

avg_final_acc = np.zeros(1)
final_bwt = np.zeros(1)
# args = parse_option()
# path_acc_list_ori = save_path(args,'top1_acc_list_ori_icarl_cl' + str(2))
# top1_acc_list_ori = torch.load(path_acc_list_ori)
# print(top1_acc_list_ori[-1])
# num_task = top1_acc_list_ori.size(0)

# acc_table = top1_acc_list_ori.T
acc_table = np.zeros((7,7))
num_task = 7
acc_table[:,-1] = [ 58.60, 65.25, 61.83, 98.43, 88.54, 98.43, 72.52]
acc_table[:,-2] = [ 65.5, 74, 91.79, 99.88, 96.03, 99.92, 0.0 ]
acc_table[:,-3] = [ 60.25, 72.25, 90.46, 99.65, 85.21, 0.0, 0.0]
acc_table[:,-4] = [ 83.55, 92.38, 96.18, 99.96, 0.0, 0.0, 0.0]
acc_table[:,-5] = [ 94.7, 95.25, 97.14, 0.0, 0.0, 0.0, 0.0]
acc_table[:,-6] = [ 97.7, 98.12, 0.0, 0.0, 0.0, 0.0, 0.0]
acc_table[:,-7] = [ 99.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
avg_acc_history = [0] * num_task
bwt_history = [0] * num_task
for i in range(num_task):
    train_name = i
    cls_acc_sum = 0
    backward_transfer = 0
    for j in range(i + 1):
        val_name = j
        cls_acc_sum += acc_table[val_name][train_name]
        backward_transfer += acc_table[val_name][train_name] - \
            acc_table[val_name][val_name]
        # cls_acc_sum += acc_table[val_name][1][train_name]
        # backward_transfer += acc_table[val_name][1][train_name] - \
        #     acc_table[val_name][1][val_name]
    avg_acc_history[i] = cls_acc_sum / (i + 1)
    bwt_history[i] = backward_transfer / i if i > 0 else 0
    print('Task', train_name, 'average acc:', avg_acc_history[i])
    print('Task', train_name, 'backward transfer:', bwt_history[i])

r =0
# Gather the final avg accuracy
avg_final_acc[r] = avg_acc_history[-1]
final_bwt[r] = bwt_history[-1]

# Print the summary so far
print('===Summary of experiment repeats:',
         '===')
print('The last avg acc of all repeats:', avg_final_acc)
print('The last bwt of all repeats:', final_bwt)
print('acc mean:', avg_final_acc.mean(),
        'acc std:', avg_final_acc.std())
print('bwt mean:', final_bwt.mean(), 'bwt std:', final_bwt.std())