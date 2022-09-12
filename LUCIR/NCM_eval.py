import os

import torch
import torch.nn as nn


from methods.lucir.inc_func import incremental_train_and_eval
from methods.exemplars.herding import herding_examplers

from configs import TrainOptions
from data.datasets import getDataSplitFunc, IncrementalDataset


# python lucir_main.py --name icarl_df --checkpoints_dir ./checkpoints  --dataroot /home/yabin/workspace/data/DeepFake_Data/CL_data/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass 0 0 1 0 0 0 0 --batch_size 32 --num_epochs 50 > ep50lr1e-4_cosinefc



# weight_root = './logs/_2022-02-15-22_12_40' # norm
weight_root = './logs/_2022-02-15-22_12_07' # cosine

args = TrainOptions().parse()

task_names = args.task_name
print('Task order:', task_names)


class_features = {}
for iteration in range(0, len(task_names)):
    train_dict, val_dict = getDataSplitFunc(args, task_names[iteration], iteration)
    tg_model = torch.load(os.path.join(weight_root, 'iter_{}.pth'.format(iteration)))
    taskimgs = {}
    for i in train_dict.keys():
        if train_dict[i] not in taskimgs.keys():
            taskimgs[train_dict[i]] = [i]
        else:
            taskimgs[train_dict[i]].append(i)

    previousProt_dict = herding_examplers(args, tg_model, iteration, taskimgs, {})

    testset = IncrementalDataset(args, currentDataDic=previousProt_dict, previousDataDic={},
                                 iteration=0, isTrain=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=args.num_workers)



    tg_model = torch.load(os.path.join(weight_root, 'iter_{}.pth'.format(len(task_names)-1)))
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    tg_feature_model.eval()
    # import pdb;pdb.set_trace()
    for batch_idx, (inputs, targets, path) in enumerate(testloader):
        # Get a batch of training samples, transfer them to the device
        inputs, targets = inputs.cuda(), targets.cuda()
        feature = tg_feature_model(inputs)
        norm_feature = feature.squeeze()/torch.norm(feature)

        # import pdb;pdb.set_trace()
        if targets.item() not in class_features.keys():
            class_features[targets.item()] = [norm_feature.detach().cpu().numpy()]
        else:
            class_features[targets.item()].append(norm_feature.detach().cpu().numpy())

        # import pdb;pdb.set_trace()

import numpy as np
class_means = []
for iteration in range(0, 2*len(task_names)):
    np_features = np.array(class_features[iteration])
    classmean = np_features.mean(0)
    class_means.append(classmean)

class_means = np.array(class_means)


tg_model = torch.load(os.path.join(weight_root, 'iter_{}.pth'.format(len(task_names) - 1)))

for iteration in range(0, len(task_names)):
    train_dict, val_dict = getDataSplitFunc(args, task_names[iteration], iteration)

    testset = IncrementalDataset(args, currentDataDic=val_dict, previousDataDic={}, iteration=0,isTrain=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = tg_model(inputs)
            logits = outputs['logits']
            features = outputs['features'] 
            features = features / torch.norm(features)
            pred = features.cpu().numpy().dot(class_means.T).argmax(1)

            total += targets.size(0)
            correct += ((pred)==((targets).cpu().numpy())).sum()
            # correct += ((pred%2)==((targets%2).cpu().numpy())).sum()
    print("NCM accuracy on " + str(task_names[iteration]) + "::\t{:.2f} %".format(100. * correct / total))
