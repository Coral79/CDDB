""" Class-incremental learning base trainer. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import default_collate

import os
import os.path as osp
import copy
import math
import warnings
import numpy as np
from typing import Optional, List, Callable, Tuple
from scipy.spatial.distance import cdist

try:
    import cPickle as pickle
except:
    import pickle

# from utils.incremental.compute_features import compute_features_origin_lucia
from data.datasets import getDataSplitFunc, IncrementalDataset
from models import modified_linear
from methods.lucir.inc_func import incremental_train_and_eval
from methods.exemplars.herding import herding_examplers

warnings.filterwarnings('ignore')


class Trainer(object):
    """The class that contains the code for base trainer class.
    This file only contains the related functions used in the training process.
    If you hope to view the overall training process, you may find it in the file named trainer.py in the same folder.
    """

    def __init__(self, the_args, val_args, expname):
        """The function to initialize this class.
        Args:
          the_args: all inputted parameter.
        """
        self.args = the_args
        self.val_args = val_args
        self.expname = expname
        self.set_save_path()
        self.set_cuda_device()
        self.network = self.set_model()

    def set_save_path(self):
        """The function to set the saving path."""
        self.log_dir = './logs/'
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.save_path = self.log_dir + '_' + self.expname
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_model(self):
        from models.modified_resnet import resnet50
        from models.modified_xception import xception
        return resnet50


    def init_current_phase_model(self, iteration, start_iter, tg_model):
        """The function to intialize the models for the current phase
        Args:
          iteration: the iteration index
          start_iter: the iteration index for the 0th phase
          model: current model from last phase
        Returns:
          model: the 1st branch model from the current phase
          ref_model: the 1st branch model from last phase (frozen, not trainable)
          the_lambda_mult, cur_the_lambda: the_lambda-related parameters for the current phase
        """
        if iteration == start_iter:
            # The 0th phase
            tg_model = self.network(num_classes=2) # 0# phase class number
            # Get the information about the input and output features from the network
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            # The 2nd branch and the reference model are not used, set them to None
            ref_model = None
            the_lambda_mult = None
        elif iteration == start_iter + 1:
            # The 1st phase # Update the index for last phase
            ref_model = copy.deepcopy(tg_model)
            # tg_model = self.network(num_classes=2)
            # ref_dict = ref_model.state_dict()
            # tg_dict = tg_model.state_dict()
            # tg_dict.update(ref_dict)
            # tg_model.load_state_dict(tg_dict)
            # tg_model.to(self.device)

            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, 2)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            # Update the lambda parameter for the current phase
            the_lambda_mult = out_features * 1.0 / 2
        else:
            # The i-th phase, i>=2
            ref_model = copy.deepcopy(tg_model)
            # Get the information about the input and output features from the network
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features1 + out_features2)
            # Set the final FC layer for classification
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2, 2)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            # Update the lambda parameter for the current phase
            the_lambda_mult = (out_features1 + out_features2) * 1.0 / 2

        # Update the current lambda value for the current phase
        if iteration > start_iter:
            cur_the_lambda = self.args.the_lambda * math.sqrt(the_lambda_mult)
        else:
            cur_the_lambda = self.args.the_lambda
        return tg_model, ref_model, cur_the_lambda


    def update_train_and_valid_loader(self, train_dict, test_dict, previousProt_dict, previousTest_dict, iteration):
        print('Setting the dataloaders ...')
        trainset = IncrementalDataset(self.args, currentDataDic=train_dict, previousDataDic=previousProt_dict,
                                     iteration=iteration, isTrain=True)

        testset = IncrementalDataset(self.args, currentDataDic=test_dict, previousDataDic=previousTest_dict,
                                     iteration=iteration, isTrain=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batch_size,
                                                  shuffle=True, num_workers=self.args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batch_size,
                                                 shuffle=False, num_workers=self.args.num_workers)
        return trainloader, testloader

    def set_optimizer(self, iteration, start_iter, tg_model):
        if iteration > start_iter:
            ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, tg_model.parameters())
            base_params = filter(lambda p: p.requires_grad, base_params)
            tg_params = [
                {'params': base_params, 'lr': self.args.base_lr, 'weight_decay': self.args.custom_weight_decay},
                {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}, ]
        else:
            tg_params = [{'params': tg_model.parameters(), 'lr': self.args.base_lr,
                          'weight_decay': self.args.custom_weight_decay}, ]

        # tg_params = [{'params': tg_model.fc.fc2.parameters(), 'lr': self.args.base_lr, 'weight_decay': 0}, ]


        # tg_optimizer = optim.SGD(tg_params, lr=self.args.base_lr, momentum=self.args.custom_momentum,
        #                          weight_decay=self.args.custom_weight_decay)
        tg_optimizer = optim.Adam(tg_params, lr=self.args.base_lr, weight_decay=self.args.custom_weight_decay)

        if self.args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(tg_optimizer, step_size=self.args.lr_step_size,
                                                                gamma=self.args.lr_factor)
        elif self.args.lr_scheduler == "multisteplr":
            self.lr_strat = [int(self.args.num_epochs * 0.334), int(self.args.num_epochs * 0.667)]
            print(self.lr_strat)
            main_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=self.lr_strat,
                                                         gamma=self.args.lr_factor)
        elif self.args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                tg_optimizer, T_max=self.args.num_epochs - self.args.lr_warmup_epochs
            )
        elif self.args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(tg_optimizer, gamma=self.args.lr_factor)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{self.args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )
        tg_lr_scheduler = main_lr_scheduler
        return tg_optimizer, tg_lr_scheduler

    def compute_acc(self, tg_model, valloader, iteration, device, top1_acc_list_cumul):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                logits = outputs['logits']
                _, predicted = (logits).max(1)
                total += targets.size(0)

                correct += (predicted%2).eq((targets%2)).sum().item()
        cumul_acc = 100. * correct / total
        top1_acc_list_cumul[iteration, 0] = cumul_acc
        return top1_acc_list_cumul

    def set_exemplar_set(self, tg_model, iteration, traindict, previousProt_dict):
        taskimgs = {}
        for i in traindict.keys():
            if traindict[i] not in taskimgs.keys():
                taskimgs[traindict[i]] = [i]
            else:
                taskimgs[traindict[i]].append(i)

        previousProt_dict, class_means = herding_examplers(self.args, tg_model, iteration, taskimgs, previousProt_dict)
        # import pdb;pdb.set_trace()
        return previousProt_dict, class_means


    def train(self):

        tg_model = None
        ref_model = None
        previousProt_dict = {}
        previousTest_dict = {}
        previousTest_list = []

        task_names = self.args.task_name
        print('Task order:', task_names)
        start_iter = 0

        top1_acc_list_cumul = torch.zeros(len(task_names), 3, 1)
        top1_acc_list_ori = torch.zeros(len(task_names), 3, 1)

        for iteration in range(start_iter, len(task_names)):
            ckp_name = osp.join(self.save_path, 'iter_{}.pth'.format(iteration)) # Set the names for the checkpoints
            print('Check point name: ', ckp_name)
            tg_model, ref_model, cur_the_lambda = self.init_current_phase_model(iteration, start_iter, tg_model)

            train_dict, val_dict = getDataSplitFunc(self.args, task_names[iteration], iteration)
            trainloader, valloader = self.update_train_and_valid_loader(train_dict, val_dict,
                                            previousProt_dict, previousTest_dict, iteration) # Update training and test dataloader

            # if iteration > start_iter:
            #     tg_model = self.imprint_weights(tg_model, trainloader) #TODO: imprint new weights for fc2. for init weights

            if iteration == start_iter and self.args.resume_fg: # Resume the 0-th phase model according to the config
                tg_model = torch.load(self.args.ckpt_dir_fg)
            elif self.args.resume and os.path.exists(ckp_name): # Resume other models according to the config
                tg_model = torch.load(ckp_name)
            else:
                tg_optimizer, tg_lr_scheduler = self.set_optimizer(iteration, start_iter, tg_model) # Set the optimizer
                if iteration > start_iter:
                    ref_model = ref_model.to(self.device)
                tg_model = tg_model.to(self.device)
                tg_model = incremental_train_and_eval(self.args, tg_model, ref_model, tg_optimizer,
                        tg_lr_scheduler, trainloader, valloader, iteration, start_iter, cur_the_lambda)

            # Select the exemplars according to the current model TODO: train_dict just current data, so we need remap labels
            previousProt_dict, class_means = self.set_exemplar_set(tg_model, iteration, train_dict, previousProt_dict)

            for k, v in val_dict.items():
                previousTest_dict[k] = v+iteration*2

            tmp_current = {}
            for k, v in val_dict.items():
                tmp_current[k] = v+iteration*2
            previousTest_list.append(tmp_current)
            # top1_acc_list_cumul = self.compute_acc(tg_model, valloader, iteration, self.device, top1_acc_list_cumul)

            # TODO: new accuracy
            print("\n")
            print('Computing accuracy on the original batch of classes...')
            for i in range(iteration + 1):
                print('Multi accuracy:')
                top1_acc_list_ori = self.accuracy_measure(class_means, previousTest_list[i], tg_model, task_names[i],
                                        top1_acc_list_ori, iteration, 0, self.device)
                print('Binary accuracy:')
                top1_acc_list_ori = self.accuracy_measure_binary(class_means, previousTest_list[i], tg_model, task_names[i],
                                        top1_acc_list_ori, iteration, 0, self.device)

            print('Computing accuracy on the all batch of classes...')
            print('Multi accuracy:')
            top1_acc_list_cumul = self.accuracy_measure(class_means, previousTest_dict, tg_model, 'cumul of',
                                        top1_acc_list_cumul, iteration, 0, self.device)
            print('Binary accuracy:')
            top1_acc_list_cumul = self.accuracy_measure_binary(class_means, previousTest_dict, tg_model, 'binary cumul of',
                                                               top1_acc_list_cumul, iteration, 0, self.device)

            torch.save(tg_model, ckp_name)

            # num_of_testing = iteration - start_iter + 1
            # avg_cumul_acc_fc = np.sum(top1_acc_list_cumul) / num_of_testing
            # print('Computing average accuracy...')
            # print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
        #     wandb.log({
        #         'iteration': iteration,
        #         'Average accuracy (FC)': float(avg_cumul_acc_fc),
        #         'Average accuracy (Proto)': float(avg_cumul_acc_icarl),
        #     })

    def accuracy_measure(self, class_means, testdata, tg_model, type_data: str, top1_acc_list: torch.Tensor,
                          iteration: int, iteration_total: int, device: Optional[torch.device]):

        testset = IncrementalDataset(self.args, currentDataDic=testdata, previousDataDic={}, iteration=0,
                                     isTrain=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batch_size,
                                                 shuffle=False, num_workers=self.args.num_workers)
        total = 0
        correct = 0
        correct_icarl = 0
        correct_ncm = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                logits = outputs['logits']
                outputs_feature = outputs['features']
                _, predicted = (logits).max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # compute icarl
                sqd_icarl = cdist(class_means[:, :, 0].T, outputs_feature.cpu(), 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.eq(targets).sum().item()
                # compute ncm
                sqd_ncm = cdist(class_means[:, :, 1].T, outputs_feature.cpu(), 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.eq(targets).sum().item()


        print("FC accuracy on " + type_data + " classes::\t{:.2f} %".format(100. * correct / total))
        print("Icarl accuracy on " + type_data + " classes::\t{:.2f} %".format(100. * correct_icarl / total))
        print("NCM accuracy on " + type_data + " classes::\t{:.2f} %".format(100. * correct_ncm / total))
        top1_acc_list[iteration, 0, iteration_total] = 100. * correct / total

        return top1_acc_list

    def accuracy_measure_binary(self, class_means, testdata, tg_model, type_data: str, top1_acc_list: torch.Tensor,
                          iteration: int, iteration_total: int, device: Optional[torch.device]):

        testset = IncrementalDataset(self.args, currentDataDic=testdata, previousDataDic={}, iteration=0,
                                     isTrain=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batch_size,
                                                 shuffle=False, num_workers=self.args.num_workers)
        total = 0
        correct = 0
        correct_icarl = 0
        correct_ncm = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets % 2

                outputs = tg_model(inputs)
                logits = outputs['logits']
                outputs_feature = outputs['features']
                _, predicted = (logits).max(1)
                total += targets.size(0)
                correct += (predicted%2).eq((targets%2)).sum().item()

                # compute icarl
                sqd_icarl = cdist(class_means[:, :, 0].T, outputs_feature.cpu(), 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += (predicted_icarl%2).eq((targets%2)).sum().item()
                # compute ncm
                sqd_ncm = cdist(class_means[:, :, 1].T, outputs_feature.cpu(), 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += (predicted_ncm%2).eq((targets%2)).sum().item()

        print("FC accuracy on " + type_data + " classes::\t{:.2f} %".format(100. * correct / total))
        print("Icarl accuracy on " + type_data + " classes::\t{:.2f} %".format(100. * correct_icarl / total))
        print("NCM accuracy on " + type_data + " classes::\t{:.2f} %".format(100. * correct_ncm / total))
        top1_acc_list[iteration, 0, iteration_total] = 100. * correct / total

        return top1_acc_list






    #
    #
    # def imprint_weights(self, tg_model, iteration, traindict):
    #
    #     old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
    #     average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
    #     tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    #     num_features = tg_model.fc.in_features
    #     novel_embedding = torch.zeros((self.args.nb_cl, num_features))
    #
    #
    #     taskimgs = {}
    #     for i in traindict.keys():
    #         if traindict[i] not in taskimgs.keys():
    #             taskimgs[traindict[i]] = [i]
    #         else:
    #             taskimgs[traindict[i]].append(i)

    #     for cls_idx in range(iteration * self.args.nb_cl, (iteration + 1) * self.args.nb_cl):
    #         # Get the indexes of samples for one class
    #         cls_indices = np.array([i == cls_idx for i in map_Y_train])
    #         # Check the number of samples in this class
    #         assert (len(np.where(cls_indices == 1)[0]) <= dictionary_size)
    #         # Set a temporary dataloader for the current class
    #         current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
    #         self.evalset.imgs = self.evalset.samples = current_eval_set
    #         evalloader = torch.utils.data.DataLoader(self.evalset, batch_size=self.args.eval_batch_size,
    #                                                  shuffle=False, num_workers=2)
    #         num_samples = len(X_train[cls_indices])
    #         # Compute the feature maps using the current model
    #         cls_features = compute_features_origin_lucia(tg_feature_model, evalloader, num_samples, num_features)
    #         # Compute the normalized feature maps
    #         norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
    #         # Update the FC weights using the imprint weights, i.e., the normalized averged feature maps
    #         cls_embedding = torch.mean(norm_features, dim=0)
    #         novel_embedding[cls_idx - iteration * self.args.nb_cl] = F.normalize(cls_embedding, p=2,
    #                                                                              dim=0) * average_old_embedding_norm
    #     # Transfer all weights of the model to GPU
    #     tg_model.to(self.device)
    #     tg_model.fc.fc2.weight.data = novel_embedding.to(self.device)
    #
    #     return tg_model
    #


