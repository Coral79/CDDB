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

from options.train_options import TrainOptions
from data import *
from CNND_model import Model_CNND
from options.train_options import TrainOptions
from utils.theano_utils import make_theano_training_function_add_binary,make_theano_training_function_binary,make_theano_validation_function_binary,  \
    make_theano_training_function_mixup,make_theano_training_function_ls,make_theano_validation_function_to_binary

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
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

    args = TrainOptions().parse()
    ######### Modifiable Settings ##########
    batch_size = args.batch_size # Batch size
    # n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
    # nb_val     = 0            # Validation samples per class
    nb_cl      = 2             # Classes per group
    nb_protos  = args.nb_protos             # Number of prototypes per class at the end: total protoset memory/ total number of classes
    epochs     = int(args.num_epochs)     # Total number of epochs
    lr_old     = args.init_lr        # Initial learning rate 0.0005
    lr_strat   = args.schedule       # Epochs where learning rate gets decreased
    lr_factor  = 5.             # Learning rate decrease factor
    wght_decay = 0.00001        # Weight Decay
    nb_runs    = 1              # Number of runs (random ordering of classes at each run)
    # torch.manual_seed(1993)     # Fix the random seed
    ########################################

    # fixed_class_order = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
    #                      94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
    #                      84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
    #                      69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
    #                      17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
    #                       1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
    #                      38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
    #                      60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
    #                      40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
    #                      98, 13, 99,  7, 34, 55, 54, 26, 35, 39]

    # fixed_class_order = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Line 31: Load the dataset
    # Asserting nb_val == 0 equals to full cifar100
    # That is, we only declare transformations here
    # Notes: dstack and reshape already done inside CIFAR100 class
    # Mean is calculated on already scaled (by /255) images

    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # ToTensor scales from [0, 255] to [0, 1.0]
    # ])
    #
    # per_pixel_mean = get_dataset_per_pixel_mean(CIFAR100('./data/cifar100', train=True, download=True,
    #                                                      transform=transform))

    # # https://github.com/srebuffi/iCaRL/blob/90ac1be39c9e055d9dd2fa1b679c0cfb8cf7335a/iCaRL-TheanoLasagne/utils_cifar100.py#L146
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     lambda img_pattern: img_pattern - per_pixel_mean,
    #     icarl_cifar100_augment_data,
    # ])
    #
    # # Must invert previous ToTensor(), otherwise RandomCrop and RandomHorizontalFlip won't work
    # transform_prototypes = transforms.Compose([
    #     icarl_cifar100_augment_data,
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     lambda img_pattern: img_pattern - per_pixel_mean,  # Subtract per-pixel mean
    # ])

    # Line 43: Initialization

    # Line 48: # Launch the different runs
    # Skipped as this script will only manage singe runs

    # Lines 51, 52, 54 already managed in NCProtocol

    # protocol = NCProtocol(CIFAR100('./data/cifar100', train=True, download=True, transform=transform),
    #                       CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
    #                       n_tasks=100//nb_cl, shuffle=True, seed=None, fixed_class_order=fixed_class_order)

    # model: IcarlNet = make_icarl_net(num_classes=100)
    # model.apply(initialize_icarl_net)

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    train_dataset_splits = get_total_Data_multi(args)
    val_opt = get_val_opt()
    val_dataset_splits = get_total_Data_multi(val_opt)
    task_output_space = get_tos_multi(args)
    print(task_output_space)
    # dataset_name = args.task_name

    task_names = args.task_name
    print('Task order:', task_names)
    task_num = len(task_names)
    class_num = task_num*2

    # acc_table = OrderedDict()

    fixed_class_order = list(range(task_num*2))

    protocol = NCProtocol(train_dataset_splits,
                          val_dataset_splits,
                          n_tasks=len(task_names), shuffle=True, seed=None, fixed_class_order=fixed_class_order)
    if args.binary:
        model = Model_CNND(1, args)
    else:
        model = Model_CNND(class_num, args)
    model = model.to(device)

    criterion = BCELoss()  # Line 66-67

    # Line 74, 75
    # Note: sh_lr is a theano "shared"
    sh_lr = lr_old

    # noinspection PyTypeChecker
    val_fn: Callable[[Tensor, Tensor],
                     Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function, model,
                                                              BCELoss(), 'feature_extractor',
                                                              device=device)

    val_fn_to_binary: Callable[[Tensor, Tensor],
                     Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function_to_binary, model,
                                                              BCELoss(), 'feature_extractor',
                                                              device=device)
    
    val_fn_binary: Callable[[Tensor, Tensor],
                    Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function_binary, model,
                                                            BCELoss(), 'feature_extractor',
                                                            device=device)

    # noinspection PyTypeChecker
    function_map: Callable[[Tensor], Tensor] = partial(make_theano_feature_extraction_function, model,
                                                       'feature_extractor', device=device, batch_size=batch_size)

    # Lines 90-97: Initialization of the variables for this run
    # dictionary_size = 500 ###????
    top1_acc_list_cumul = torch.zeros(task_num, 3, task_num)
    top1_acc_list_ori = torch.zeros(task_num, 3, task_num)

    x_protoset_cumuls: List[Tensor] = []
    y_protoset_cumuls: List[Tensor] = []
    # alpha_dr_herding = torch.zeros((task_num, dictionary_size, nb_cl), dtype=torch.float) ###???
    alpha_dr_herding_list = []
    dictionary_size_list = []
    # Lines 101-103: already managed by NCProtocol/NCProtocolIterator

    # train_dataset: Dataset
    # task_info: NCProtocolIterator

    func_pred: Callable[[Tensor], Tensor]
    # func_pred_feat: Callable[[Tensor], Tensor] # Unused
    # for task_idx in range(len(task_names)):
    #     train_name = task_names[task_idx]
    #     print('======================', train_name,
    #           '=======================')
    #     data_size = train_dataset_splits[train_name].__len__()
    #     print("Training Size: ", data_size)
    #     print('Iteration: ', task_idx)
    for task_idx, (train_dataset, task_info) in enumerate(protocol):

        print('Classes in this batch:', task_info.classes_in_this_task)
        print('Data Size:', len(train_dataset))
        dictionary_size = int(len(train_dataset)/2) ###????
        dictionary_size_list.append(dictionary_size)
        alpha_dr_herding_task = torch.zeros((dictionary_size, nb_cl), dtype=torch.float) ###???
        alpha_dr_herding_list.append(alpha_dr_herding_task)

        # if task_idx == 0:
            
        #     alpha_dr_herding = alpha_dr_herding_task
        # else:
        #     alpha_dr_herding = torch.stack((alpha_dr_herding,alpha_dr_herding_task))
        # Lines 107, 108: Save data results at each increment
        # torch.save(top1_acc_list_cumul, 'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
        # torch.save(top1_acc_list_ori, 'top1_acc_list_ori_icarl_cl' + str(nb_cl))
        save_p_list_cul = save_path(args,'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
        save_p_list_ori = save_path(args,'top1_acc_list_ori_icarl_cl' + str(nb_cl))
        torch.save(top1_acc_list_cumul, save_p_list_cul)
        torch.save(top1_acc_list_ori, save_p_list_ori)

        # Note: lines 111-125 already managed in NCProtocol/NCProtocolIterator

        # Lines 128-135: Add the stored exemplars to the training data
        # Note: X_valid_ori and Y_valid_ori already managed in NCProtocol/NCProtocolIterator
        # train_dataset = train_dataset_splits[train_name]
        # if task_idx != 0:
        #     protoset = TensorDataset(torch.cat(x_protoset_cumuls), torch.cat(y_protoset_cumuls))
        #     train_dataset = apd_name(args,train_name,train_dataset, protoset)

        # train_loader = torch.utils.data.DataLoader(train_dataset,
        #                                            batch_size=args.batch_size, shuffle=True,
        #                                            num_workers=int(args.num_threads))
        # test_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
        #                                          batch_size=args.batch_size, shuffle=False,
        #                                          num_workers=int(args.num_threads))
        # Note: X_valid_ori and Y_valid_ori already managed in NCProtocol/NCProtocolIterator
        if task_idx != 0:
            protoset = TransformationDataset(TensorDataset(torch.cat(x_protoset_cumuls), torch.cat(y_protoset_cumuls)), 
                                            target_transform=None)
            train_dataset = ConcatDataset((train_dataset, protoset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        #######################here###############################
        # Line 137: # Launch the training loop
        # From lines: 69, 70, 76, 77
        # Note optimizer == train_fn
        # weight_decay == l2_penalty
        # optimizer = torch.optim.SGD(model.parameters(), lr=sh_lr, weight_decay=wght_decay, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=sh_lr, weight_decay=wght_decay)
        if args.binary:
            train_fn = partial(make_theano_training_function_binary, model, criterion, optimizer, device=device)
        elif args.mixup:
            train_fn_mixup = partial(make_theano_training_function_mixup, model, criterion, optimizer, device=device)
        elif args.label_smoothing:
            train_fn_ls = partial(make_theano_training_function_ls, model, criterion, optimizer, device=device)
        else:
            train_fn = partial(make_theano_training_function, model, criterion, optimizer, device=device)
        train_fn_add_binary = partial(make_theano_training_function_add_binary, model, criterion, optimizer, device=device)
        scheduler = MultiStepLR(optimizer, lr_strat, gamma=0.3)

        print("\n")

        # Added (not found in original code): validation accuracy before first epoch
        if args.binary:
            acc_result, val_err, _, _ = get_accuracy_binary(model, task_info.get_current_test_set(),  device=device,
                                                    required_top_k=[1], return_detailed_outputs=False,
                                                    criterion=BCELoss(), make_one_hot=False, n_classes=2,
                                                    batch_size=batch_size, shuffle=False, num_workers=1)
        else:
            acc_result, val_err, _, _ = get_accuracy(model, task_info.get_current_test_set(), device=device,
                                                    required_top_k=[1, 2], return_detailed_outputs=False,
                                                    criterion=BCELoss(), make_one_hot=True, n_classes=class_num,
                                                    batch_size=batch_size, shuffle=False, num_workers=1)
        print("Before first epoch")
        print("  validation loss:\t\t{:.6f}".format(val_err))  # Note: already averaged
        print("  top 1 accuracy:\t\t{:.2f} %".format(acc_result[0].item() * 100))
        if not args.binary:
            print("  top 2 accuracy:\t\t{:.2f} %".format(acc_result[1].item() * 100))
        # End of added code

        print('Batch of classes number {0} arrives ...'.format(task_idx + 1))

        # Sets model in train mode
        model.train()
        for epoch in range(epochs):
            # Note: already shuffled (line 143-146)

            # Lines 148-150
            train_err: float = 0
            train_batches: int = 0
            start_time: float = time.time()

            patterns: Tensor
            labels: Tensor
            for patterns, labels in train_loader:  # Line 151
                # Lines 153-154
                if args.binary:
                    targets = labels%2
                elif (not args.mixup) and (not args.label_smoothing):
                    targets = make_batch_one_hot(labels, class_num)
                else:
                    targets = labels

                old_train = train_err  # Line 155

                targets = targets.to(device)
                patterns = patterns.to(device)

                if task_idx == 0:   # Line 156
                    if args.add_binary:
                        train_err += train_fn_add_binary(args, patterns, labels)
                    elif args.binary:
                        train_err += train_fn(patterns, targets, targets)  # Line 157
                    elif args.mixup:
                        train_err += train_fn_mixup(args, patterns, targets)
                    elif args.label_smoothing:
                        train_err += train_fn_ls(args, patterns, targets)                        
                    else:
                         train_err += train_fn(patterns, targets)

                # Lines 160-163: Distillation
                if task_idx > 0:
                    prediction_old = func_pred(patterns)
                    if args.binary:
                        targets_old = prediction_old.squeeze(1)
                    elif (not args.mixup) and (not args.label_smoothing):
                        targets[:, task_info.prev_classes] = prediction_old[:, task_info.prev_classes]
                    if args.add_binary:
                        train_err += train_fn_add_binary(args, patterns, labels)
                    elif args.binary:
                        train_err += train_fn(patterns, targets, targets_old)
                    elif args.mixup:
                        train_err += train_fn_mixup(args, patterns, targets, prediction_old)
                    elif args.label_smoothing:
                        train_err += train_fn_ls(args, patterns, targets, prediction_old)                          
                    else:
                        train_err += train_fn(patterns, targets)  

                if (train_batches % 100) == 1:
                    print(train_err - old_train)

                train_batches += 1

            # Lines 171-186: And a full pass over the validation data:
            if args.binary:
                acc_result, val_err, _, _ = get_accuracy_binary(model, task_info.get_current_test_set(),  device=device,
                                                        required_top_k=[1], return_detailed_outputs=False,
                                                        criterion=BCELoss(), make_one_hot=False, n_classes=2,
                                                        batch_size=batch_size, shuffle=False, num_workers=1)
            else:
                acc_result, val_err, _, _ = get_accuracy(model, task_info.get_current_test_set(),  device=device,
                                        required_top_k=[1, 2], return_detailed_outputs=False,
                                        criterion=BCELoss(), make_one_hot=True, n_classes=class_num,
                                        batch_size=batch_size, shuffle=False, num_workers=1)

            # Lines 188-202: Then we print the results for this epoch:
            print("Batch of classes {} out of {} batches".format(
                task_idx + 1, task_num))
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1,
                epochs,
                time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err))  # Note: already averaged
            print("  top 1 accuracy:\t\t{:.2f} %".format(
                acc_result[0].item() * 100))
            if not args.binary:
                print("  top 2 accuracy:\t\t{:.2f} %".format(
                    acc_result[1].item() * 100))
            # adjust learning rate
            scheduler.step()

        # Lines 205-213: Duplicate current network to distillate info
        if task_idx == 0:
            # model2 = make_icarl_net(100, n=n)
            if args.binary:
                model2 = Model_CNND(1, args)
            else:
                model2 = Model_CNND(class_num, args)
            model2 = model2.to(device)
            # noinspection PyTypeChecker
            func_pred = partial(make_theano_inference_function, model2, device=device)

            # Note: func_pred_feat is unused
            # func_pred_feat = partial(make_theano_feature_extraction_function, model=model2,
            #                          feature_extraction_layer='feature_extractor')

        model2.load_state_dict(model.state_dict())

        # Lines 216, 217: Save the network
        
        # torch.save(model.state_dict(), 'net_incr'+str(task_idx+1)+'_of_'+str(len(task_names)))
        # torch.save(model.feature_extractor.state_dict(), 'intermed_incr'+str(task_idx+1)+'_of_'+str(len(task_names)))

        save_p_model = save_path(args,'net_incr'+str(task_idx+1)+'_of_'+str(len(task_names)))
        save_p_feature = save_path(args,'intermed_incr'+str(task_idx+1)+'_of_'+str(len(task_names)))
        torch.save(model.state_dict(), save_p_model)
        torch.save(model.feature_extractor.state_dict(), save_p_feature)

        # Lines 220-242: Exemplars
        # nb_protos_cl = int(ceil(nb_protos * len(task_names) * 2. / nb_cl / (task_idx + 1)))
        # nb_protos_cl = int(ceil(1536. / nb_cl / (task_idx + 1)))
        nb_protos_cl = int(ceil(nb_protos* 1. / nb_cl / (task_idx + 1)))

        # Herding
        print('Updating exemplar set...')
        for iter_dico in range(nb_cl):
            # Possible exemplars in the feature space and projected on the L2 sphere
            prototypes_for_this_class, _ = task_info.swap_transformations() \
                .get_current_training_set()[iter_dico*dictionary_size:(iter_dico+1)*dictionary_size]

            mapped_prototypes: Tensor = function_map(prototypes_for_this_class)
            D: Tensor = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            # Herding procedure : ranking of the potential exemplars
            mu = torch.mean(D, dim=1)
            # alpha_dr_herding[task_idx, :, iter_dico] = alpha_dr_herding[task_idx, :, iter_dico] * 0
            alpha_dr_herding_list[task_idx][ :, iter_dico] = alpha_dr_herding_list[task_idx][ :, iter_dico] * 0
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0
            # while not (torch.sum(alpha_dr_herding[task_idx, :, iter_dico] != 0) ==
            #            min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            #     tmp_t = torch.mm(w_t.unsqueeze(0), D)
            #     ind_max = torch.argmax(tmp_t)
            #     iter_herding_eff += 1
            #     if alpha_dr_herding[task_idx, ind_max, iter_dico] == 0:
            #         alpha_dr_herding[task_idx, ind_max, iter_dico] = 1 + iter_herding
            #         iter_herding += 1
            #     w_t = w_t + mu - D[:, ind_max]
            while not (torch.sum(alpha_dr_herding_list[task_idx][ :, iter_dico] != 0) ==
                       min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
                tmp_t = torch.mm(w_t.unsqueeze(0), D)
                ind_max = torch.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding_list[task_idx][ ind_max, iter_dico] == 0:
                    alpha_dr_herding_list[task_idx][ ind_max, iter_dico] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]            

        # Lines 244-246: Prepare the protoset
        x_protoset_cumuls: List[Tensor] = []
        y_protoset_cumuls: List[Tensor] = []

        # Lines 249-276: Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = torch.zeros((2048, class_num, 2), dtype=torch.float)
        for iteration2 in range(task_idx + 1):
            dictionary_size = dictionary_size_list[iteration2]
            for iter_dico in range(nb_cl):
                prototypes_for_this_class: Tensor
                current_cl = task_info.classes_seen_so_far[list(
                    range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl))]
                current_class = current_cl[iter_dico].item()

                prototypes_for_this_class, _ = task_info.swap_transformations().get_task_training_set(iteration2)[
                                            iter_dico * dictionary_size:(iter_dico + 1)*dictionary_size]

                # Collect data in the feature space for each class
                mapped_prototypes: Tensor = function_map(prototypes_for_this_class)
                D: Tensor = mapped_prototypes.T
                D = D / torch.norm(D, dim=0)

                # Flipped version also
                # PyTorch doesn't support ::-1 yet
                # And with "yet" I mean: PyTorch will NEVER support ::-1
                # See: https://github.com/pytorch/pytorch/issues/229 (<-- year 2016!)
                # Also: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
                mapped_prototypes2: Tensor = function_map(torch.from_numpy(
                    prototypes_for_this_class.numpy()[:, :, :, ::-1].copy()))
                D2: Tensor = mapped_prototypes2.T
                D2 = D2 / torch.norm(D2, dim=0)

                # iCaRL
                # alph = alpha_dr_herding[iteration2, :, iter_dico]
                alph = alpha_dr_herding_list[iteration2][ :, iter_dico]
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.
                print(torch.sum(alph == 1))

                # Adds selected replay patterns
                x_protoset_cumuls.append(prototypes_for_this_class[torch.where(alph == 1)[0]])
                # Appends labels of replay patterns -> Tensor([current_class, current_class, current_class, ...])
                y_protoset_cumuls.append(current_class * torch.ones(len(torch.where(alph == 1)[0])))
                alph = alph / torch.sum(alph)
                class_means[:, current_cl[iter_dico], 0] = (torch.mm(D, alph.unsqueeze(1)).squeeze(1) +
                                                            torch.mm(D2, alph.unsqueeze(1)).squeeze(1)) / 2
                class_means[:, current_cl[iter_dico], 0] /= torch.norm(class_means[:, current_cl[iter_dico], 0])

                # Normal NCM
                alph = torch.ones(dictionary_size) / dictionary_size
                class_means[:, current_cl[iter_dico], 1] = (torch.mm(D, alph.unsqueeze(1)).squeeze(1) +
                                                            torch.mm(D2, alph.unsqueeze(1)).squeeze(1)) / 2

                class_means[:, current_cl[iter_dico], 1] /= torch.norm(class_means[:, current_cl[iter_dico], 1])

        # torch.save(class_means, 'cl_means')  # Line 278

        path_class_means = save_path(args,'cl_means')
        torch.save(class_means, path_class_means)  # Line 278

        # acc_table[task_idx] = OrderedDict()
        # Calculate validation error of model on the first nb_cl classes:
        print('Computing accuracy on the original batch of classes...')
        for i in range(task_idx + 1):
            if args.binary:
                top1_acc_list_ori = icarl_accuracy_measure_binary(task_info.get_task_test_set(i), class_means, val_fn_binary,
                                                        top1_acc_list_ori, task_idx, i, task_names[i],
                                                        make_one_hot=False, n_classes=2,
                                                        batch_size=batch_size, num_workers=2)

            else:
                top1_acc_list_ori = icarl_accuracy_measure(task_info.get_task_test_set(i), class_means, val_fn,
                                                        top1_acc_list_ori, task_idx, i, task_names[i],
                                                        make_one_hot=True, n_classes=class_num,
                                                        batch_size=batch_size, num_workers=1)
                print('Binary accuracy:')
                top1_acc_list_ori = icarl_accuracy_measure_to_binary(task_info.get_task_test_set(i), class_means, val_fn, 
                                                        top1_acc_list_ori, task_idx, i, task_names[i],
                                                        make_one_hot=False, n_classes=class_num,
                                                        batch_size=batch_size, num_workers=1)

        if args.binary:
            top1_acc_list_cumul = icarl_accuracy_measure_binary(task_info.get_cumulative_test_set(), class_means, val_fn_binary,
                                                        top1_acc_list_cumul, task_idx, 0, 'cumul of',
                                                        make_one_hot=False, n_classes=2,
                                                        batch_size=batch_size, num_workers=1)
        else:
            top1_acc_list_cumul = icarl_accuracy_measure(task_info.get_cumulative_test_set(), class_means, val_fn,
                                                        top1_acc_list_cumul, task_idx, 0, 'cumul of',
                                                        make_one_hot=True, n_classes=class_num,
                                                        batch_size=batch_size, num_workers=1)
            print('Binary accuracy:')
            top1_acc_list_cumul = icarl_accuracy_measure_to_binary(task_info.get_cumulative_test_set(), class_means, val_fn,
                                                        top1_acc_list_cumul, task_idx, 0, 'binary cumul of',
                                                        make_one_hot=False, n_classes=class_num,
                                                        batch_size=batch_size, num_workers=1)
    # Final save of the data
    # torch.save(top1_acc_list_cumul, 'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
    # torch.save(top1_acc_list_ori, 'top1_acc_list_ori_icarl_cl' + str(nb_cl))
    print(top1_acc_list_ori)
    print(torch.mean(top1_acc_list_ori[-1],1))

    path_acc_list_cum = save_path(args,'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
    path_acc_list_ori = save_path(args,'top1_acc_list_ori_icarl_cl' + str(nb_cl))
    torch.save(top1_acc_list_cumul, path_acc_list_cum)
    torch.save(top1_acc_list_ori, path_acc_list_ori)


if __name__ == '__main__':
    main()
