import torch
import os
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from data.wrapper import Subclass, AppendName, CacheClassLabel1,CacheClassLabel_multi

from .datasets import dataset_folder


def get_dataset1(opt, name, id):
    dset_lst = []
    if opt.isTrain:
        # root = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)
        root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)
        opt.classes = os.listdir(root_) if opt.multiclass[id] else ['']
        for cls in opt.classes:
            root = root_ + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    else:
        # root = opt.dataroot + '/' + name + '/{}/'.format(opt.val_split)
        root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.val_split)
        opt.classes = os.listdir(root_) if opt.multiclass[id] else ['']
        for cls in opt.classes:
            root = root_ + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    # dset = dataset_folder(opt, root)
    # dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)

def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def get_Data(opt,remap_class=False):
    class_list = [0,1]
    dataset_splits = {}
    for id, name in enumerate(opt.task_name):
        dataset = get_dataset1(opt, name, id)
        dataset = CacheClassLabel1(dataset)
        dataset_splits[name] = AppendName(
            Subclass(dataset, class_list, remap_class), name)

    # shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    # sampler = get_bal_sampler(dataset) if opt.class_bal else None
    #
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=opt.batch_size,
    #                                           shuffle=shuffle,
    #                                           sampler=sampler,
    #                                           num_workers=int(opt.num_threads))
    return dataset_splits


def get_Data_multi(opt,remap_class=False):
    class_list = range(2*len(opt.task_name))
    dataset_splits = {}
    for id, name in enumerate(opt.task_name):
        dataset = get_dataset1(opt, name, id)
        dataset = CacheClassLabel_multi(dataset,id)
        dataset_splits[name] = dataset
        # dataset_splits[name] = AppendName(
        #     Subclass(dataset, class_list, remap_class), name)

    # shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    # sampler = get_bal_sampler(dataset) if opt.class_bal else None
    #
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=opt.batch_size,
    #                                           shuffle=shuffle,
    #                                           sampler=sampler,
    #                                           num_workers=int(opt.num_threads))
    return dataset_splits

def apd_name(opt,name,dataset_splits,data_add,remap_class=False):
    class_list = range(2*len(opt.task_name))
    dataset = dataset_splits[name]
    data_add = CacheClassLabel1(data_add)
    dataset = torch.utils.data.ConcatDataset((dataset, data_add))
    dataset_splits[name] = AppendName(
        Subclass(dataset, class_list, remap_class), name)

    # shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    # sampler = get_bal_sampler(dataset) if opt.class_bal else None
    #
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=opt.batch_size,
    #                                           shuffle=shuffle,
    #                                           sampler=sampler,
    #                                           num_workers=int(opt.num_threads))
    return dataset_splits

def get_total_Data_multi(opt,remap_class=False):
    class_list = range(2*len(opt.task_name))
    # dataset_splits = {}
    dataset_total=[]
    for id, name in enumerate(opt.task_name):
        dataset = get_dataset1(opt, name, id)
        dataset = CacheClassLabel_multi(dataset, id)
        dataset_total.append(dataset)
        # if id == 0:
        #     dataset_total = dataset
        # else:
        #     dataset_total = torch.utils.data.ConcatDataset((dataset_total, dataset))

    dataset_total = torch.utils.data.ConcatDataset(dataset_total)
    # import pdb;pdb.set_trace()
    return dataset_total

def get_total_Data_sep(opt,remap_class=False):
    """
    same to get_total_Data_multi but no concatDataset
    """
    # dataset_splits = {}
    dataset_total=[]
    for id, name in enumerate(opt.task_name):
        dataset = get_dataset1(opt, name, id)
        dataset = CacheClassLabel_multi(dataset, id)
        dataset_total.append(dataset)
    return dataset_total



def get_tos(opt):
    task_output_space = {}
    for name in opt.task_name:
        task_output_space[name] = 1
    return task_output_space

def get_tos_multi(opt):
    task_output_space = {}
    # for name in opt.task_name:
    task_output_space['All'] = len(opt.task_name)*2
    return task_output_space


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
