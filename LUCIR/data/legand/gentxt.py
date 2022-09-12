""""
Generate data split txt file.
train
val
test
"""
import os
import random

settings = {
    "tasks_names": ['CycleGAN', 'GGAN256', 'GGAN1024', 'glow', 'starGAN', 'dcgan'],
    "GGAN256": ['celeba256', 'lsun_bicycle', 'lsun_churchoutdoor', 'lsun_tower', 'lsun_bedroom', 'lsun_bridge', 'lsun_kitchen'],
    "GGAN1024": ['celebhq'],
    "CycleGAN": ['apple2orange', 'monet2photo', 'photo2ukiyoe', 'winter2summer', 'cityscapes', 'orange2apple', 'photo2vangogh',
                'zebra2horse', 'facades', 'photo2cezanne', 'sats', 'horse2zebra', 'photo2monet', 'summer2winter'],
    "glow": ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Smiling'],
    "starGAN": ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Smiling'],
    "dcgan": ['celeba64'],
}

rootpath = '/home/yabin/workspace/data/new'


def gen_CycleGAN_Train_Val_Test(rootpath = rootpath):
    """
    rootpath : ie：/model/yabin/datasets/new
                which has subdir: Generated  Pristine
    """
    train = ['apple2orange', 'orange2apple', 'horse2zebra', 'zebra2horse', 'monet2photo', 'photo2cezanne', 'photo2monet']
    test = ['photo2ukiyoe', 'photo2vangogh']
    false = 'Generated'
    true = 'Pristine'
    total_train = {}
    total_test = {}
    for i in train:
        for imgname in os.listdir(os.path.join(rootpath, false, 'CycleGAN', i)):
            total_train[(os.path.join(false, 'CycleGAN', i, imgname))] = 0
    for i in train:
        for imgname in os.listdir(os.path.join(rootpath, true, 'CycleGAN', i)):
            total_train[os.path.join(true, 'CycleGAN', i, imgname)] = 1
    for i in test:
        for imgname in os.listdir(os.path.join(rootpath, false, 'CycleGAN', i)):
            total_test[(os.path.join(false, 'CycleGAN', i, imgname))] = 0
    for i in test:
        for imgname in os.listdir(os.path.join(rootpath, true, 'CycleGAN', i)):
            total_test[os.path.join(true, 'CycleGAN', i, imgname)] = 1

    # split train to train and val
    total_train_list = list(total_train.keys())
    numtrain = len(total_train_list)
    random.shuffle(total_train_list)
    train_split = total_train_list[:int(numtrain*0.8)]
    val_split = total_train_list[int(numtrain*0.8):]

    train_dict = {}
    val_dict = {}
    for i in train_split:
        train_dict[i] = total_train[i]
    for i in val_split:
        val_dict[i] = total_train[i]
    return train_dict, val_dict, total_test


def gen_GGAN256_Train_Val_Test(rootpath = rootpath):
    """
    rootpath : ie：/model/yabin/datasets/new
                which has subdir: Generated  Pristine
    """
    train = ['lsun_bedroom', 'lsun_bridge', 'lsun_churchoutdoor']
    test = ['lsun_kitchen', 'lsun_tower']
    false = 'Generated'
    true = 'Pristine'
    total_train = {}
    total_test = {}
    for i in train:
        for imgname in os.listdir(os.path.join(rootpath, false, 'GGAN256', i)):
            total_train[(os.path.join(false, 'GGAN256', i, imgname))] = 0
    for i in train:
        for imgname in os.listdir(os.path.join(rootpath, true, 'GGAN256', i)):
            total_train[os.path.join(true, 'GGAN256', i, imgname)] = 1
    for i in test:
        for imgname in os.listdir(os.path.join(rootpath, false, 'GGAN256', i)):
            total_test[(os.path.join(false, 'GGAN256', i, imgname))] = 0
    for i in test:
        for imgname in os.listdir(os.path.join(rootpath, true, 'GGAN256', i)):
            total_test[os.path.join(true, 'GGAN256', i, imgname)] = 1

    # split train to train and val
    total_train_list = list(total_train.keys())
    numtrain = len(total_train_list)
    random.shuffle(total_train_list)
    train_split = total_train_list[:int(numtrain * 0.8)]
    val_split = total_train_list[int(numtrain * 0.8):]

    train_dict = {}
    val_dict = {}
    for i in train_split:
        train_dict[i] = total_train[i]
    for i in val_split:
        val_dict[i] = total_train[i]
    return train_dict, val_dict, total_test

# train_dict, val_dict, total_test = gen_GGAN256_Train_Val_Test(rootpath)

def gen_GGAN1024_Train_Val_Test(rootpath = rootpath):
    """
    rootpath : ie：/model/yabin/datasets/new
                which has subdir: Generated  Pristine
    """
    train = ['celebhq']; test = ['HQ-IMG']; false = 'Generated'; true = 'Pristine'
    total_train = {}; total_test = {}
    for i in train:
        for imgname in os.listdir(os.path.join(rootpath, false, 'GGAN1024', i)):
            total_train[(os.path.join(false, 'GGAN1024', i, imgname))] = 0
    for i in test:
        for imgname in os.listdir(os.path.join(rootpath, true, 'GGAN1024', i)):
            total_train[os.path.join(true, 'GGAN1024', i, imgname)] = 1

    # split train to train, val, test
    total_train_list = list(total_train.keys())
    numtrain = len(total_train_list)
    random.shuffle(total_train_list)
    train_split = total_train_list[:int(numtrain * 0.6)]
    val_split = total_train_list[int(numtrain * 0.6): int(numtrain * 0.8)]
    test_split = total_train_list[int(numtrain * 0.8):]

    train_dict = {}
    val_dict = {}
    test_dict = {}
    for i in train_split:
        train_dict[i] = total_train[i]
    for i in val_split:
        val_dict[i] = total_train[i]
    for i in test_split:
        test_dict[i] = total_train[i]
    return train_dict, val_dict, test_dict

# train_dict, val_dict, total_test = gen_GGAN1024_Train_Val_Test(rootpath)


