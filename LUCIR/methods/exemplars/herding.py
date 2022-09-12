""" Class-incremental learning base trainer. """
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import warnings
try:
    import cPickle as pickle
except:
    import pickle
warnings.filterwarnings('ignore')
from PIL import Image
from torchvision import transforms
import wandb

# Herding algorithm
def herding_examplers(args, tg_model, iteration, taskimgs:dict, examplers_dict):
    if examplers_dict is None:
        examplers_dict = {}
    examplers_dict_copy = {}

    # set for dynamic budget
    dynamic_memory = False
    if dynamic_memory:
        exampler_size = args.exampler_size
    else:
        exampler_size = args.memory_size/((iteration+1)*2)
    print('exampler_size: '+ str(exampler_size))


    # exampler_size = args.exampler_size
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    num_features = tg_model.fc.in_features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()

    trans = transforms.Compose([transforms.Resize(256), \
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # herding for current images
    for task in taskimgs.keys():
        imgs = taskimgs[task]
        num_samples = len(imgs)
        mapped_prototypes = np.zeros([num_samples, num_features])
        start_idx = 0
        for img in imgs:
            img = trans(Image.open(img).convert('RGB'))
            inputs = img.to(device)
            the_feature = tg_feature_model(inputs.unsqueeze(0)).squeeze().detach()
            mapped_prototypes[start_idx, :] = np.squeeze(the_feature.cpu())
            start_idx += 1

        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)
        mu = np.mean(D, axis=1)

        tmp_herding = np.array([0]*num_samples)
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(tmp_herding != 0) == min(exampler_size, 10000)) and iter_herding_eff < 5000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if tmp_herding[ind_max] == 0:
                tmp_herding[ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]
        alph = (tmp_herding > 0) * (tmp_herding < exampler_size + 1) * 1.
        # import pdb;pdb.set_trace()
        selectidx = np.where(alph == 1)[0]

        for ii in selectidx:
            examplers_dict_copy[imgs[ii]] = task+iteration*2


    # herding for saved images
    if not dynamic_memory and iteration>0:
        examplers_dict_reversed={i: [] for i in range((iteration)*2)}
        for key, value in examplers_dict.items():
            examplers_dict_reversed[value].append(key)
        for task in examplers_dict_reversed.keys():
            imgs = examplers_dict_reversed[task]
            num_samples = len(imgs)
            mapped_prototypes = np.zeros([num_samples, num_features])
            start_idx = 0
            for img in imgs:
                img = trans(Image.open(img).convert('RGB'))
                inputs = img.to(device)
                the_feature = tg_feature_model(inputs.unsqueeze(0)).squeeze().detach()
                mapped_prototypes[start_idx, :] = np.squeeze(the_feature.cpu())
                start_idx += 1

            D = mapped_prototypes.T
            D = D / np.linalg.norm(D, axis=0)
            mu = np.mean(D, axis=1)

            tmp_herding = np.array([0] * num_samples)
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0
            while not (np.sum(tmp_herding != 0) == min(exampler_size, 2000)) and iter_herding_eff < 1000:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if tmp_herding[ind_max] == 0:
                    tmp_herding[ind_max] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]
            alph = (tmp_herding > 0) * (tmp_herding < exampler_size + 1) * 1.
            # import pdb;pdb.set_trace()
            selectidx = np.where(alph == 1)[0]

            for ii in selectidx:
                examplers_dict_copy[imgs[ii]] = task


    # examplers_dict
    examplers_dict_reversed={i: [] for i in range((iteration+1)*2)}
    for key, value in examplers_dict_copy.items():
        examplers_dict_reversed[value].append(key)

    class_means = np.zeros((num_features, (iteration+1)*2, 2))
    for iteration in range((iteration+1)*2):
        imgs = examplers_dict_reversed[iteration]
        num_samples = len(imgs)
        mapped_prototypes = np.zeros([num_samples, num_features])
        start_idx = 0
        for img in imgs:
            img = trans(Image.open(img).convert('RGB'))
            inputs = img.to(device)
            the_feature = tg_feature_model(inputs.unsqueeze(0)).squeeze().detach()
            mapped_prototypes[start_idx, :] = np.squeeze(the_feature.cpu())
            start_idx += 1
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)
        mapped_prototypes2 = mapped_prototypes[::-1,:]
        D2 = mapped_prototypes2.T
        D2 = D2 / np.linalg.norm(D2, axis=0)

        alph = np.array([1 / np.sum(num_samples)]*num_samples)
        class_means[:, iteration, 0] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
        class_means[:, iteration, 0] /= np.linalg.norm(class_means[:, iteration, 0])
        alph = np.ones(num_samples) / num_samples
        class_means[:, iteration, 1] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
        class_means[:, iteration, 1] /= np.linalg.norm(class_means[:, iteration, 1])

    return examplers_dict_copy, class_means

