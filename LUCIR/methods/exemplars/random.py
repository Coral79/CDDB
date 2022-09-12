""" Class-incremental learning base trainer. """
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_features import compute_features_origin_lucia
from utils.incremental.compute_accuracy import compute_accuracy_origin_lucia
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

import warnings
try:
    import cPickle as pickle
except:
    import pickle
warnings.filterwarnings('ignore')

import wandb

# Herding algorithm
def herding_examplers(args, tg_model, iteration, evalset, last_iter, order, alpha_dr_herding, prototypes, dictionary_size):
    """The function to select the exemplars
    Args:
      b1_model: the 1st branch model from the current phase
      is_start_iteration: a bool variable, which indicates whether the current phase is the 0th phase
      iteration: the iteration index
      last_iter: the iteration index for last phase
      order: the array for the class order
      alpha_dr_herding: the empty array to store the indexes for the exemplars
      prototypes: the array contains all training samples for all phases
    Returns:
      X_protoset_cumuls: an array that contains old exemplar samples
      Y_protoset_cumuls: an array that contains old exemplar labels
      class_means: the mean values for each class
      alpha_dr_herding: the empty array to store the indexes for the exemplars, updated
    """
    # Use the dictionary size defined in this class-incremental learning class
    dictionary_size = dictionary_size
    if args.dynamic_budget:
        # Using dynamic exemplar budget, i.e., 20 exemplars each class. In this setting, the total memory budget is increasing
        nb_protos_cl = args.nb_protos
    else:
        # Using fixed exemplar budget. The total memory size is unchanged
        nb_protos_cl = int(np.ceil(args.nb_protos * 100. / args.nb_cl / (iteration + 1)))
    # Get tg_feature_model, which is a model copied from b1_model, without the FC layer
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    # Get the shape for the feature maps
    num_features = tg_model.fc.in_features
    for iter_dico in range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl):
        # Set a temporary dataloader for the current class
        evalset.data = prototypes[iter_dico].astype('uint8')
        evalset.targets = np.zeros(evalset.data.shape[0])
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size,
                                                 shuffle=False, num_workers=args.num_workers)
        num_samples = evalset.data.shape[0]
        # Compute the features for the current class
        mapped_prototypes = compute_features_origin_lucia(tg_feature_model, evalloader, num_samples, num_features)
        # Herding algorithm
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)
        mu = np.mean(D, axis=1)

        index1 = int(iter_dico / args.nb_cl)
        index2 = iter_dico % args.nb_cl
        ############################################
        # modelindex = int(10*index1 + index2)
        # mu = tg_model.rpcenter[modelindex].detach().cpu().numpy()
        # mu = (tg_model.rpcenter[modelindex].detach().cpu().numpy() / np.linalg.norm(tg_model.rpcenter[modelindex].detach().cpu().numpy(), axis=0))

        alpha_dr_herding[index1, :, index2] = alpha_dr_herding[index1, :, index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(alpha_dr_herding[index1, :, index2] != 0) ==
                   min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[index1, ind_max, index2] == 0:
                alpha_dr_herding[index1, ind_max, index2] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]


    # Set two empty lists for the exemplars and the labels
    X_protoset_cumuls = []
    Y_protoset_cumuls = []

    class_means = np.zeros((num_features, args.num_classes, 2))
    for iteration2 in range(iteration + 1):
        for iter_dico in range(args.nb_cl):
            # Compute the D and D2 matrizes, which are used to compute the class mean values
            current_cl = order[range(iteration2 * args.nb_cl, (iteration2 + 1) * args.nb_cl)]
            current_eval_set = merge_images_labels(prototypes[iteration2 * args.nb_cl + iter_dico], \
                                                   np.zeros(len(
                                                       prototypes[iteration2 * args.nb_cl + iter_dico])))
            evalset.imgs = evalset.samples = current_eval_set
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size,
                                                     shuffle=False, num_workers=args.num_workers,
                                                     pin_memory=True)
            num_samples = len(prototypes[iteration2 * args.nb_cl + iter_dico])
            mapped_prototypes = compute_features_origin_lucia(tg_feature_model, evalloader, num_samples,
                                                              num_features)
            D = mapped_prototypes.T
            D = D / np.linalg.norm(D, axis=0)
            D2 = D
            # Using the indexes selected by herding
            alph = alpha_dr_herding[iteration2, :, iter_dico]
            assert ((alph[num_samples:] == 0).all())
            alph = alph[:num_samples]
            alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.
            # Add the exemplars and the labels to the lists
            X_protoset_cumuls.append(
                prototypes[iteration2 * args.nb_cl + iter_dico][np.where(alph == 1)[0]])
            Y_protoset_cumuls.append(
                order[iteration2 * args.nb_cl + iter_dico] * np.ones(len(np.where(alph == 1)[0])))
            # Compute the class mean values
            alph = alph / np.sum(alph)
            class_means[:, current_cl[iter_dico], 0] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
            class_means[:, current_cl[iter_dico], 0] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])
            alph = np.ones(num_samples) / num_samples
            class_means[:, current_cl[iter_dico], 1] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
            class_means[:, current_cl[iter_dico], 1] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 1])
    else:
        raise ValueError('Please set correct dataset.')


    return X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding
