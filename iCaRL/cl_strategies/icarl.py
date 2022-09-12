from typing import Optional, List, Callable, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from utils import make_batch_one_hot
import numpy as np


def icarl_accuracy_measure(test_dataset: Dataset, class_means: Tensor,
                           val_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]], top1_acc_list: Tensor,
                           iteration: int, iteration_total: int, type_data: str,
                           make_one_hot: bool = False, n_classes: int = -1, device: Optional[torch.device] = None,
                           **kwargs) -> (float, Optional[float]):
    test_loader: DataLoader = DataLoader(test_dataset, **kwargs)

    stat_hb1: List[bool] = []
    stat_icarl: List[bool] = []
    stat_ncm: List[bool] = []

    if make_one_hot and n_classes <= 0:
        raise ValueError("n_class must be set when using one_hot_vectors")

    with torch.no_grad():
        patterns: Tensor
        labels: Tensor
        targets: Tensor
        output: Tensor

        for patterns, labels in test_loader:
            if make_one_hot:
                targets = make_batch_one_hot(labels, n_classes)
            else:
                targets = labels

            # Send data to device
            if device is not None:
                patterns = patterns.to(device)
                targets = targets.to(device)

            _, pred, pred_inter = val_fn(patterns, targets)
            pred = pred.detach().cpu()

            pred_inter = (pred_inter.T / torch.norm(pred_inter.T, dim=0)).T

            # Lines 191-195: Compute score for iCaRL
            sqd = torch.cdist(class_means[:, :, 0].T, pred_inter)
            score_icarl = (-sqd).T
            # Compute score for NCM
            sqd = torch.cdist(class_means[:, :, 1].T, pred_inter)
            score_ncm = (-sqd).T

            # Compute the accuracy over the batch
            stat_hb1 += (
            [ll in best for ll, best in zip(labels, torch.argsort(pred, dim=1)[:, -1:])])
            stat_icarl += (
            [ll in best for ll, best in zip(labels, torch.argsort(score_icarl, dim=1)[:, -1:])])
            stat_ncm += (
            [ll in best for ll, best in zip(labels, torch.argsort(score_ncm, dim=1)[:, -1:])])

        # https://stackoverflow.com/a/20840816
        stat_hb1_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_hb1])
        stat_icarl_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_icarl])
        stat_ncm_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_ncm])

        print("Final results on " + type_data + " classes:")
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(torch.mean(stat_icarl_numerical) * 100))
        print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(torch.mean(stat_hb1_numerical) * 100))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(torch.mean(stat_ncm_numerical) * 100))

        top1_acc_list[iteration, 0, iteration_total] = torch.mean(stat_icarl_numerical) * 100
        top1_acc_list[iteration, 1, iteration_total] = torch.mean(stat_hb1_numerical) * 100
        top1_acc_list[iteration, 2, iteration_total] = torch.mean(stat_ncm_numerical) * 100

        return top1_acc_list


def icarl_accuracy_measure_binary(test_dataset: Dataset, class_means: Tensor,
                           val_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]], top1_acc_list: Tensor,
                           iteration: int, iteration_total: int, type_data: str,
                           make_one_hot: bool = False, n_classes: int = 2, device: Optional[torch.device] = None,
                           **kwargs) -> (float, Optional[float]):
    test_loader: DataLoader = DataLoader(test_dataset, **kwargs)

    stat_hb1: List[bool] = []
    stat_icarl: List[bool] = []
    stat_ncm: List[bool] = []

    if make_one_hot and n_classes <= 0:
        raise ValueError("n_class must be set when using one_hot_vectors")

    with torch.no_grad():
        patterns: Tensor
        labels: Tensor
        targets: Tensor
        output: Tensor

        for patterns, labels in test_loader:
            labels = labels%2
            if make_one_hot:
                targets = make_batch_one_hot(labels, n_classes)
            else:
                # labels_binary = labels%2
                # targets = make_batch_one_hot(labels, n_classes)
                targets = labels

            # Send data to device
            if device is not None:
                patterns = patterns.to(device)
                targets = targets.to(device)

            _, pred, pred_inter = val_fn(patterns, targets)
            pred = pred.detach().cpu()
            pred = (pred>0.5)
            pred_inter = (pred_inter.T / torch.norm(pred_inter.T, dim=0)).T

            # Lines 191-195: Compute score for iCaRL
            sqd = torch.cdist(class_means[:, :, 0].T, pred_inter)
            score_icarl = (-sqd).T
            # Compute score for NCM
            sqd = torch.cdist(class_means[:, :, 1].T, pred_inter)
            score_ncm = (-sqd).T

            # Compute the accuracy over the batch
            stat_hb1 += (
            [ll in best for ll, best in zip(labels, pred)])
            stat_icarl += (
            [ll in (best%2) for ll, best in zip(labels, torch.argsort(score_icarl, dim=1)[:, -1:])])
            stat_ncm += (
            [ll in (best%2) for ll, best in zip(labels, torch.argsort(score_ncm, dim=1)[:, -1:])])

        # https://stackoverflow.com/a/20840816
        stat_hb1_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_hb1])
        stat_icarl_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_icarl])
        stat_ncm_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_ncm])

        print("Final results on " + type_data + " classes:")
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(torch.mean(stat_icarl_numerical) * 100))
        print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(torch.mean(stat_hb1_numerical) * 100))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(torch.mean(stat_ncm_numerical) * 100))

        top1_acc_list[iteration, 0, iteration_total] = torch.mean(stat_icarl_numerical) * 100
        top1_acc_list[iteration, 1, iteration_total] = torch.mean(stat_hb1_numerical) * 100
        top1_acc_list[iteration, 2, iteration_total] = torch.mean(stat_ncm_numerical) * 100

        return top1_acc_list

def icarl_accuracy_measure_to_binary(test_dataset: Dataset, class_means: Tensor,
                           val_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]], top1_acc_list: Tensor,
                           iteration: int, iteration_total: int, type_data: str,
                           make_one_hot: bool = False, n_classes: int = 2, device: Optional[torch.device] = None,
                           **kwargs) -> (float, Optional[float]):
    test_loader: DataLoader = DataLoader(test_dataset, **kwargs)

    stat_hb1: List[bool] = []
    stat_icarl: List[bool] = []
    stat_ncm: List[bool] = []

    if make_one_hot and n_classes <= 0:
        raise ValueError("n_class must be set when using one_hot_vectors")

    with torch.no_grad():
        patterns: Tensor
        labels: Tensor
        targets: Tensor
        output: Tensor

        for patterns, labels in test_loader:
            if make_one_hot:
                targets = make_batch_one_hot(labels, n_classes)
            else:
                labels_binary = labels%2
                # targets = labels_binary
                targets = make_batch_one_hot(labels, n_classes)
                # targets = labels

            # Send data to device
            if device is not None:
                patterns = patterns.to(device)
                targets = targets.to(device)

            _, pred, pred_inter = val_fn(patterns, targets)
            pred = pred.detach().cpu()

            pred_inter = (pred_inter.T / torch.norm(pred_inter.T, dim=0)).T

            # Lines 191-195: Compute score for iCaRL
            sqd = torch.cdist(class_means[:, :, 0].T, pred_inter)
            score_icarl = (-sqd).T
            # Compute score for NCM
            sqd = torch.cdist(class_means[:, :, 1].T, pred_inter)
            score_ncm = (-sqd).T

            # Compute the accuracy over the batch
            stat_hb1 += (
            [ll in (best%2) for ll, best in zip(labels_binary, torch.argsort(pred, dim=1)[:, -1:])])
            stat_icarl += (
            [ll in (best%2) for ll, best in zip(labels_binary, torch.argsort(score_icarl, dim=1)[:, -1:])])
            stat_ncm += (
            [ll in (best%2) for ll, best in zip(labels_binary, torch.argsort(score_ncm, dim=1)[:, -1:])])

        # https://stackoverflow.com/a/20840816
        stat_hb1_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_hb1])
        stat_icarl_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_icarl])
        stat_ncm_numerical = torch.as_tensor([float(int(val_t)) for val_t in stat_ncm])

        print("Final results on " + type_data + " classes:")
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(torch.mean(stat_icarl_numerical) * 100))
        print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(torch.mean(stat_hb1_numerical) * 100))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(torch.mean(stat_ncm_numerical) * 100))

        top1_acc_list[iteration, 0, iteration_total] = torch.mean(stat_icarl_numerical) * 100
        top1_acc_list[iteration, 1, iteration_total] = torch.mean(stat_hb1_numerical) * 100
        top1_acc_list[iteration, 2, iteration_total] = torch.mean(stat_ncm_numerical) * 100

        return top1_acc_list


def icarl_cifar100_augment_data(img: Tensor) -> Tensor:
    # as in paper :
    # pad feature arrays with 4 pixels on each side
    # and do random cropping of 32x32
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode='constant')
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.random_integers(0, high=8, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[:, crop[0]:(crop[0] + 32), crop[1]:(crop[1] + 32)]
    else:
        random_cropped[:, :, :] = padded[:, crop[0]:(crop[0] + 32), crop[1]:(crop[1] + 32)][:, :, ::-1]

    return torch.tensor(random_cropped)
