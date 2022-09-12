import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Sequence, List, Dict, SupportsFloat
from utils import make_batch_one_hot
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

class CumulativeStatistic:
    def __init__(self):
        self.values_sum = 0.0
        self.overall_count = 0.0
        self.average = 0.0

    def update_using_counts(self, value: SupportsFloat, count: SupportsFloat = 1.0):
        value = float(value)
        count = float(count)
        self.values_sum += value
        self.overall_count += count
        self.average = self.values_sum / self.overall_count

    def update_using_averages(self, value: SupportsFloat, count: SupportsFloat = 1.0):
        value = float(value) * float(count)
        count = float(count)
        self.values_sum += value
        self.overall_count += count
        self.average = self.values_sum / self.overall_count


def get_accuracy(model: Module, test_dataset: Dataset, device: Optional[torch.device] = None,
                 required_top_k: Optional[Sequence[int]] = None, return_detailed_outputs: bool = False,
                 criterion: Module = None, make_one_hot: bool = False, n_classes: int = -1, **kwargs) -> \
        (torch.FloatTensor, Optional[float], Optional[Tensor], Optional[Tensor]):
    stats: Dict[int, CumulativeStatistic]
    max_top_k: int
    all_val_outputs_tmp: List[Tensor] = []
    all_val_labels_tmp: List[Tensor] = []
    all_val_outputs: Optional[Tensor] = None
    all_val_labels: Optional[Tensor] = None
    test_loss: Optional[CumulativeStatistic] = None
    test_loader: DataLoader

    if required_top_k is None:
        required_top_k = [1]

    max_top_k = max(required_top_k)

    stats = {}
    for top_k in required_top_k:
        stats[top_k] = CumulativeStatistic()

    if criterion is not None:
        # Enable test loss
        test_loss = CumulativeStatistic()

    if make_one_hot and n_classes <= 0:
        raise ValueError("n_class must be set when using one_hot_vectors")

    test_loader = DataLoader(test_dataset, **kwargs)

    model.eval()
    with torch.no_grad():
        patterns: Tensor
        labels: Tensor
        targets: Tensor
        output: Tensor
        for patterns, labels in test_loader:
            # Clear grad
            model.zero_grad()

            if return_detailed_outputs:
                all_val_labels_tmp.append(labels.detach().cpu())

            if make_one_hot:
                targets = make_batch_one_hot(labels, n_classes)
            else:
                targets = labels

            # Send data to device
            if device is not None:
                patterns = patterns.to(device)
                targets = targets.to(device)

            # Forward
            output = model(patterns)

            if criterion is not None:
                test_loss.update_using_averages(criterion(output, targets).detach().cpu().item(), count=len(labels))

            output = output.detach().cpu()
            if return_detailed_outputs:
                all_val_outputs_tmp.append(output)

            # https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
            # Gets the indexes of max_top_k elements
            _, top_k_idx = output.topk(max_top_k, 1)
            top_k_idx = top_k_idx.t()

            # correct will have values True where index == label
            correct = top_k_idx.eq(labels.reshape(1, -1).expand_as(top_k_idx))
            for top_k in required_top_k:
                correct_k = correct[:top_k].reshape(-1).float().sum(0)  # Number of correct patterns for this top_k
                stats[top_k].update_using_counts(correct_k, len(labels))

        if return_detailed_outputs:
            all_val_outputs = torch.cat(all_val_outputs_tmp)
            all_val_labels = torch.cat(all_val_labels_tmp)

    acc_results = torch.empty(len(required_top_k), dtype=torch.float)

    for top_idx, top_k in enumerate(required_top_k):
        acc_results[top_idx] = stats[top_k].average

    test_loss_result = None
    if criterion is not None:
        test_loss_result = test_loss.average

    return acc_results, test_loss_result, all_val_outputs, all_val_labels

def get_accuracy_binary(model: Module, test_dataset: Dataset, device: Optional[torch.device] = None,
                 required_top_k: Optional[Sequence[int]] = None, return_detailed_outputs: bool = False,
                 criterion: Module = None, make_one_hot: bool = False, n_classes: int = -1, **kwargs) -> \
        (torch.FloatTensor, Optional[float], Optional[Tensor], Optional[Tensor]):
    stats: Dict[int, CumulativeStatistic]
    max_top_k: int
    all_val_outputs_tmp: List[Tensor] = []
    all_val_labels_tmp: List[Tensor] = []
    all_val_outputs: Optional[Tensor] = None
    all_val_labels: Optional[Tensor] = None
    test_loss: Optional[CumulativeStatistic] = None
    test_loader: DataLoader

    if required_top_k is None:
        required_top_k = [1]

    max_top_k = max(required_top_k)

    stats = {}
    for top_k in required_top_k:
        stats[top_k] = CumulativeStatistic()

    if criterion is not None:
        # Enable test loss
        test_loss = CumulativeStatistic()

    if make_one_hot and n_classes <= 0:
        raise ValueError("n_class must be set when using one_hot_vectors")

    test_loader = DataLoader(test_dataset, **kwargs)

    model.eval()
    with torch.no_grad():
        patterns: Tensor
        labels: Tensor
        targets: Tensor
        output: Tensor
        for patterns, labels in test_loader:
            # Clear grad
            model.zero_grad()
            labels = labels%2

            if return_detailed_outputs:
                all_val_labels_tmp.append(labels.detach().cpu())

            if make_one_hot:
                targets = make_batch_one_hot(labels, n_classes)
            else:
                targets = labels

            # Send data to device
            if device is not None:
                patterns = patterns.to(device)
                targets = targets.to(device)

            # Forward
            output = model(patterns)

            if criterion is not None:
                test_loss.update_using_averages(criterion(output.squeeze(1), targets.float()).detach().cpu().item(), count=len(labels))

            output = output.detach().cpu()
            if return_detailed_outputs:
                all_val_outputs_tmp.append(output)

            # https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
            # Gets the indexes of max_top_k elements
            y_pred = output.flatten().tolist()
            y_true = targets.flatten().tolist()
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc = accuracy_score(y_true, y_pred > 0.5)
            # _, top_k_idx = output.topk(max_top_k, 1)
            # top_k_idx = top_k_idx.t()

            # # correct will have values True where index == label
            # correct = top_k_idx.eq(labels.reshape(1, -1).expand_as(top_k_idx))
            for top_k in required_top_k:
                stats[top_k]= acc
                # correct_k = correct[:top_k].reshape(-1).float().sum(0)  # Number of correct patterns for this top_k
                # stats[top_k].update_using_counts(correct_k, len(labels))

        if return_detailed_outputs:
            all_val_outputs = torch.cat(all_val_outputs_tmp)
            all_val_labels = torch.cat(all_val_labels_tmp)

    acc_results = torch.empty(len(required_top_k), dtype=torch.float)

    for top_idx, top_k in enumerate(required_top_k):
        acc_results[top_idx] = np.average(stats[top_k])
    test_loss_result = None
    if criterion is not None:
        test_loss_result = test_loss.average

    return acc_results, test_loss_result, all_val_outputs, all_val_labels
