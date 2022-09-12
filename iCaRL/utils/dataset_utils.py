from typing import SupportsInt

import torch
from torch.utils.data import Dataset
from torch import Tensor


def get_dataset_per_pixel_mean(dataset: Dataset) -> Tensor:
    result = None
    patterns_count = 0

    for img_pattern, _ in dataset:
        if result is None:
            # Only on first iteration
            result = torch.zeros_like(img_pattern, dtype=torch.float)

        result += img_pattern
        patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result


def make_single_pattern_one_hot(input_label: SupportsInt, n_classes: int, dtype=torch.float) -> Tensor:
    target = torch.zeros(n_classes, dtype=dtype)
    target[int(input_label)] = 1
    return target


def make_batch_one_hot(input_tensor: Tensor, n_classes: int, dtype=torch.float) -> Tensor:
    targets = torch.zeros(input_tensor.shape[0], n_classes, dtype=dtype)
    targets[range(len(input_tensor)), input_tensor.long()] = 1
    return targets
