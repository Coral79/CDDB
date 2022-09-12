from __future__ import annotations

from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from torch import Tensor
from functools import partial
from typing import Protocol, Sequence, Union, Any, List, Iterable, Optional, TypeVar, Dict


class GetItemType(Protocol):
    def __getitem__(self, index: Any):
        ...


class DatasetThatSupportsTargets(GetItemType, Protocol):
    targets: Any


T_idx = TypeVar('T_idx')


def tensor_as_list(sequence):
    if isinstance(sequence, Tensor):
        return sequence.tolist()
    # Numpy already returns the correct format
    # Example list(np.array([1, 2, 3])) returns [1, 2, 3]
    # whereas list(torch.tensor([1, 2, 3])) returns [tensor(1), tensor(2), tensor(3)], which is "bad"
    return list(sequence)


def __get_indexes_with_patterns_ordered_by_classes(sequence: Sequence[T_idx], search_elements: Sequence[T_idx],
                                                   sort_indexes: bool = True, sort_classes: bool = True) -> Tensor:
    # list() handles the situation in which search_elements is a torch.Tensor
    # without it: result_per_class[element].append(idx) -> error
    # as result_per_class[0] won't exist while result_per_class[tensor(0)] will

    result_per_class: Dict[T_idx, List[int]] = OrderedDict()
    result: List[int] = []

    search_elements = tensor_as_list(search_elements)
    sequence = tensor_as_list(sequence)

    if sort_classes:
        search_elements = sorted(search_elements)

    for search_element in search_elements:
        result_per_class[search_element] = []

    for idx, element in enumerate(sequence):
        if element in search_elements:
            result_per_class[element].append(idx)

    for search_element in search_elements:
        if sort_indexes:
            result_per_class[search_element].sort()
        result.extend(result_per_class[search_element])

    return torch.tensor(result, dtype=torch.int)


def __get_indexes_without_class_bucketing(sequence: Sequence[T_idx], search_elements: Sequence[T_idx],
                                          sort_indexes: bool = False) -> Tensor:
    sequence = tensor_as_list(sequence)
    result: List[T_idx] = []

    for idx, element in enumerate(sequence):
        if element in search_elements:
            result.append(idx)

    if sort_indexes:
        result.sort()
    return torch.tensor(result, dtype=torch.int)


def get_indexes_from_set(sequence: Sequence[T_idx], search_elements: Sequence[T_idx], bucket_classes: bool = True,
                         sort_classes: bool = False, sort_indexes: bool = False) -> Tensor:
    if bucket_classes:
        return __get_indexes_with_patterns_ordered_by_classes(sequence, search_elements, sort_indexes=sort_indexes,
                                                              sort_classes=sort_classes)
    else:
        return __get_indexes_without_class_bucketing(sequence, search_elements, sort_indexes=sort_indexes)


class TransformationDataset(Dataset):
    """
    A Dataset that applies transformations before returning patterns/targets
    Also, this Dataset supports slicing
    """
    def __init__(self, dataset: Dataset, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

    def __getitem__(self, idx):
        patterns: List[Tensor] = []
        labels: List[Tensor] = []
        indexes_iterator: Iterable[int]

        # Makes dataset sliceable
        if isinstance(idx, slice):
            indexes_iterator = range(*idx.indices(len(self.dataset)))
        elif isinstance(idx, int):
            indexes_iterator = [idx]
        else:  # Should handle other types (ndarray, Tensor, Sequence, ...)
            if hasattr(idx, 'shape') and len(getattr(idx, 'shape')) == 0:  # Manages 0-d ndarray / Tensor
                indexes_iterator = [int(idx)]
            else:
                indexes_iterator = idx

        for single_idx in indexes_iterator:
            pattern, label = self.__get_single_item(single_idx)
            patterns.append(pattern.unsqueeze(0))
            labels.append(label.unsqueeze(0))

        if len(patterns) == 1:
            return patterns[0].squeeze(0), labels[0].squeeze(0)
        else:
            return torch.cat(patterns), torch.cat(labels)

    def __len__(self):
        return len(self.dataset)

    def __get_single_item(self, idx: int):
        pattern, label = self.dataset[idx]
        if self.transform is not None:
            pattern = self.transform(pattern)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return torch.as_tensor(pattern), torch.as_tensor(label, dtype=torch.long)


class TransformationSubset(Dataset):
    def __init__(self, dataset: Union[Dataset, DatasetThatSupportsTargets], indices: Sequence[int],
                 transform=None, target_transform=None):
        super().__init__()
        self.dataset = TransformationDataset(dataset, transform=transform, target_transform=target_transform)
        self.indices = indices

    def __getitem__(self, idx) -> (Tensor, Tensor):
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


def make_transformation_subset(dataset: DatasetThatSupportsTargets, transform: Any, target_transform: Any,
                               classes: Sequence[int], bucket_classes=True, sort_classes=False, sort_indexes=False):
    return TransformationSubset(dataset, get_indexes_from_set(dataset.targets, classes,
                                                              bucket_classes=bucket_classes,
                                                              sort_classes=sort_classes,
                                                              sort_indexes=sort_indexes),
                                transform=transform, target_transform=target_transform)


class NCProtocol:
    def __init__(self, train_dataset: DatasetThatSupportsTargets, test_dataset: DatasetThatSupportsTargets,
                 n_tasks: int, shuffle: bool = True, seed: Optional[int] = None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 steal_transforms_from_datasets=True, fixed_class_order: Optional[Sequence[int]] = None):
        self.train_dataset: DatasetThatSupportsTargets = train_dataset
        self.test_dataset: DatasetThatSupportsTargets = test_dataset
        self.validation_dataset: DatasetThatSupportsTargets = train_dataset
        self.n_tasks: int = n_tasks
        # self.classes_order: Tensor = torch.unique(torch.tensor(train_dataset.targets))
        self.classes_order: Tensor = torch.unique(torch.tensor(fixed_class_order))
        self.train_transform = train_transform
        self.train_target_transform = train_target_transform
        self.test_transform = test_transform
        self.test_target_transform = test_target_transform

        if len(self.classes_order) % n_tasks > 0:
            raise ValueError('Invalid number of tasks: classes contained in dataset cannot be divided by n_tasks')

        self.classes_per_task: int = len(self.classes_order) // n_tasks

        if fixed_class_order is not None:
            self.classes_order = torch.tensor(fixed_class_order)
        elif shuffle:
            if seed is not None:
                torch.random.manual_seed(seed)
            self.classes_order = self.classes_order[torch.randperm(len(self.classes_order))]

        if steal_transforms_from_datasets:
            if hasattr(train_dataset, 'transform'):
                self.train_transform = train_dataset.transform
                train_dataset.transform = None
            if hasattr(train_dataset, 'target_transform'):
                self.train_target_transform = train_dataset.target_transform
                train_dataset.target_transform = None

            if hasattr(test_dataset, 'transform'):
                self.test_transform = test_dataset.transform
                test_dataset.transform = None
            if hasattr(test_dataset, 'target_transform'):
                self.test_target_transform = test_dataset.target_transform
                test_dataset.target_transform = None

    def __iter__(self) -> NCProtocolIterator:
        return NCProtocolIterator(self)


class NCProtocolIterator:
    def __init__(self, protocol: NCProtocol,
                 swap_train_test_transformations: bool = False,
                 initial_current_task: int = -1):
        self.current_task: int = -1
        self.protocol: NCProtocol = protocol
        self.are_train_test_transformations_swapped = swap_train_test_transformations

        self.classes_seen_so_far: Tensor = torch.empty(0, dtype=torch.long)
        self.classes_in_this_task: Tensor = torch.empty(0, dtype=torch.long)
        self.prev_classes: Tensor = torch.empty(0, dtype=torch.long)

        if self.are_train_test_transformations_swapped:
            self.train_subset_factory = partial(make_transformation_subset, self.protocol.train_dataset,
                                                self.protocol.test_transform, self.protocol.test_target_transform)
            self.test_subset_factory = partial(make_transformation_subset, self.protocol.test_dataset,
                                               self.protocol.train_transform, self.protocol.train_target_transform)
        else:
            self.train_subset_factory = partial(make_transformation_subset, self.protocol.train_dataset,
                                                self.protocol.train_transform, self.protocol.train_target_transform)
            self.test_subset_factory = partial(make_transformation_subset, self.protocol.test_dataset,
                                               self.protocol.test_transform, self.protocol.test_target_transform)

        for _ in range(initial_current_task+1):
            self.__go_to_next_task()

    def __next__(self) -> (Dataset, 'NCProtocolIterator'):
        self.__go_to_next_task()
        return self.get_current_training_set(), self

    # Training set utils
    def get_current_training_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) -> Dataset:
        return self.train_subset_factory(self.classes_in_this_task, bucket_classes=bucket_classes,
                                         sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_task_training_set(self, task_id: int, bucket_classes=True, sort_classes=False,
                              sort_indexes=False) -> Dataset:
        classes_start_idx = self.protocol.classes_per_task * task_id
        classes_end_idx = classes_start_idx + self.protocol.classes_per_task

        classes_in_required_task = self.protocol.classes_order[classes_start_idx:classes_end_idx]
        return self.train_subset_factory(classes_in_required_task, bucket_classes=bucket_classes,
                                         sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_cumulative_training_set(self, include_current_task: bool = True, bucket_classes=True, sort_classes=False,
                                    sort_indexes=False) -> Dataset:
        if include_current_task:
            return self.train_subset_factory(self.classes_seen_so_far, bucket_classes=bucket_classes,
                                             sort_classes=sort_classes, sort_indexes=sort_indexes)
        else:
            return self.train_subset_factory(self.prev_classes, bucket_classes=bucket_classes,
                                             sort_classes=sort_classes, sort_indexes=sort_indexes)

    # Test set utils
    def get_current_test_set(self, bucket_classes=True, sort_classes=False, sort_indexes=False) -> Dataset:
        return self.test_subset_factory(self.classes_in_this_task, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_cumulative_test_set(self, include_current_task: bool = True, bucket_classes=True, sort_classes=False,
                                sort_indexes=False) -> Dataset:
        if include_current_task:
            return self.test_subset_factory(self.classes_seen_so_far, bucket_classes=bucket_classes,
                                            sort_classes=sort_classes, sort_indexes=sort_indexes)
        else:
            return self.test_subset_factory(self.prev_classes, bucket_classes=bucket_classes,
                                            sort_classes=sort_classes, sort_indexes=sort_indexes)

    def get_task_test_set(self, task_id: int, bucket_classes=True, sort_classes=False, sort_indexes=False) -> Dataset:
        classes_start_idx = self.protocol.classes_per_task * task_id
        classes_end_idx = classes_start_idx + self.protocol.classes_per_task

        classes_in_required_task = self.protocol.classes_order[classes_start_idx:classes_end_idx]
        return self.test_subset_factory(classes_in_required_task, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes, sort_indexes=sort_indexes)

    def swap_transformations(self) -> NCProtocolIterator:
        return NCProtocolIterator(self.protocol,
                                  swap_train_test_transformations=not self.are_train_test_transformations_swapped,
                                  initial_current_task=self.current_task)

    def __go_to_next_task(self):
        if self.current_task == (self.protocol.n_tasks - 1):
            raise StopIteration()

        self.current_task += 1
        classes_start_idx = self.protocol.classes_per_task * self.current_task
        classes_end_idx = classes_start_idx + self.protocol.classes_per_task

        self.classes_in_this_task = self.protocol.classes_order[classes_start_idx:classes_end_idx]
        self.prev_classes = self.classes_seen_so_far
        self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, self.classes_in_this_task])
