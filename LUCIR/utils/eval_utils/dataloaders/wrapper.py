from os import path
import torch
import torch.utils.data as data

dic_to_binary = {
  "animal": ['beaver','dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
  'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard','lion', 'tiger', 'wolf','camel',
  'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster',
  'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake',
  'turtle','hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
  "non_animal": ['orchids','poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates',
   'apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers','clock', 'computer_keyboard', 'lamp', 'telephone', 'television',
   'bed', 'chair', 'couch', 'table', 'wardrobe','bridge', 'castle', 'house', 'road', 'skyscraper','cloud', 'forest',
   'mountain', 'plain', 'sea', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree', 'bicycle', 'bus', 'motorcycle', 'pickup_truck',
   'train','lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

class CacheClassLabel(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """

    def __init__(self, dataset):
        super(CacheClassLabel, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        label_cache_filename = path.join(dataset.root, str(
            type(dataset))+'_'+str(len(dataset))+'.pth')
        if path.exists(label_cache_filename):
            self.labels = torch.load(label_cache_filename)
        else:
            for i, data_ in enumerate(dataset):
                self.labels[i] = data_[1]
            torch.save(self.labels, label_cache_filename)
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target


class CacheClassLabel_binary(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """

    def __init__(self, dataset):
        super(CacheClassLabel_binary, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        label_cache_filename = path.join(dataset.root, str(
            type(dataset))+'_'+str(len(dataset))+'.pth')
        if path.exists(label_cache_filename):
            self.labels = torch.load(label_cache_filename)
        else:
            for i, data_ in enumerate(dataset):
                if data_[1] in dic_to_binary['animal']:
                    self.labels[i] = 'animal'
                else:
                    self.labels[i] = 'non_animal'
            torch.save(self.labels, label_cache_filename)
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.dataset.classes[target] in dic_to_binary['animal']:
            target = 0
        else:
            target = 1
        return img, target


class CacheClassLabel1(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """

    def __init__(self, dataset):
        super(CacheClassLabel1, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        for i, data_ in enumerate(dataset):
            self.labels[i] = data_[1]
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target

class CacheClassLabel_multi(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """

    def __init__(self, dataset, id):
        super(CacheClassLabel_multi, self).__init__()
        self.dataset = dataset
        self.id = id
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        for i, data_ in enumerate(dataset):
            self.labels[i] = data_[1] + id*2
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target_ = self.dataset[index]
        target = self.labels[index]
        return img, target


class AppendName(data.Dataset):
    """
    A dataset wrapper that also return the name of the dataset/task
    """

    def __init__(self, dataset, name, first_class_ind=0):
        super(AppendName, self).__init__()
        self.dataset = dataset
        self.name = name
        self.first_class_ind = first_class_ind  # For remapping the class index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target = target + self.first_class_ind
        return img, target, self.name


class Subclass(data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """

    def __init__(self, dataset, class_list, remap=True):
        '''
        :param dataset: (CacheClassLabel)
        :param class_list: (list) A list of integers
        :param remap: (bool) Ex: remap class [2,4,6 ...] to [0,1,2 ...]
        '''
        super(Subclass, self).__init__()
        assert isinstance(dataset, CacheClassLabel) or isinstance(dataset, CacheClassLabel1) or isinstance(dataset, CacheClassLabel_binary) or isinstance(dataset, CacheClassLabel_multi), 'dataset must be wrapped by CacheClassLabel'
        self.dataset = dataset
        self.class_list = class_list
        self.remap = remap
        self.indices = []
        for c in class_list:
            self.indices.extend(
                (dataset.labels == c).nonzero().flatten().tolist())
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, target = self.dataset[self.indices[index]]
        if self.remap:
            raw_target = target.item() if isinstance(target, torch.Tensor) else target
            target = self.class_mapping[raw_target]
        return img, target


class Permutation(data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """

    def __init__(self, dataset, permute_idx):
        super(Permutation, self).__init__()
        self.dataset = dataset
        self.permute_idx = permute_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        shape = img.size()
        img = img.view(-1)[self.permute_idx].view(shape)
        return img, target


class Storage(data.Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """

    def __init__(self):
        super(Storage, self).__init__()
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def append(self, x):
        self.storage.append(x)

    def extend(self, x):
        self.storage.extend(x)
