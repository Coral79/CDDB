import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


# def get_dataset_dict(opt, name, id):
#     # only reture dict with labels and img pathes
#     dset_lst = {}
#     if opt.isTrain:
#         root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)
#         opt.classes = os.listdir(root_) if opt.multiclass[id] else ['']
#         for cls in opt.classes:
#             root = root_ + '/' + cls
#             dset = dataset_folder(opt, root)
#             dset_lst.append(dset)
#     else:
#         # root = opt.dataroot + '/' + name + '/{}/'.format(opt.val_split)
#         root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.val_split)
#         opt.classes = os.listdir(root_) if opt.multiclass[id] else ['']
#         for cls in opt.classes:
#             root = root_ + '/' + cls
#             dset = dataset_folder(opt, root)
#             dset_lst.append(dset)
#
#     return torch.utils.data.ConcatDataset(dset_lst)


class IncrementalDataset(Dataset):
    def __init__(self, opt, currentDataDic, previousDataDic=None, iteration=0, isTrain=False):
        """
        rootpath : ie：/model/yabin/datasets/new
                    which has subdir: Generated  Pristine
        currentDataDic : train_dict, val_dict, total_dict
        """
        self.rootpath = opt.dataroot
        self.currentDataDic = currentDataDic
        self.previousDataDic = previousDataDic
        imgs = []
        labels = []
        for i in currentDataDic.keys():
            imgs.append(i)
            labels.append(currentDataDic[i]+iteration*2)

        for i in previousDataDic.keys():
            imgs.append(i)
            labels.append(previousDataDic[i])

        self.imgs = imgs
        self.labels = labels
        opt.no_resize = False
        opt.no_crop = False
        opt.serial_batches = True
        opt.jpg_method = ['pil']

        if isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)
        if not isTrain and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

        self.transform = transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        # import pdb;pdb.set_trace()

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(os.path.join(self.rootpath, fn)).convert('RGB')
        img = self.transform(img)
        return img, self.labels[index], fn

    def __len__(self):
        return len(self.imgs)   # 返回图片的长度




""""
for getDataSplitFunc
"""

def getDataSplitFunc(opt, name, id):
    train_dict = {}
    val_dict = {}
    root_ = os.path.join(opt.dataroot, name, 'train')
    sub_classes = os.listdir(root_) if opt.multiclass[id] else ['']
    for cls in sub_classes:
        for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
            train_dict[os.path.join(root_, cls, '0_real', imgname)] = 0
        for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
            train_dict[os.path.join(root_, cls, '1_fake', imgname)] = 1

    root_ = os.path.join(opt.dataroot, name, 'val')
    sub_classes = os.listdir(root_) if opt.multiclass[id] else ['']
    for cls in sub_classes:
        for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
            val_dict[os.path.join(root_, cls, '0_real', imgname)] = 0
        for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
            val_dict[os.path.join(root_, cls, '1_fake', imgname)] = 1

    return train_dict, val_dict