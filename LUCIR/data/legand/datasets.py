from PIL import Image
from torch.utils.data import Dataset
import os


class IncrementalDataset(Dataset):
    def __init__(self, rootpath, currentDataDic, previousDataDic=None, iteration=0, transform=None):
        """
        rootpath : ie：/model/yabin/datasets/new
                    which has subdir: Generated  Pristine
        currentDataDic : train_dict, val_dict, total_dict
        """
        self.rootpath = rootpath
        self.currentDataDic = currentDataDic
        self.previousDataDic = previousDataDic
        imgs = []
        labels = []
        for i in currentDataDic.keys():
            imgs.append(i)
            labels.append(currentDataDic[i]+iteration*2)
        if previousDataDic is not None:
            for i in previousDataDic.keys():
                imgs.append(i)
                labels.append(previousDataDic[i])
        self.imgs = imgs
        self.labels = labels

        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(os.path.join(self.rootpath, fn)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index], fn

    def __len__(self):
        return len(self.imgs)   # 返回图片的长度


