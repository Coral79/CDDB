import os
import argparse
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import sklearn

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--resume', type=str, default='', help='resume model')
    parser.add_argument('--dataroot', type=str, default='/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/', help='data path')
    parser.add_argument('--datatype', type=str, default='deepfake', help='data type')
    return parser

class DummyDataset(Dataset):
    def __init__(self, data_path, data_type):

        self.trsf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = []
        labels = []
        if data_type == "deepfake":
            # subsets = ["gaugan","biggan","cyclegan","imle","deepfake","crn","wild"]
            # multiclass = [0, 0, 1, 0, 0, 0, 0]
            subsets = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]
            multiclass = [0, 0, 0, 0, 0]
            # subsets = ["gaugan","biggan","cyclegan","imle","deepfake","crn","wild","glow","stargan_gf","stylegan","whichfaceisreal","san"]
            # multiclass = [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]


            for id, name in enumerate(subsets):
                root_ = os.path.join(data_path, name, 'val')
                # sub_classes = ['']
                sub_classes = os.listdir(root_) if multiclass[id] else ['']
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                        images.append(os.path.join(root_, cls, '0_real', imgname))
                        labels.append(0 + 2 * id)

                    for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                        images.append(os.path.join(root_, cls, '1_fake', imgname))
                        labels.append(1 + 2 * id)
        else:
            pass

        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(self.pil_loader(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

args = setup_parser().parse_args()


from models.classifier import Classifier
from models.convit_timm import convit_small, convit_base

# pretrained_model = torch.load("/home/wangyabin/workspace/dytox/checkpoints/22-04-26_dytox_ptconvit_1/checkpoint_4.pth")['model']
# dytox hard 100 sum logit,sum feature, max
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_hard_m100_mc_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_hard_m100_sumasig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_hard_m100_sumbsig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_hard_m100_max0.3_1/checkpoint_4.pth"
# sytox hard 1000/500 sumfeature/sumlogit/mc
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_hard_m1000_sumasig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_hard_m1000_sumbsig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_hard_m1000_mc_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_hard_m500_sumasig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_hard_m500_sumbsig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-12_dytox_ganfake_hard_m500_mc_1/checkpoint_4.pth"
modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-30_dytox_ganfake_hard_m0_mc_1/checkpoint_4.pth"

# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_hard_m1500_mc_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_hard_m1500_sumbsig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_hard_m1500_sumasig0.3_1/checkpoint_4.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-13_dytox_ganfake_joint_training_hard_1/checkpoint_0.pth"

# sytox easy 4种loss
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_easy_m1500_mc_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_easy_m1500_sumasig0.3_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-07_dytox_ganfake_easy_m1500_sumbsig0.3_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-12_dytox_ganfake_easy_m1500_sumblog0.001_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_easy_m1500_max0.3_1/checkpoint_6.pth"

# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_easy_m100_mc_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_easy_m100_sumasig0.3_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_easy_m100_sumbsig0.3_1/checkpoint_6.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-05_dytox_ganfake_easy_m100_max0.3_1/checkpoint_6.pth"

# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-12_dytox_ganfake_joint_training_easy_1/checkpoint_0.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-09_dytox_ganfake_zero-shot_1/checkpoint_0.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-11_dytox_ganfake_finetune_1/checkpoint_6.pth"


# long
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_long_m1500_mc_1/checkpoint_11.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_long_m1500_sumasig0.3_1/checkpoint_11.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-08_dytox_ganfake_long_m1500_sumbsig0.3_1/checkpoint_11.pth"
# modelpath = "/home/wangyabin/workspace/TransformerCIL/checkpoints/22-07-13_dytox_ganfake_joint_training_long_1/checkpoint_0.pth"


print(modelpath)
pretrained_model = torch.load(modelpath)['model']
from models.dytox import DyTox_ptconvit

initial_increment = 2
ind_clf = "1-1"
head_div = 0.1
head_div_mode = "tr"
increment = 2

model = convit_base(pretrained=True)
model.head = Classifier(model.embed_dim, 0, initial_increment, increment, 5)

for task_id in range(5):
    if task_id == 0:
        model = DyTox_ptconvit(
            model,
            nb_classes=initial_increment,
            individual_classifier=ind_clf,
            head_div=head_div > 0.,
            head_div_mode=head_div_mode,
            joint_tokens=False
        )
    else:
        model.add_model(increment)

model.load_state_dict(pretrained_model, strict=True)

device = "cuda:0"
model = model.to(device)
test_dataset = DummyDataset(args.dataroot, args.datatype)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)


# calculate for wacv
# y_pred, y_true = [], []
# for _, (path, inputs, targets) in enumerate(test_loader):
#     inputs = inputs.to(device)
#     targets = targets.to(device)
#     with torch.no_grad():
#         outputs = torch.softmax(model(inputs)['logits'], 1)
#         # outputs = model(inputs)['logits']
#
#     output_norm = outputs / torch.sum(outputs, 1, keepdim=True)
#     output_norm = output_norm.detach().cpu()
#     output = outputs.detach().cpu()
#     task = int(output.size(1) / 2)
#     output_real = torch.zeros(output.size(0), task)
#     output_fake = torch.zeros(output.size(0), task)
#     for i in range(task):
#         output_real[:, i] = output[:, i * 2]
#         output_fake[:, i] = output[:, i * 2 + 1]
#     output_max_real, _ = torch.max(output_real, 1)
#     output_max_fake, _ = torch.max(output_fake, 1)
#     pred = torch.div(output_max_fake, (output_max_real + output_max_fake))
#     y_pred.append(pred.cpu().numpy())
#     y_true.append(targets.cpu().numpy())


y_pred, y_true = [], []
for _, (path, inputs, targets) in enumerate(test_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        outputs = model(inputs)['logits']
    predicts = torch.topk(outputs, k=2, dim=1, largest=True, sorted=True)[1]
    y_pred.append(predicts.cpu().numpy())
    y_true.append(targets.cpu().numpy())

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

# import pdb;pdb.set_trace()


from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score

# print(average_precision_score(y_true%2, y_pred))

def multimetric_binary(y_pred, y_true, increment=2):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_f1 = {}
    all_precision = {}
    all_recall = {}
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_f1[label] = f1_score(y_true[idxes]%2, y_pred[idxes]%2, average='binary')
        all_precision[label] = precision_score(y_true[idxes]%2, y_pred[idxes]%2, average='binary')
        all_recall[label] = recall_score(y_true[idxes]%2, y_pred[idxes]%2, average='binary')

    return all_f1, all_precision, all_recall


f1, pre, recal = multimetric_binary(y_pred.T[0], y_true)
print(f1)
print(pre)
print(recal)

# subsets = ["GauGAN", "BigGAN", "CycleGAN", "IMLE", "Deepfake", "CRN", "WildDeepfake"]
# subsets = ["GauGAN", "BigGAN", "WildDeepfake", "WhichFaceIsReal", "SAN"]
# subsets = ["GauGAN", "BigGAN", "CycleGAN", "IMLE", "Deepfake", "CRN", "WildDeepfake", "GLOW", "StarGAN", "StyleGAN",
#            "WhichFaceIsReal", "SAN"]
# plt.figure(1)
# for i in range(7):
#     inds = (y_true >= i)*(y_true <=  (i+1))
#     precision, recall, thre = precision_recall_curve(y_true[inds]%2, y_pred[inds])
#     plt.plot(recall, precision, label = subsets[i])
# plt.title('Precision/Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.savefig('ganfake_easy_m1500_sumasig.png')




def accuracy_binary(y_pred, y_true, increment=2):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}

    all_acc['total'] = np.around((y_pred%2 == y_true%2).sum()*100 / len(y_true), decimals=2)
    task_acc = []
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2)
        task_acc.append(np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2))
    all_acc['task_wise'] = sum(task_acc)/len(task_acc)
    return all_acc

print(accuracy_binary(y_pred.T[0], y_true))







