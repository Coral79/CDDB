import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from PIL import Image
import torch.distributed as dist


class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.5):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.detach().data)


'''
def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        #p2.data.mul_(m).add_(1 - m, p1.detach().data)
        #p2.data.mul_(m).add_(p1.detach().data, 1-m)
        #p2.data = p2.data*m + p1.detach().data*(1-m)
        #p2.data.mul_(m).add_(1. - m, p1.data)
        p2.data.copy_(p1.data)
        p2.requires_grad = False
'''

def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


def moment_update(model, model_ema, m, global_step):
    """ model_ema = m * model_ema + (1 - m) model """
    m = min(1 - 1 / (global_step + 1), m)
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def weight_norm(model):
    w_data_a = model.fc.fc1.weight.data
    w_data_b = model.fc.fc2.weight.data

    L_norm_a = torch.norm(w_data_a, p=2, dim =1)
    L_norm_b = torch.norm(w_data_b, p=2, dim =1)
    #print ('norm_a:', L_norm_a, ' norm_b: ', L_norm_b)
    print ('norm_a mean: ', L_norm_a.mean(0), ' norm_b norm: ', L_norm_b.mean(0))
    return L_norm_a.mean(0)/L_norm_b.mean(0)

def weight_norm_dot(model):
    w_data_a = model.fc.weight.data
    L_norm_a = torch.norm(w_data_a, p=2, dim =1)
    #print ('norm_a:', L_norm_a)
    print ('norm_old mean: ', L_norm_a[:-10].mean(0), 'norm_new mean: ', L_norm_a[-10:].mean(0))
    return L_norm_a[:-10].mean(0)/L_norm_a[-10:].mean(0)
    #return L_norm_a

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
