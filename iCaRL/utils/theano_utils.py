# from types import NoneType
from typing import List, Sequence, Union, Tuple, Any

import math
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable


from .training_utils import extract_features_from_layer, extract_features_from_layer_to_binary
from .dataset_utils import make_batch_one_hot
from .essential_utils import  moment_update, TransformTwice, weight_norm, mixup_data, mixup_criterion, LabelSmoothingCrossEntropy



# This code was adapted from the one of lasagne.init
# which is distributed under MIT License (MIT)
# Original source: https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
class LasagneInitializer(object):
    def __call__(self, shape: Union[List[int], Tuple[int]]) -> Tensor:
        return self.sample(shape)

    def sample(self, shape: Union[List[int], Tuple[int]]) -> Tensor:
        raise NotImplementedError()


class LasagneNormal(LasagneInitializer):
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return torch.normal(self.mean, self.std, size=shape)


class LasagneHe(LasagneInitializer):
    def __init__(self, initializer, gain: Any = 1.0, c01b: bool = False):
        if gain == 'relu':
            gain = math.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            fan_in = torch.prod(torch.tensor(shape[:3]))
        else:
            if len(shape) == 2:
                fan_in = shape[0]
            elif len(shape) > 2:
                fan_in = torch.prod(torch.tensor(shape[1:]))
            else:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

        std = self.gain * math.sqrt(1.0 / fan_in)
        return self.initializer(std=std).sample(shape)


class LasagneHeNormal(LasagneHe):
    def __init__(self, gain=1.0, c01b=False):
        super().__init__(LasagneNormal, gain, c01b)
# End of lasagne-adapted init code


def make_theano_training_function(model: Module, criterion: Module, optimizer: Optimizer, x: Tensor, y: Tensor,
                                  device=None) -> \
        float:
    model.train()
    model.zero_grad()
    if device is not None:
        x = x.to(device)
        y = y.to(device)

    output = model(x)
    loss: Tensor = criterion(output, y)
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()


def make_theano_training_function_mixup( model: Module, criterion: Module, optimizer: Optimizer,args, x: Tensor, y: Tensor,y_old = None,
                                  device=None) -> \
        float:
    model.train()
    model.zero_grad()

    inputs, targets_a, targets_b, lam = mixup_data(x, y, args.mixup_alpha)
    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

    if device is not None:
        inputs = inputs.to(device)
        targets_a = targets_a.to(device)
        targets_b = targets_b.to(device)


    logits = model.get_logits(inputs)
    outputs = logits
    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    if y_old == None:
        loss_kd = 0
    else:
        output = model(x)
        loss_kd = criterion(output[:, :y_old.size(1)], y_old)
    loss_ce: Tensor = mixup_criterion(criterion_ce, outputs, targets_a, targets_b, lam)
    loss = loss_kd + loss_ce
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()


def make_theano_training_function_ls( model: Module, criterion: Module, optimizer: Optimizer,args, x: Tensor, y: Tensor, y_old = None,
                                  device=None) -> \
        float:
    model.train()
    model.zero_grad()

    criterion_ce_smooth = LabelSmoothingCrossEntropy()

    if device is not None:
        x = x.to(device)
        y = y.to(device)


    logits = model.get_logits(x)
    if y_old == None:
        loss_kd = 0
    else:
        output = model(x)
        loss_kd = criterion(output[:, :y_old.size(1)], y_old)
    loss_ce: Tensor = criterion_ce_smooth(logits, y, args.smoothing_alpha)
    loss = loss_kd + loss_ce
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()
    

def make_theano_training_function_binary(model: Module, criterion: Module, optimizer: Optimizer, x: Tensor, y: Tensor, y_old: Tensor,
                                  device=None) -> \
        float:
    model.train()
    model.zero_grad()
    if device is not None:
        x = x.to(device)
        y = y.to(device)

    output = model(x)
    loss_new: Tensor = criterion(output.squeeze(1), y.float())
    loss_old: Tensor = criterion(output.squeeze(1), y_old.float())
    loss = 0.9*loss_new + 0.1*loss_old
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()

def make_theano_training_function_add_binary(model: Module, criterion: Module, optimizer: Optimizer, args, x: Tensor, y: Tensor,
                                  device=None) -> \
        float:
    model.train()
    model.zero_grad()
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    y_binary_ = y%2
    y_binary = make_batch_one_hot(y_binary_, 2).to(device)
    output = model(x)
    fc_output = model.get_logits(x)
    y = make_batch_one_hot(y, output.size(1)).to(device)
    # output_binary = torch.zeros_like(output[:, 0:2], requires_grad=False).to(device)
    output_binary = torch.zeros(output.size(0), 2).to(device)
    task_num = int(output.size(1)/2)
    if args.binary_loss == 'sum_b_sig':
    ##sum before sigmoid
        for i in range(task_num):
            # output_binary[:, 0] += output[:, i*2] 
            # output_binary[:, 1] += output[:, i*2+1]
            output_binary[:, 0] += fc_output[:, i*2] 
            output_binary[:, 1] += fc_output[:, i*2+1]
        output_binary = torch.sigmoid(output_binary)
        loss_binary: Tensor = criterion(output_binary, y_binary)
    elif args.binary_loss == 'sum_a_sig':
    ##sum after sigmoid
        for i in range(task_num):
            output_binary[:, 0] += output[:, i*2] 
            output_binary[:, 1] += output[:, i*2+1]
        output_binary = output_binary/torch.sum(output_binary, 1, keepdim = True)
        loss_binary: Tensor = criterion(output_binary, y_binary)
    elif args.binary_loss == 'sum_b_log':
    ##sum outside log
        for i in range(task_num):
            output_binary[:, 0] += torch.log(output[:, i*2]) 
            output_binary[:, 1] += torch.log(output[:, i*2+1])
        output_binary = output_binary/torch.sum(output_binary, 1, keepdim = True) 
        loss_binary = ((1-y_binary_).mul(output_binary[:, 0]) + y_binary_.mul(output_binary[:, 1])).sum()/(output.size(0))
        loss_binary = Variable(loss_binary, requires_grad=True)
        # print(output_binary,y_binary)
    else:
    ##max 
        output_real = torch.zeros(output.size(0), task_num)
        output_fake = torch.zeros(output.size(0), task_num)
        for i in range(task_num):
            output_real[:, i] = output[:, i*2]
            output_fake[:, i] = output[:, i*2+1]
        output_max_real,_ = torch.max(output_real, 1)
        output_max_fake,_ = torch.max(output_fake, 1)
        output_binary[:, 0] = output_max_real
        output_binary[:, 1] = output_max_fake
        output_binary = output_binary/torch.sum(output_binary, 1, keepdim = True)
        loss_binary: Tensor = criterion(output_binary, y_binary)

    loss_multi: Tensor = criterion(output, y)
    loss = (1-args.binary_weight)*loss_multi + args.binary_weight*loss_binary
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()

def make_theano_validation_function(model: Module, criterion: Module, feature_extraction_layer: str,
                                    x: Tensor, y: Tensor, device=None) -> (float, Tensor, Tensor):
    output: Tensor
    output_features: Tensor
    loss: Tensor

    model.eval()
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        output, output_features = extract_features_from_layer(model, feature_extraction_layer, x)
        loss = criterion(output, y)

    return loss.detach().cpu().item(), output, output_features

def make_theano_validation_function_to_binary(model: Module, criterion: Module, feature_extraction_layer: str,
                                    x: Tensor, y: Tensor, device=None) -> (float, Tensor, Tensor):
    output: Tensor
    output_features: Tensor
    loss: Tensor

    model.eval()
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        output, output_binary, output_features = extract_features_from_layer_to_binary(model, feature_extraction_layer, x)
        loss = criterion(output, y)

    return loss.detach().cpu().item(), output_binary, output_features


def make_theano_validation_function_binary(model: Module, criterion: Module, feature_extraction_layer: str,
                                    x: Tensor, y: Tensor, device=None) -> (float, Tensor, Tensor):
    output: Tensor
    output_features: Tensor
    loss: Tensor

    model.eval()
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        output, output_features = extract_features_from_layer(model, feature_extraction_layer, x)
        loss = criterion(output.squeeze(1), y.float())

    return loss.detach().cpu().item(), output, output_features

def make_theano_inference_function(model: Module, x: Tensor, device=None) -> Tensor:
    output: Tensor

    model.eval()
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
        output = model(x)

    return output


def make_theano_feature_extraction_function(model: Module, feature_extraction_layer: str, x: Tensor,
                                            device=None, **kwargs) -> Tensor:
    output_features: List[Tensor] = []

    x_dataset = TensorDataset(x)
    x_dataset_loader = DataLoader(x_dataset, **kwargs)

    model.eval()
    with torch.no_grad():
        for (patterns,) in x_dataset_loader:
            if device is not None:
                patterns = patterns.to(device)
            output_features.append(extract_features_from_layer(model, feature_extraction_layer, patterns)[1])

    return torch.cat(output_features)
