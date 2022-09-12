from typing import Dict
from pandas import RangeIndex

import torch
from torch import Tensor
from torch.nn import Module


def extract_features_from_layer(model: Module, layer_name: str, x: Tensor) -> (Tensor, Tensor):
    activation: Dict[str, Tensor] = {}

    def get_activation(name):
        def hook(model_hook: Module, x_hook: Tensor, out_hook: Tensor):
            activation[name] = out_hook.detach().cpu()
        return hook

    model.eval()
    with torch.no_grad():
        with getattr(model, layer_name).register_forward_hook(get_activation(layer_name)):
            output = model(x)

    return output, activation[layer_name]

def extract_features_from_layer_to_binary(model: Module, layer_name: str, x: Tensor) -> (Tensor, Tensor):
    activation: Dict[str, Tensor] = {}

    def get_activation(name):
        def hook(model_hook: Module, x_hook: Tensor, out_hook: Tensor):
            activation[name] = out_hook.detach().cpu()
        return hook

    model.eval()
    with torch.no_grad():
        with getattr(model, layer_name).register_forward_hook(get_activation(layer_name)):
            output = model(x)
            fc_output = model.get_logits(x)
        output_binary = torch.zeros(output.size(0), 2).cuda()
        for i in range(fc_output.size(1)//2):
            output_binary[:, 0] += fc_output[:, i*2] 
            output_binary[:, 1] += fc_output[:, i*2+1]
        output_binary = torch.sigmoid(output_binary)

    return output,output_binary, activation[layer_name]
