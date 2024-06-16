import torch.nn as nn
import torch.nn.functional as F

def make_mlp(*dims):
    layers = [nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims)-1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[i], dims[i+1]))
    return nn.Sequential(*layers)


def t_prod(a, b):
    return (a.permute(1, 0) * b.permute(2, 1, 0)).permute(2, 1, 0)
