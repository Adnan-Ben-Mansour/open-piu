import torch
import torch.nn as nn

def MLP(n:int=112, k:int=16):
    # x: (B, N, 15)
    return nn.Sequential(
            nn.Flatten(), # (B, 15N)
            nn.Linear(15*n,100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100,  k*15),
            nn.Unflatten(1, torch.Size((k, 15))),
        )