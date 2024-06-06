import torch
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

class NKLSTM(nn.Module):
    def __init__(self, n:int=112, k:int=16, d:int=15):
        super().__init__()

        self.n, self.k = n, k
        self.nk = self.n + self.k
        self.d = d

        self.intern_dim = 20
        hdim = 200

        self.emb_input = make_mlp(self.d, hdim, hdim, hdim, hdim, self.intern_dim)

        self.lstm1 = nn.LSTM(input_size=self.intern_dim, hidden_size=self.intern_dim, num_layers=12, batch_first=True)
        self.emb_query_level = nn.Embedding(30, self.intern_dim) # 30 levels

        self.proj = make_mlp(self.intern_dim, hdim, hdim, hdim, hdim, self.d)
    
    def forward(self, x, l, m=None):
        # x: (B, N, D)
        # l: (B,)
        # m: (B, N+K)
        if m is None:
            m = torch.ones((x.shape[0], self.nk))
            m[:, :x.shape[1]] = 0 # p
            m = m.to(device=x.device, dtype=torch.float32)

        x = self.emb_input(x)
        padded_x = F.pad(x, (0, 0, 0, self.k, 0, 0)) # (B, N+K, d)
        emb_l = self.emb_query_level(l)

        mpx = t_prod(1 - m, padded_x) + t_prod(m, torch.bmm(m.unsqueeze(2), emb_l.unsqueeze(1))) # (B, N+K, d)


        y, _ = self.lstm1(mpx, )
        y = self.proj(y)
        # y: (B, N+K, D)

        return y