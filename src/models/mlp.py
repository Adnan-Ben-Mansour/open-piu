import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.torch_aux import make_mlp, t_prod


class NKMLP(nn.Module):
    def __init__(self, n:int=64, d:int=15):
        super().__init__()

        self.n = n
        self.d = d

        self.intern_dim = 20
        hdim = 200

        self.emb_input = make_mlp(self.d, hdim, hdim, hdim, hdim, self.intern_dim)

        self.core_lin = make_mlp((self.n+1)*self.intern_dim, hdim, hdim, hdim, self.intern_dim)

        self.emb_query_level = nn.Embedding(30*2, self.intern_dim) # 30x2 levels
        self.emb_query_level.weight.data.fill_(-1.0)
        self.proj = make_mlp(self.intern_dim, hdim, hdim, hdim, hdim, self.d)
    

    def forward(self, x, l, m=None):
        # x: (B, N, D)
        # l: (B,)
        # m: (B, N+1)
        if True:
            m = torch.ones((x.shape[0], self.n+1))
            m[:, :x.shape[1]] = 0 # p
            m = m.to(device=x.device, dtype=torch.float32)

        x = self.emb_input(x)
        padded_x = F.pad(x, (0, 0, 0, 1, 0, 0)) # (B, N+1, d)
        emb_l = self.emb_query_level(l)

        mpx = t_prod(1 - m, padded_x) + t_prod(m, torch.bmm(m.unsqueeze(2), emb_l.unsqueeze(1))) # (B, N+1, d)
        mpx = mpx.view(-1, (self.n+1)*self.intern_dim)
        dx = self.core_lin(mpx)
        mpx = mpx.view(-1, self.n+1, self.intern_dim)
        mpx = torch.concatenate([mpx[:, :-1], dx.view(-1, 1, self.intern_dim)], dim=1)
        y = self.proj(mpx)

        return y
