import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.utils.torch_aux import make_mlp, t_prod


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x # self.norm(x)


class NKTRANSFO(nn.Module):
    def __init__(self, n:int=64, d:int=15):
        super().__init__()

        self.n = n
        self.d = d

        self.intern_dim = 50
        hdim = 100

        self.emb_input = make_mlp(self.d, hdim, hdim, hdim, self.intern_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n + 1, self.intern_dim))

        self.transformer = Transformer(self.intern_dim, 8, 8, self.intern_dim, self.intern_dim)

        self.emb_query_level = nn.Embedding(30*2, self.intern_dim) # 30x2 levels
        self.emb_query_level.weight.data.fill_(-1.0)
        self.proj = make_mlp(self.intern_dim, hdim, hdim, hdim, self.d)
    

    def forward(self, x, l, m=None):
        # x: (B, N, D)
        # l: (B,)
        # m: (B, N+1)
        if m is None:
            m = torch.ones((x.shape[0], self.n+1))
            m[:, :x.shape[1]] = 0 # p
            m = m.to(device=x.device, dtype=torch.float32)

        x = self.emb_input(x)
        padded_x = F.pad(x, (0, 0, 0, 1, 0, 0)) # (B, N+1, d)
        emb_l = self.emb_query_level(l)

        mpx = t_prod(1 - m, padded_x) + t_prod(m, torch.bmm(m.unsqueeze(2), emb_l.unsqueeze(1))) # (B, N+1, d)
        mpx += self.pos_embedding

        mpx = self.transformer(mpx)

        y = self.proj(mpx)

        return y
