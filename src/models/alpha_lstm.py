import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.torch_aux import make_mlp, t_prod


class ALPHALSTM(nn.Module):
    def __init__(self, n:int, l:int, input_dim:int):
        """
        n: sequence size
        l: number of levels
        input_dim: input dimension
        output_dim: output dimension
        """
        super().__init__()

        self.n = n
        self.l = l
        self.input_dim = input_dim # 20+15
        self.output_dim = 19+self.l

        self.intern_dim = 20
        hdim = 50

        self.emb_input = make_mlp(self.input_dim, hdim, hdim, self.intern_dim)

        self.lstm1 = nn.LSTM(input_size=self.intern_dim, hidden_size=self.intern_dim, num_layers=8, batch_first=True)

        self.proj = make_mlp(self.intern_dim, hdim, hdim, self.output_dim)
    
    def forward(self, x):
        # x: (B, N, D+15)
        # ----------------
        # l: (B,)

        x = self.emb_input(x)
        padded_x = F.pad(x, (0, 0, 0, 1, 0, 0)) # (B, N+1, d)

        y, _ = self.lstm1(padded_x)
        y = self.proj(y) # y: (B, N+1, 15+2+L+2)

        return y[:, -1, :]