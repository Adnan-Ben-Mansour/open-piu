import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import tqdm

from src.core.compactor import Compactor


class ChartDataset(data.Dataset):
    def __init__(self, n=112, folder:str='./data/dataset/', brk=-1):
        super().__init__()

        self.datacsv = pd.read_csv(folder+"data.csv")
        self.compactor = Compactor()
        self.compactor.load(folder+"compactor.json")

        self.files = []
        self.idxs = []

        self.n = n
        self.d = len(self.compactor.labels)

        print(f"Loading dataset (n={self.n}, d={self.d})...")

        for i, elem in tqdm.tqdm(self.datacsv.iterrows(), total=len(self.datacsv)):
            self.files.append(folder+elem["data"]+".npy")
            level = elem["level"]
            single = elem["single"]
            length = elem["length"]
            
            sd_level = level + 30*(1-int(single))

            if 0 <= sd_level < 60: # LEVEL
                for j in range(length-self.n-1):
                    self.idxs.append((i, j, 0, sd_level)) # normal
                    self.idxs.append((i, j, 1, sd_level)) # reversed
                    if single: # if P1 only
                        self.idxs.append((i, j, 2, sd_level)) # to P2 simply
                        self.idxs.append((i, j, 3, sd_level)) # to P1 reversed

            if i > brk >= 0:
                break
    
    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        i, j, s, l = self.idxs[idx] # file, position, data-augmentation, level

        data = np.load(self.files[i])[j:j+self.n+1] # (N+1,)
        
        vdata = self.compactor.convert(data) # (N+1, 20)
        vdata = vdata[:, self.compactor.ts[s]] # (N+1, 20)
        data = self.compactor.read_fast(vdata) # (N+1,)
        
        return l, data[:self.n], data[self.n:]
