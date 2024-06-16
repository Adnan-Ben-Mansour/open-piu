import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import tqdm

from src.core.compactor import Compactor


class TempoDataset(data.Dataset):
    def __init__(self, n=112, k=16, folder:str='./data/dataset/', brk=False):
        super().__init__()

        self.datacsv = pd.read_csv(folder+"data.csv")
        self.compactor = Compactor()
        self.compactor.load(folder+"compactor.json")

        self.files = []
        self.idxs = []
        self.n, self.k = n, k
        self.nk = self.n + self.k

        self.p2 = np.array([i%10>4 for i in range(20)]) # player2 notes

        for i, elem in tqdm.tqdm(self.datacsv.iterrows(), total=len(self.datacsv)):
            self.files.append(folder+elem["data"]+".npy")
            l = elem["level"]
            sd = elem["single"]
            l = l + 30*(1-int(sd))

            if l < 60: # LEVEL
                data = np.load(self.files[-1]) # (L,)
                realdata = self.compactor.convert(data) # (L, 20)
                
                
                for j in range(elem["length"]-self.nk):
                    if True or (np.sum(realdata[j:j+self.nk, 10:]) == 0): # only TAP notes
                        self.idxs.append((i, j, 0, l)) # normal
                        self.idxs.append((i, j, 1, l)) # reversed

                        if realdata[j:j+self.nk, self.p2].sum() == 0: # if P1 only
                            self.idxs.append((i, j, 2, l)) # to P2 simply
                            self.idxs.append((i, j, 3, l)) # to P1 reversed
            if i > 50 and brk: break
    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        i, j, s, l = self.idxs[idx] # file, position, data-augmentation

        data = np.load(self.files[i])[j:j+self.nk] # (L,)
        
        vdata = self.compactor.convert(data) # (L, 20) !
        vdata = vdata[:, self.compactor.ts[s]] # (N+K, 20)
        data = self.compactor.read_fast(vdata)

        taps = vdata[:, :10].sum(axis=1)
        holds = vdata[:, 10:].sum(axis=1)
        vdata = 5*taps + holds - taps*(taps-1)//2 # (N+K,)
        
        return l, data[:self.n], data[self.n:]
