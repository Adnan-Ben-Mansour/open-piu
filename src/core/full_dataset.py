import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import tqdm

from src.utils.make_dataset import Compactor


class FullDataset(data.Dataset):
    def __init__(self, n=112, k=16, folder:str='./data/dataset/'):
        super().__init__()

        self.datacsv = pd.read_csv(folder+"data.csv")
        self.compactor = Compactor()
        self.compactor.load(folder+"compactor.json")

        self.files = []
        self.idxs = []
        self.n, self.k = n, k
        self.nk = self.n + self.k

        self.p2 = np.array([i%10>4 for i in range(20)])

        for i, elem in tqdm.tqdm(self.datacsv.iterrows()):
            self.files.append(folder+elem["data"]+".npy")
            if elem["level"] < 14:
                data = np.load(self.files[-1])
                realdata = self.compactor.convert(data)
                
                for j in range(elem["length"]-self.nk):
                    if np.sum(realdata[10:, j:j+self.nk]) == 0: # only TAP notes
                        if realdata[:, j+self.n:j+self.nk].sum() > 0: # not always 0 as target
                            self.idxs.append((i, j, 0)) # normal
                            self.idxs.append((i, j, 1)) # reversed

                            if realdata[self.p2, j:j+self.nk].sum() == 0: # if P1 only
                                self.idxs.append((i, j, 2)) # to P2 simply
                                self.idxs.append((i, j, 3)) # to P1 reversed
            if i > 50: break
    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        i, j, s = self.idxs[idx]
        data = np.load(self.files[i])[j:j+self.nk]
        
        vdata = self.compactor.convert(data)
        vdata = vdata[self.compactor.ts[s], :]
        data = self.compactor.read_fast(vdata)

        return vdata[:, :self.n].T, data[self.n:]


