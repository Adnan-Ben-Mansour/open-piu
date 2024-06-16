import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import tqdm

from src.core.compactor import Compactor


class AlphaDataset(data.Dataset):
    def __init__(self, n=63, folder:str='./data/dataset/', brk=-1):
        super().__init__()

        self.datacsv = pd.read_csv(folder+"data.csv")
        self.compactor = Compactor()
        self.compactor.load(folder+"compactor.json")

        self.files = []
        self.idxs = []

        self.n = n
        self.d = len(self.compactor.labels)

        print(f"Loading dataset (n={self.n}, d={self.d})...")

        self.levels = set()
        
        for i, elem in tqdm.tqdm(self.datacsv.iterrows(), total=len(self.datacsv)):
            self.files.append(folder+elem["data"]+".npy")

            level = elem["level"]
            double = int(not elem["single"])
            length = elem["length"]

            self.levels.add(level)

            if (level != 99):
                for j in range(length-self.n-1):
                    self.idxs.append((i, j, level, double))

            if i > brk >= 0:
                break

        self.levels = list(sorted(self.levels))
        print(f"Dataset loaded (levels={self.levels}).")
    
    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        i, j, l, d = self.idxs[idx] # file, position, level, double

        data = np.load(self.files[i])[j:j+self.n+1] # (N+1,)
        
        if d == 1:
            s = np.random.choice([0, 1])
        else: 
            s = np.random.choice([0, 1, 2, 3])
        
        vdata = self.compactor.convert(data) # (N+1, 20)
        vdata = vdata[:, self.compactor.ts[s]] # (N+1, 20)
        data = self.compactor.read_fast(vdata) # (N+1,)
        
        return self.levels.index(l), d, data, vdata
