import json
import numpy as np


class Compactor:
    def __init__(self):
        self.ts = [np.array([i for i in range(20)], dtype=np.int64),                                         # T0 = IDENTITY
                   np.array([10*(i//10) + (9-i%10) for i in range(20)], dtype=np.int64),                     # T1 = FULL REVERSED, P1->P2R
                   np.array([10*(i//10) + (5+i)%10 for i in range(20)], dtype=np.int64),                     # T2 = P1->P2
                   np.array([10*(i//10) + (4-i%10 if i%10<5 else i%10) for i in range(20)], dtype=np.int64), # T3 = P1->P2
                   ] 
        
        self.labels = []
        self.label_keys = {}
        self.occurrences = {}
    
    def save(self, filename):
        print(f"[INFO]: Saving compactor to {filename}.")

        tab = {}
        tot = 0
        for t in self.occurrences:
            i = self.occurrences[t]
            if i not in tab:
                tab[i] = [t]
            else:
                tab[i].append(t)
        
        for i in sorted(tab.keys(), reverse=True):
            tot += len(tab[i])
            print(f'- {i}: {tot} ({len(tab[i])})')

        with open(filename, 'w') as file:
            json.dump(self.labels, file)

    def load(self, filename):
        with open(filename, 'r') as file:
            self.labels = json.load(file)
        self.label_keys = {tuple(t): i for i, t in enumerate(self.labels)}
        self.labels = [np.array(t, dtype=np.int64) for t in self.labels]

    def read(self, array):
        # array: (L, 20)
        res = []
        for i in range(array.shape[0]):
            t_array_i = tuple(array[i].tolist())
            if t_array_i not in self.label_keys:
                self.label_keys[t_array_i] = len(self.labels)
                self.occurrences[t_array_i] = 0
                self.labels.append(t_array_i)
            res.append(self.label_keys[t_array_i])
            self.occurrences[t_array_i] += 1
        return np.array(res, dtype=np.int64)
    
    def read_slow(self, array):
        # array: (L, 20)
        # -> read(T_i(array)) | T0=ID, T1=REVERSED, T2=P2, T3=P2+REVERSED
        
        self.read(array[:, self.ts[1]])
        self.read(array[:, self.ts[2]])
        self.read(array[:, self.ts[3]])
        return self.read(array)

    def read_fast(self, array):
        res = []
        for i in range(array.shape[0]):
            res.append(self.label_keys[tuple(array[i].tolist())])
        return np.array(res, dtype=np.int64)
    
    def convert(self, labels):
        # labels: (L:np.int64)
        return np.array(self.labels, dtype=np.int64)[labels]

