import glob
import tqdm
import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data


from src.core.make_dataset import make_dataset

from src.core.alpha_dataset import AlphaDataset

from src.models.alpha_lstm import ALPHALSTM
from src.models.alpha_transformer import ALPHATRANSFO

from src.core.alpha_train import train_alpha

from src.core.generate_tempo import generate_tempo

from src.display.displayer import Displayer
from src.core.charts import Chart

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=False)
    
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument("--brk", type=int, default=-1)
    parser.add_argument("--modelname", type=str, default='alpha_006_')

    parser.add_argument("--output", type=str, default="outputs/sortie.png")
    args = parser.parse_args()

    if args.make: 
        make_dataset()
        exit(0)
    
    n = args.n
    k = args.k

    ds = AlphaDataset(n=(n+k), brk=args.brk)
    test_size = int(len(ds)*0.1)
    train_size = len(ds) - test_size
    print(f"Dataset: train({train_size}), test({test_size})")
    trainset, testset = torch.utils.data.random_split(ds, [train_size, test_size])
    

    nb_labels = len(ds.compactor.labels)
    nb_levels = len(ds.levels)
    model_base = ALPHALSTM(n=n, l=nb_levels, input_dim=20+15)
    model_fine = ALPHATRANSFO(n=n//4, l=nb_levels, input_dim=20+15, output_dim=nb_labels)


    output_base = f'./src/weights/{args.modelname}_{n}_base.pt'
    output_fine = f'./src/weights/{args.modelname}_{n//4}_fine.pt'

    if os.path.exists(output_base):
        model_base.load_state_dict(torch.load(output_base))
    if os.path.exists(output_fine):
        model_fine.load_state_dict(torch.load(output_fine))
    
    if args.train:
        train_alpha(model_base,
                    model_fine,
                    trainset,
                    ds,
                    cst_k=k,
                    nb_epochs=1000,
                    learningrate=1e-3,
                    batchsize=64,
                    samplersize=64_000,
                    device='cuda', 
                    output_base=output_base,
                    output_fine=output_fine)
    
    
    exit(0)
    # make seed
    tempo_seed = np.zeros((64, 20), dtype=np.int64)
    for i in range(28):
        if (i*8 < 60):
            tempo_seed[-i*8-1, 5+(i%2)] = 1
        if (i%4 != 0):
            tempo_seed[-i*2-1, i%5] = 1
    
    tempo_seed = ds.compactor.read_fast(tempo_seed)
    print("seed shape:", tempo_seed.shape)
    
    # generate charts
    tdata = []
    for l in [50]: # list(range(5, 20))+list(range(35,50)):
        tdata.append(generate_tempo(model, tempo_seed, 512, n, l, tmp=50, cmp=ds.compactor))
        tdata[-1] = ds.compactor.convert(tdata[-1])
        tdata.append(np.ones((tdata[-1].shape[0], 1)))
    
    # display charts
    import PIL.Image as Image
    colors = np.array([[0,0,0], [255,250,0], [255,150,0], [255,50,0], [255,0,0],
                       [0,255,0], [255,0,250], [255,0,200], [255,0,150],
                       [0,0,255], [255,150,150], [255,50,50],
                       [0,255,255], [255,0,0],
                       [255,255,255]], dtype=np.uint8)
    dt = 255*np.concatenate(tdata, 1)
    print(dt.shape)
    reds = np.stack([dt[:, i] if (i%21 in [1,3,6,8,11,13,16,18,20,2,7,12,17]) else np.zeros(dt.shape[0]) for i in range(dt.shape[1])], 1)
    greens = np.stack([dt[:, i] if (i%21 in [20, 2, 7, 12, 17]) else np.zeros(dt.shape[0]) for i in range(dt.shape[1])], 1)
    blues = np.stack([dt[:, i] if (i%21 in [0,4,5,9,10,14,15,19,20]) else np.zeros(dt.shape[0]) for i in range(dt.shape[1])], 1)
    dt = np.stack([reds, greens, blues], -1)
    nd = np.zeros((dt.shape[0]+1, (dt.shape[1]//21)*11+1, 3))
    loca = np.array([i if (i < 64) else i+1 for i in range(dt.shape[0])], dtype=np.int64)
    for l in range(16-5):
        nd[loca, l*11+1:(l+1)*11+1] = np.concatenate([0.8*dt[:, l*21:l*21+10] + 0.5*dt[:, l*21+10:l*21+20], dt[:, l*21+20:l*21+21]], 1)
    nd[64:65, :] = 255
    nd[:, 0:1] = 255

    dt = np.clip(nd, 0, 255)
    print(dt.shape)
    print(dt)
    im = Image.fromarray(dt.astype(np.uint8))
    im.save(args.output)

    print(tdata[-2])
    # u = ds.compactor.convert(tdata[-1])

    displayer = Displayer()
    displayer.running_chart = Chart(length=tdata[-2].shape[0])
    displayer.running_chart.data = tdata[-2]
    displayer.reader = displayer.running_chart.read_notes()
    displayer.run()