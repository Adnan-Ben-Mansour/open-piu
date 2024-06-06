import glob
import tqdm
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data


from src.core.make_dataset import make_dataset

from src.core.tempo_dataset import TempoDataset

from src.models.lstm import NKLSTM
from src.models.mlp import MLP

from src.core.train_tempo import train_tempo

from src.core.generate_tempo import generate_tempo

if __name__ == "__main__":
    mk_dataset = False
    ld_dataset = True
    fld_dataset = True
    mk_model = True
    tr_model = True and (fld_dataset and mk_model)
    
    
    if mk_dataset:
        make_dataset()
        exit(0)
    
    n = 32
    k = 4

    if ld_dataset:
        ds = TempoDataset(n, k, brk=not(fld_dataset))
        test_size = int(len(ds)*0.1)
        train_size = len(ds) - test_size
        print(f"Dataset: train({train_size}), test({test_size})")
        trainset, testset = torch.utils.data.random_split(ds, [train_size, test_size])
    
    if mk_model:
        model_name = "old_LSTM"
        model = NKLSTM(n, k, len(ds.compactor.labels))

    output = f'./src/weights/{model_name}_{n}_{k}.pt'
    if os.path.exists(output):
        model.load_state_dict(torch.load(output))
    if tr_model:
        train_tempo(model, trainset, nb_epochs=50, learningrate=4e-5, batchsize=64, samplersize=500_000, device='cuda', output=output)
    
    
    tempo_seed = np.zeros(112, dtype=np.int64)
    for i in range(19):
        tempo_seed[-i*4-1-8] = 5
    
    tdata = []
    for l in range(5, 16):
        tdata.append(generate_tempo(model, tempo_seed, 1024, n, k, l))
        print(tdata[-1])
    
    import PIL.Image as Image
    colors = np.array([[0,0,0], [255,250,0], [255,150,0], [255,50,0], [255,0,0],
                       [0,255,0], [255,0,250], [255,0,200], [255,0,150],
                       [0,0,255], [255,150,150], [255,50,50],
                       [0,255,255], [255,0,0],
                       [255,255,255]], dtype=np.uint8)
    dt = np.stack(colors[tdata].swapaxes(0, 1), 0)
    print(dt.shape)
    print(dt)
    im = Image.fromarray(dt)
    im.save("outputs/sortie.png")

    exit(0)
    seed = torch.zeros((40, 112), dtype=torch.float32)
    seed = seed.to(device='cuda')
    for i in range(19):
        seed[abs(8-i)%5, -i*6-1] = 1

    seed = generate(model, seed, 400).detach().cpu().numpy()
    seed = np.round(seed)

    chart = Chart(length=seed.shape[1])
    chart.data = seed.astype(dtype=np.int64)[0]
    
    show_chart(chart.data.copy())
    exit(0)

    if False:
        if not os.path.exists("./data/examples/"):
            os.makedirs("./data/examples/")
        for i in range(10):
            show_chart(trainset[i][0].astype(np.int64), f"./data/examples/chart_{i}.png")

    # chart = read("/home/adnan/Bureau/Projetarium/open-piu/data/PIU-Simfiles-main/14 - XX/1670 - 8 6/1670 - 8 6.ssc")
    displayer = Displayer()
    displayer.running_chart = chart
    displayer.reader = chart.read()
    displayer.run()