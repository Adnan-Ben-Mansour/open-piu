import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


def train_tempo(model, dataset, **settings):
    nb_epochs = settings.get("nb_epochs", 10)
    device = settings.get("device", 'cuda')
    batchsize = settings.get("batchsize", 64)
    samplersize = settings.get("samplersize", 200_000)
    learningrate = settings.get("learningrate", 1e-4)
    output = settings.get("output", './src/weights/mlp.pt')

    trainloader = data.DataLoader(dataset, batch_size=batchsize, sampler=data.RandomSampler(dataset, True, samplersize), num_workers=24)
    

    optim = torch.optim.Adam(model.parameters(), lr=learningrate)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    model.train()
    model.to(device=device)

    for epoch in range(nb_epochs):
        e_loss = 0.0
        for l, x, y in tqdm.tqdm(trainloader):
            l = l.to(device=device, dtype=torch.int64)
            x = x.to(device=device, dtype=torch.int64)
            y = y.to(device=device, dtype=torch.int64)

            fx = F.one_hot(x, num_classes=model.d).to(dtype=torch.float32)

            m = torch.ones((x.shape[0], x.shape[1]+y.shape[1]))
            m[:, :x.shape[1]] = 0.1 # p
            m = torch.bernoulli(m).to(device=device, dtype=torch.float32)

            y_hat = model(fx, l, m)
            
            loss = loss_fn(y_hat.mT, torch.cat([x, y], dim=-1))
            with torch.no_grad():
                e_loss += loss.detach().item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"{epoch+1}/{nb_epochs} >>> loss = {e_loss/samplersize}")
        
        if output:
            outputdir = os.path.dirname(output)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            torch.save(model.state_dict(), output)

