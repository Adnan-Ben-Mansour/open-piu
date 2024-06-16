import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np


def train_alpha(model_base, model_fine, dataset, ds, **settings):
    device = settings.get("device", 'cuda')
    nb_epochs = settings.get("nb_epochs", 100)
    batchsize = settings.get("batchsize", 64)
    samplersize = settings.get("samplersize", 1000)
    learningrate = settings.get("learningrate", 1e-4)
    output_base = settings.get("output_base", './src/weights/alpha-base.pt')
    output_fine = settings.get("output_fine", './src/weights/alpha-fine.pt')

    cst_k = settings.get("cst_k", 5)
    cst_n = ds.n - cst_k
    cst_l = len(ds.levels)

    trainloader = data.DataLoader(dataset, batch_size=batchsize, sampler=data.RandomSampler(dataset, True, samplersize), num_workers=24)
    

    optim_base = torch.optim.Adam(model_base.parameters(), lr=learningrate)
    optim_fine = torch.optim.Adam(model_fine.parameters(), lr=learningrate)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_cl = nn.MSELoss(reduction='sum')

    model_base.train()
    model_base.to(device=device)

    model_fine.train()
    model_fine.to(device=device)

    for epoch in range(nb_epochs):
        losses = {'nbsteps': 0.,
                  'double': 0.,
                  'level': 0.,
                  'playstyle': 0.,
                  'final': 0.}
        b_total = 0

        for l, d, x, v in tqdm.tqdm(trainloader):
            bsize = l.shape[0]
            l = l.to(device=device, dtype=torch.int64)
            d = d.to(device=device, dtype=torch.int64)
            x = x.to(device=device, dtype=torch.int64)
            v = v.to(device=device, dtype=torch.int64)
            
            k = np.random.randint(1, cst_k+1)
            

            # x_hot = F.one_hot(x, num_classes=ds.d) # (B, N+K+1, D)
            x_hot_taps = v[:, :, :10].sum(axis=-1) # (B, N+K+1)
            x_hot_holds = v[:, :, 10:].sum(axis=-1) # (B, N+K+1)
            x_conf_cat = 5*x_hot_taps + x_hot_holds - x_hot_taps*(x_hot_taps-1)//2
            x_conf_cat = torch.clamp(x_conf_cat, 0, 14) # (B,N+K+1):15
            x_conf = F.one_hot(x_conf_cat, num_classes=15) # (B, N+K+1, 15)

            x_base = torch.cat([v, x_conf], dim=-1).to(dtype=torch.float32) # (B, N+K+1, 20+15)

            x1 = x_base[:, -cst_n-1:-1, :]
            y1 = x_base[:, -1, :]
            x2 = x_base[:, -cst_n-k-1:-k-1, :]
            y2 = x_base[:, -k-1, :]

            y1_hat = model_base(x1) # (B, 15+2+L+2)
            # y2_hat = model_base(x2)

            # y_out = model_fine(x1[:, -cst_n//4:, :], x_conf[:, -1, :], l, F.one_hot(d, num_classes=2), y1_hat[:, -2:]) # (B, D)
            
            # loss_nbsteps    = 100 * loss_fn(y1_hat[:, :15], x_conf_cat[:, -1])
            loss_double     = loss_fn(y1_hat[:, 15:17], d)
            # loss_level      = loss_fn(y1_hat[:, 17:17+cst_l], l)
            # loss_playstyle  = loss_cl(y1_hat[:, -2:], y2_hat[:, -2:])
            # loss_final      = 100 * loss_fn(y_out, x[:, -1])

            loss = loss_double # loss_nbsteps + loss_double + loss_level + loss_playstyle + loss_final

            with torch.no_grad():
                # losses['nbsteps'] += loss_nbsteps.detach().item()
                losses['double'] += loss_double.detach().item()
                # losses['level'] += loss_level.detach().item()
                # losses['playstyle'] += loss_playstyle.detach().item()
                # losses['final'] += loss_final.detach().item()
                b_total += bsize

            optim_base.zero_grad()
            optim_fine.zero_grad()
            loss.backward()
            optim_base.step()
            optim_fine.step()

        print(f"{epoch+1}/{nb_epochs} losses >>> nbsteps = {losses['nbsteps']/b_total} | double = {losses['double']/b_total} | level = {losses['level']/b_total} | playstyle = {losses['playstyle']/b_total} | final = {losses['final']/b_total}")
        
        if output_base:
            outputdir = os.path.dirname(output_base)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            torch.save(model_base.state_dict(), output_base)
        

        if output_fine:
            outputdir = os.path.dirname(output_fine)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            torch.save(model_fine.state_dict(), output_fine)

