import torch
import torch.nn.functional as F
import numpy as np

def generate_tempo(model, seed, length=512, n=112, k=16, l=4):
    # model(N,K) for tempo
    # seed: (L,) L >= N

    device = 'cuda'

    model.eval()
    model.to(device=device)
    seed = torch.tensor(seed, dtype=torch.int64, device=device).reshape(1, -1)
    level = l * torch.ones((1,), dtype=torch.int64, device=device)
    seed = seed.to(device=device)
    with torch.no_grad():
        while seed.shape[-1] < length:
            frag = model(F.one_hot(seed[:, -n:], num_classes=15).to(dtype=torch.float32), level)[:, -k:]
            frag = torch.softmax(np.random.uniform(0.5, 5)*frag, -1)
            frag = torch.multinomial(frag[0], num_samples=1).T
            seed = torch.cat([seed, frag[:, :1]], dim=1)

    return seed.detach().cpu().numpy()[0]