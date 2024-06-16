import torch
import torch.nn.functional as F
import numpy as np

def generate_alpha(model_base, model_fine, seed, length=512, n=112, l=4, tmp=55, cmp=None):
    # model(N,D) for tempo
    # seed: (L,) L >= N

    device = 'cuda'

    model_base.eval()
    model_base.to(device=device)
    
    model_fine.eval()
    model_fine.to(device=device)

    seed = torch.tensor(seed, dtype=torch.int64, device=device).reshape(1, -1)
    level = l * torch.ones((1,), dtype=torch.int64, device=device)
    print(f"Making level {(level%30).item()} ({['single', 'double'][level//30]}).")
    seed = seed.to(device=device)
    with torch.no_grad():
        while seed.shape[-1] < length:
            frag = model(F.one_hot(seed[:, -n:], num_classes=model.d).to(dtype=torch.float32), level)[:, -1:]
            frag = torch.softmax(frag, -1)
            if frag[0, 0, 0] <= 0.1:
                xtmp = 0.6*tmp if (level[0]<30) else 0.8*tmp
                qs = torch.quantile(frag, q=3105/3109)
                if level[0] <= 30:
                    frag = 0.4 * frag * (frag >= qs) + 5 * (frag >= qs)
                else:
                    frag = 0.4 * frag * (frag >= qs) + 5 * (frag >= qs)
                # print(frag)
            else:
                xtmp = tmp
            frag = torch.softmax(xtmp*frag, -1)
            if (xtmp < tmp) and (level[0] <= 30):
                fg = (frag*100).to(dtype=torch.int64)[0, 0].detach().cpu().numpy()
                print('-'*50)
                for a in range(fg.shape[0]):
                    if fg[a]>0:
                        print(f"- {cmp.convert(np.array([a], dtype=np.int64).reshape(1, -1))[0, 0]} ({fg[a]}%)")
            frag = torch.multinomial(frag[0], num_samples=1).T
            seed = torch.cat([seed, frag[:, :1]], dim=1)

    return seed.detach().cpu().numpy()[0]