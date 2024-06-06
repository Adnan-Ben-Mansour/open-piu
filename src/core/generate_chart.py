import torch
import tqdm


def generate_chart(model, seed, k=1):
    model.eval()
    model.to(device='cuda')
    
    seed = seed.T
    seed = seed.reshape(1, *seed.shape) # 1, 40, N

    with torch.no_grad():
        for i in tqdm.tqdm(range(k)):
            frag = model(seed[:, :, -112:])
            frag = torch.softmax(1.1*frag, dim=-1)
            frag = torch.multinomial(frag[0], num_samples=1)
            frag = ds.compactor.convert(frag.detach().cpu().numpy()[:, 0]).T
            frag = torch.tensor(frag, dtype=torch.float32, device='cuda').unsqueeze(0)
            frag = frag[:, :1, :]
            seed = torch.cat([seed, frag], dim=-2)
    
    return seed