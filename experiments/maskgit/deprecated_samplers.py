import torch 
from maskgit.samplers import mk_mdm_schedule
from tqdm import tqdm 

def mdlm_sampler(model,
                cond,
                n_steps,
                n_samples,
                cfg_weight=0.,
                schedule_type='linear'):
    """
    deprecated
    """
    mask_token = model.mask 
    alphas = mk_mdm_schedule(n_steps, schedule_type=schedule_type)
    codes = torch.full((n_samples, model.input_size ** 2), mask_token)
    mask = (codes == mask_token).view(n_samples, -1)

    pbar = tqdm(range(n_steps - 1, 0, -1))
    for t in pbar: 
        # pred logits with carry-over unmasking
        logits = model(x=codes.view(n_samples, model.input_size, -1), 
                       labels=cond, 
                       cfg_weight=cfg_weight)
        pred_codes = torch.distributions.Categorical(logits=logits).sample() # [B, L]

        prob_mask = (1 - alphas[t-1]) / (1 - alphas[t])
        mask = (torch.rand(*codes.shape) < prob_mask)  
        codes = torch.where(mask, codes, pred_codes)

    codes = torch.clamp(codes, 0, model.codebook_size - 1)
    decoded = model.decode(codes.reshape(n_samples, model.input_size, -1))
    decoded = torch.clamp(decoded, -1, 1)
    return  decoded  

def mdlm_sampler(model,
                cond,
                n_steps,
                n_samples,
                cfg_weight=0.,
                schedule_type='linear'):
    """
    deprecated, scatter version
    """
    mask_token = model.mask 
    alphas = mk_mdm_schedule(n_steps, schedule_type=schedule_type)
    codes = torch.full((n_samples, model.input_size ** 2), mask_token)

    pbar = tqdm(range(n_steps - 1, 0, -1))
    for t in pbar: 
        # pred logits with carry-over unmasking
        logits = model(x=codes.view(n_samples, model.input_size, -1), 
                       labels=cond, 
                       cfg_weight=cfg_weight)
        probs = (alphas[t-1] - alphas[t]) * logits.softmax(dim=-1) / (1 - alphas[t])
        src = (1 - alphas[t-1]) / (1 - alphas[t]) * torch.ones_like(probs)
        probs.scatter_add_(dim=-1, index=codes.unsqueeze(-1), src=src)
        codes = torch.distributions.Categorical(probs).sample()

    codes = torch.clamp(codes, 0, model.codebook_size - 1)
    decoded = model.decode(codes.reshape(n_samples, model.input_size, -1))
    decoded = torch.clamp(decoded, -1, 1)
    return  decoded  