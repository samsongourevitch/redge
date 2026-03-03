import torch 
from tqdm import tqdm

def mk_mdm_schedule(n_steps, schedule_type='linear'):
    if schedule_type == 'linear':
        schedule = torch.linspace(1.0, 1e-4, n_steps)
    elif schedule_type == 'cosine':
        pass 
    return schedule

def mk_remasking_schedule(alphas, remasking_schedule_type='constant', sigma=0.):
    if remasking_schedule_type == 'constant': 
        sigmas = sigma * torch.ones_like(alphas)

    elif remasking_schedule_type == 'remdm':
        pass 
    return sigmas 

def remdm_bridge(x0, xt, t, model, alphas, sigmas):
    mask_token = model.mask 
    prob_codes = (1 - sigmas[t]) * (1 - alphas[t-1]) / (1 - alphas[t])
    prob_mask = sigmas[t] * (1 - alphas[t-1])
    
    prob_remasked = prob_codes + prob_mask
    mask = torch.rand(*xt.shape) < prob_mask / prob_remasked
    xt.masked_fill_(mask, mask_token)

    remask = torch.rand(*xt.shape) < prob_remasked
    x0 = torch.where(remask, xt, x0)
    return x0

def remdm_sampler(model,
                cond,
                n_steps,
                n_samples,
                cfg_weight=0.,
                sigma=0., 
                schedule_type='linear'):
     
    alphas = mk_mdm_schedule(n_steps, schedule_type=schedule_type)
    sigmas = mk_remasking_schedule(alphas=alphas,
                                   remasking_schedule_type='constant',
                                   sigma=sigma)
    codes = torch.full((n_samples, model.input_size ** 2), model.mask)

    pbar = tqdm(range(n_steps - 1, 0, -1))
    for t in pbar: 
        # pred logits with carry-over unmasking
        logits = model(x=codes.view(n_samples, model.input_size, -1), 
                       labels=cond, 
                       cfg_weight=cfg_weight)
        pred_codes = torch.distributions.Categorical(logits=logits).sample() # [B, L]
        codes = remdm_bridge(x0=pred_codes, 
                             xt=codes, 
                             t=t, 
                             model=model, 
                             alphas=alphas, 
                             sigmas=sigmas)

    codes = torch.clamp(codes, 0, model.codebook_size - 1)
    decoded = model.decode(codes.reshape(n_samples, model.input_size, -1))
    decoded = torch.clamp(decoded, -1, 1)
    return  decoded     