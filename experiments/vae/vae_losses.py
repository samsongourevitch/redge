import torch
import torch.nn.functional as F

def vae_loss(x_binary, x_recon, p):
    if x_recon.dim() == 2:
        x_recon = x_recon.unsqueeze(0)
    S, B, D = x_recon.shape  # S = sampler_batch_size, B = neural_net_batch_size, D = 784
    x_binary_expanded = x_binary.unsqueeze(0).expand(S, B, D)
    recon_loss = F.binary_cross_entropy_with_logits(input=x_recon, target=x_binary_expanded, reduction='none').sum(dim=-1).mean(dim=-1)  # (S,)
    K = p.size(-1)
    kl_per_latent = (p * torch.log(p*K + 1e-20)).sum(dim=-1)
    kl = kl_per_latent.sum(dim=-1).mean()  # scalar
    return (recon_loss + kl) # (S,)