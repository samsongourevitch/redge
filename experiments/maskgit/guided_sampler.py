import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiments.maskgit.samplers import (
    mk_remasking_schedule,
    mk_mdm_schedule,
    remdm_bridge,
)
from experiments.utils import save_im
from local_paths import REPO_DIR

# def kl_div(logits, target_logits):
#     log_probs = logits.log_softmax(dim=-1)
#     return ((log_probs - target_logits) * log_probs.exp()).sum(dim=(1, 2))


def kl_div(logits, target_logits, x0_reparam, mode="exact"):
    log_q = logits.log_softmax(dim=-1)
    log_p = target_logits.log_softmax(dim=-1)
    q = log_q.exp()
    if mode == "exact":
        return (q * (log_q - log_p)).sum(dim=(1, 2))
    elif mode == "mc_sample":
        kl = ((log_q - log_p) * x0_reparam).sum(dim=-1).sum(dim=1)
        return kl
    else:
        raise ValueError(f"Unknown mode {mode} for KL divergence computation.")


def optimize_variational_logits(
    init_logits,
    xt,
    prev_logits,
    t,
    model,
    sampler,
    sampler_cfg,
    guidance_cfg,
    reward_fn,
    callbacks,
    hard=True,
):
    """Optimize logits with reward (soft_penalty) + KL to model logits."""
    init_detached = init_logits[:, :, :-1].clone().detach()  # remove mask token
    codebook = model.ae.quantize.embedding.weight
    if prev_logits is None:
        opt_logits = init_detached.clone().detach().requires_grad_(True)
    else:
        opt_logits = (
            guidance_cfg.forget_coeff * init_detached
            + (1 - guidance_cfg.forget_coeff) * prev_logits.clone().detach()
        ).requires_grad_(True)
    optimizer = torch.optim.Adam([opt_logits], lr=guidance_cfg.lr)

    for step in range(guidance_cfg.n_opt_steps):
        optimizer.zero_grad()
        x0_reparam = sampler(logits=opt_logits, **sampler_cfg)
        if not hard:
            x0_reparam = x0_reparam.soft
        embeddings = (
            (x0_reparam @ codebook)
            .view(opt_logits.size(0), model.input_size, model.input_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        decoded_x0 = model.ae.decode(embeddings)
        reward = reward_fn(decoded_x0)
        kl = kl_div(
            logits=opt_logits,
            target_logits=init_detached,
            x0_reparam=x0_reparam,
            mode=guidance_cfg.kl_sampling_mode,
        )

        if callbacks is None:
            loss = (-reward + kl).sum()
            loss.backward()
        else:
            # has to be executed in this order for the ReinDGE estimator to work with callbacks
            llk_grad = torch.autograd.grad(
                -reward.sum(), opt_logits, retain_graph=True
            )[0]
            kl_grad = torch.autograd.grad(
                guidance_cfg.kl_weight * kl.sum(), opt_logits
            )[0]

            with torch.no_grad():
                opt_logits.grad = llk_grad + kl_grad

        optimizer.step()

        if callbacks is not None:
            for cb in callbacks:
                cb(
                    llk_val=-reward.mean().item(),
                    kl=kl.mean().item(),
                    loss=(-reward + kl).mean().item(),
                    gradient_norm=opt_logits.grad.norm().item(),
                    kl_grad=kl_grad.norm().item(),
                    llk_grad=llk_grad.norm().item(),
                    sample=decoded_x0.detach().clamp(-1, 1),
                    t=t,
                    opt_step=step,
                )

        with torch.no_grad():
            opt_logits = model._carry_over_unmasking(xt, opt_logits)

    return opt_logits.detach()


def guided_remdm_sampler(
    config,
    model,
    cond,
    n_steps,
    n_samples,
    reward_fn,
    sampler,
    callbacks,
    hard=True,
    cfg_weight=0.0,
    sigma=0.0,
    schedule_type="linear",
    drop_label=False,
):

    mask_token = model.mask
    alphas = mk_mdm_schedule(n_steps, schedule_type=schedule_type)
    sigmas = mk_remasking_schedule(
        alphas=alphas, remasking_schedule_type="constant", sigma=sigma
    )
    codes = torch.full((n_samples, model.input_size**2), mask_token)
    variational_logits = None

    pbar = tqdm(range(n_steps - 1, 0, -1))
    for t in pbar:
        # pred logits with carry-over unmasking
        logits = model(
            x=codes.view(n_samples, model.input_size, -1),
            labels=cond,
            cfg_weight=cfg_weight,
            drop_label=drop_label,
        )
        variational_logits = optimize_variational_logits(
            init_logits=logits,
            xt=codes,
            prev_logits=variational_logits,
            model=model,
            sampler=sampler,
            t=t,
            sampler_cfg=config.sampler,
            guidance_cfg=config.demo.guidance,
            reward_fn=reward_fn,
            hard=hard,
            callbacks=callbacks,
        )
        pred_codes = torch.distributions.Categorical(
            logits=variational_logits
        ).sample()  # [B, L]
        codes = remdm_bridge(
            x0=pred_codes, xt=codes, t=t, model=model, alphas=alphas, sigmas=sigmas
        )

    codes = torch.clamp(codes, 0, model.codebook_size - 1)
    decoded = model.decode(codes.reshape(n_samples, model.input_size, -1))
    decoded = torch.clamp(decoded, -1, 1)
    return decoded


def build_callbacks():
    callbacks = []

    data_fidelity_vals = []

    def record_reward(llk_val, **kwargs):
        data_fidelity_vals.append(llk_val)

    record_reward.log = data_fidelity_vals
    record_reward.descr = "Evolution of data fidelity loss"
    callbacks.append(record_reward)

    kl_regularization_vals = []

    def record_kl_vals(kl, **kwargs):
        kl_regularization_vals.append(kl)

    record_kl_vals.log = kl_regularization_vals
    record_kl_vals.descr = "Evolution of kl regularization"
    callbacks.append(record_kl_vals)

    loss_val = []

    def record_loss_vals(loss, **kwargs):
        loss_val.append(loss)

    record_loss_vals.log = loss_val
    record_loss_vals.descr = "Loss evolution"
    callbacks.append(record_loss_vals)

    gradient_norms = []

    def record_gradient_norm(gradient_norm, **kwargs):
        gradient_norms.append(gradient_norm)

    record_gradient_norm.log = gradient_norms
    record_gradient_norm.descr = "Gradient norm"
    callbacks.append(record_gradient_norm)

    llk_gradient_norms = []

    def record_llk_gradient_norm(llk_grad, **kwargs):
        llk_gradient_norms.append(llk_grad)

    record_llk_gradient_norm.log = gradient_norms
    record_llk_gradient_norm.descr = "Llk gradient norm"
    callbacks.append(record_llk_gradient_norm)

    kl_gradient_norms = []

    def record_kl_gradient_norm(kl_grad, **kwargs):
        kl_gradient_norms.append(kl_grad)

    record_kl_gradient_norm.log = kl_gradient_norms
    record_kl_gradient_norm.descr = "Kl gradient norm"
    callbacks.append(record_kl_gradient_norm)

    return callbacks
