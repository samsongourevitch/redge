import torch
from functools import partial
from tqdm import tqdm
from experiments.sudoku.model import sampler_step
from experiments.sudoku.sudoku_utils import clamp_clues


def kl_div(logits, target_logits):
    log_probs = logits.log_softmax(dim=-1)
    return ((log_probs - target_logits) * log_probs.exp()).sum(dim=(1, 2))


def _dispatch_callbacks(callbacks, metrics):
    for cb in callbacks:
        cb(**metrics)


def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().mean().item()
    return float(value)


def optimize_variational_logits(
    init_logits,
    sampler,
    sampler_cfg,
    guidance_cfg,
    penalty_fn,
    hard=True,
    callbacks=None,
    callback_context=None,
):
    """Optimize logits with reward (soft_penalty) + KL to model logits."""
    init_detached = init_logits.detach()
    opt_logits = init_detached.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_logits], lr=guidance_cfg.lr)
    callback_context = callback_context or {}
    active_threshold = getattr(guidance_cfg, "active_threshold", 1e-3)

    for step in range(guidance_cfg.n_opt_steps):
        optimizer.zero_grad()
        x0_reparam = sampler(logits=opt_logits, hard=hard, **sampler_cfg)
        penalty = penalty_fn(x0_reparam)
        kl = kl_div(logits=opt_logits, target_logits=init_detached)
        total = penalty + guidance_cfg.kl_weight * kl
        loss = total.sum()
        loss.backward()
        grad_norm = (
            opt_logits.grad.norm().item() if opt_logits.grad is not None else 0.0
        )
        optimizer.step()

        if callbacks:
            with torch.no_grad():
                log_probs = torch.log_softmax(opt_logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(log_probs * probs).sum(dim=-1).mean().item()
                active_categories = (
                    (probs > active_threshold).float().sum(dim=-1).mean().item()
                )

            metrics = {
                "penalty": _to_scalar(penalty),
                "kl": _to_scalar(kl),
                "loss": _to_scalar(total),
                "gradient_norm": grad_norm,
                "entropy": entropy,
                "active_categories": active_categories,
                "sample": x0_reparam.detach(),
                "opt_step": step,
            }
            metrics.update(callback_context)
            _dispatch_callbacks(callbacks, metrics)

    return opt_logits.detach()


def guided_sampler(
    model, sampler, batch, tokenizer, config, penalty_fn, callbacks=None
):
    x = batch["input_ids"]
    src_mask = batch["src_mask"]
    attention_mask = batch["attention_mask"]
    x_t = x.masked_fill(~src_mask.bool(), tokenizer.mask_token_id)
    pbar = tqdm(range(model.diffusion_steps, -1, -1), desc="diffusion steps")

    for t in pbar:
        logits = model(input_ids=x_t, attention_mask=attention_mask)

        variational_logits = optimize_variational_logits(
            init_logits=logits,
            sampler=sampler,
            sampler_cfg=config.sampler,
            guidance_cfg=config.guidance,
            hard=config.hard,
            penalty_fn=partial(penalty_fn, src_mask=src_mask),
            callbacks=callbacks,
            callback_context={"t": t},
        )
        log_probs = torch.log_softmax(variational_logits, dim=-1)
        log_probs[..., tokenizer.vocab_size :] = -1e3
        _, x_0 = log_probs.max(-1)

        x_t = sampler_step(
            x_0=x_0, x_t=x_t, t=t, mask=tokenizer.mask_token_id, src_mask=src_mask
        )

    return x_t


def sudoku_sampler(sampler, clues, config, penalty_fn, callbacks=None):
    """
    Sudoku sampler without mdm prior
    """
    # clues: [B,9,9]
    logits = torch.zeros((clues.shape[0], 81, 9), requires_grad=True)
    optimizer = torch.optim.Adam(params=[logits], lr=config.guidance.lr)
    pbar = tqdm(range(config.guidance.n_opt_steps))
    active_threshold = getattr(config.guidance, "active_threshold", 1e-3)

    for step in pbar:
        with torch.no_grad():
            clamp_clues(logits, clue_vals=clues.view(clues.shape[0], -1))
        optimizer.zero_grad()
        samples = sampler(logits=logits, **config.sampler)
        loss = penalty_fn(samples)
        loss.backward()
        grad_norm = logits.grad.norm().item() if logits.grad is not None else 0.0
        optimizer.step()

        if callbacks:
            with torch.no_grad():
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(log_probs * probs).sum(dim=-1).mean().item()
                active_categories = (
                    (probs > active_threshold).float().sum(dim=-1).mean().item()
                )

            metrics = {
                "penalty": _to_scalar(loss),
                "kl": 0.0,
                "loss": _to_scalar(loss),
                "gradient_norm": grad_norm,
                "entropy": entropy,
                "active_categories": active_categories,
                "sample": samples.detach(),
                "opt_step": step,
                "t": None,
            }
            _dispatch_callbacks(callbacks, metrics)
    return samples


def build_callbacks():
    callbacks = []

    penalty_vals = []

    def record_penalty(penalty, **kwargs):
        penalty_vals.append(penalty)

    record_penalty.log = penalty_vals
    record_penalty.descr = "Penalty evolution"
    callbacks.append(record_penalty)

    kl_vals = []

    def record_kl(kl, **kwargs):
        kl_vals.append(kl)

    record_kl.log = kl_vals
    record_kl.descr = "KL regularization"
    callbacks.append(record_kl)

    loss_vals = []

    def record_loss(loss, **kwargs):
        loss_vals.append(loss)

    record_loss.log = loss_vals
    record_loss.descr = "Total loss"
    callbacks.append(record_loss)

    grad_norms = []

    def record_grad_norm(gradient_norm, **kwargs):
        grad_norms.append(gradient_norm)

    record_grad_norm.log = grad_norms
    record_grad_norm.descr = "Gradient norm"
    callbacks.append(record_grad_norm)

    entropy_vals = []

    def record_entropy(entropy, **kwargs):
        entropy_vals.append(entropy)

    record_entropy.log = entropy_vals
    record_entropy.descr = "Entropy of opt logits"
    callbacks.append(record_entropy)

    active_category_vals = []

    def record_active_categories(active_categories, **kwargs):
        active_category_vals.append(active_categories)

    record_active_categories.log = active_category_vals
    record_active_categories.descr = "Average active categories"
    callbacks.append(record_active_categories)

    return callbacks
