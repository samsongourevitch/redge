import torch
import numpy as np
import matplotlib.pyplot as plt

class Polyprog:
    def __init__(
        self,
        pow=2.0,
        offset=0.45,
        length=256,
        vocab_size=2,
        lr=1e-2,
        beta_entropy=0.0,
        linear=False,
        **kwargs,
    ):
        self.length = length
        self.vocab_size = vocab_size
        self.offset_vec = torch.zeros(length, vocab_size)
        self.pow = pow
        self.offset = offset
        self.offset_vec[..., 0] = -self.offset
        self.offset_vec[..., 1] = 1 - self.offset
        self.logits = torch.zeros((self.length, self.vocab_size), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.logits], lr=lr)
        self.beta = beta_entropy
        self.linear = linear

    def get_optimizer(self):
        return self.optimizer

    def get_logits(self):
        return self.logits

    def loss(self, x, **kwargs):
        if self.linear:
            offset_sq_0 = self.offset**self.pow
            offset_sq_1 = (1 - self.offset)**self.pow
            offset_val_vec = torch.tensor([offset_sq_0, offset_sq_1], device=x.device)

            loss = (x * offset_val_vec).sum(-1).sum(-1) / self.length
        else:
            x_discrete = (x * (torch.arange(self.vocab_size, device=x.device).view(1, 1, -1))).sum(-1)  # (B, D)
            # loss = (((x_discrete - self.offset).norm(p=self.pow)**self.pow).sum(-1) / self.length)
            loss = (
                torch.norm((x_discrete - self.offset), p=self.pow, dim=-1) ** self.pow / self.length
            )
        p = self.logits.softmax(-1)
        entropy = -(p * p.log()).sum(-1).sum(-1) / self.length
        return loss + self.beta * entropy

    def plot_params(self, logits):
        pi = logits.softmax(-1).detach().cpu().numpy()
        plt.plot(pi[:, np.round(self.offset).astype(int)], label=f"Learned pi[{np.round(self.offset).astype(int)}]")
        plt.legend()
        plt.xlabel("dim")
        plt.title(f"Learned pi[{np.round(self.offset).astype(int)}]")
        plt.show()