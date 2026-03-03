import torch
import torch.nn as nn
from samplers.ddim import redge
from samplers.gumbel_sampling import gumbel_softmax
from samplers.reinmax import reinmax
from samplers.st import straight_through


class BernoulliVAEGradient(nn.Module):
    def __init__(self, input_dim, cat_dim, latent_dim, hidden_dim=400, gradient_method="gumbel", config=None):
        super().__init__()
        # Encoder
        # self.fc_enc = nn.Linear(input_dim, latent_dim*cat_dim)
        # self.fc_enc = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, latent_dim * cat_dim)
        # )
        self.fc_enc = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * cat_dim))

        # Decoder
        # self.fc_dec = nn.Linear(latent_dim*cat_dim, input_dim)
        # self.fc_dec = nn.Sequential(
        #     nn.Linear(latent_dim * cat_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, input_dim)
        # )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim * cat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        self.gradient_method = gradient_method
        self.config = config if config is not None else {}

        self.D = latent_dim
        self.K = cat_dim

    def encode(self, x):
        logits = self.fc_enc(x)
        return logits  

    def decode(self, z):
        return self.fc_dec(z)

    def forward(self, x):
        logits = self.encode(x).view(-1, self.D, self.K)
        probs = logits.softmax(-1)

        if self.gradient_method == "gumbel":
            z_onehot = gumbel_softmax(
                logits=logits, tau=self.config["gumbel"]["tau"], hard=True, batch_size=1
            )
        elif self.gradient_method == "reinmax":
            z_onehot = reinmax(
                logits, tau=self.config["reinmax"]["tau"], hard=True, batch_size=1
            )
        elif self.gradient_method == "st":
            z_onehot = straight_through(
                logits, batch_size=1, hard=True
            )
            
        elif self.gradient_method == "diffusion":
            z_onehot = redge(
                logits=logits,
                n_steps=self.config["diffusion"]["T"],
                batch_size=1,
                grad_cutoff=self.config["diffusion"]["grad_cutoff"],
                hard=True,
                schedule=self.config["diffusion"]["schedule_kwargs"],
            )

        x_recon = self.decode(z_onehot.view(-1, self.D * self.K))
        return x_recon, probs, z_onehot
    
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, cat_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 14x14 → 7x7
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim * cat_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)   
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        return self.fc(h)             # logits for discrete latent


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, cat_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * cat_dim, 64 * 7 * 7),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 14 → 28
        )

    def forward(self, z_flat):
        h = self.fc(z_flat)
        h = h.view(-1, 64, 7, 7)
        x_recon = self.deconv(h)
        return x_recon.view(-1, 28*28)


class BernoulliVAE(nn.Module):
    def __init__(self, input_dim, cat_dim, latent_dim):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, cat_dim)
        self.decoder = ConvDecoder(latent_dim, cat_dim)
        self.latent_dim = latent_dim
        self.cat_dim = cat_dim

    def encode(self, x):
        logits = self.encoder(x)
        return logits

    def decode(self, z):
        return self.decoder(z)
    
class BernoulliVAESimple(nn.Module):
    def __init__(self, input_dim, cat_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim * cat_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * cat_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, input_dim)
        )
        self.latent_dim = latent_dim
        self.cat_dim = cat_dim

    def encode(self, x):
        logits = self.encoder(x)
        return logits

    def decode(self, z):
        return self.decoder(z)
