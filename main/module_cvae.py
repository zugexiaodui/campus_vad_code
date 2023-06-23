import torch
from torch import Tensor
import torch.nn as nn


def idx2onehot(idx: torch.Tensor, n: int):
    assert torch.max(idx).item() < n, f"{idx}, {torch.max(idx).item()}, {n}"

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


class CVAE(nn.Module):

    def __init__(self, inp_size, inter_size, latent_size,
                 conditional=False, num_labels=0):
        super().__init__()

        if conditional:
            assert num_labels > 0

        self.latent_size = latent_size

        self.encoder = Encoder(inp_size, inter_size, latent_size, conditional, num_labels)
        self.decoder = Decoder(inp_size, inter_size, latent_size, conditional, num_labels)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        means: Tensor
        log_var: Tensor
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):
    def __init__(self, in_size, inter_size, latent_size, conditional, num_labels):
        super().__init__()

        self.conditional = conditional
        if self.conditional:
            in_size += num_labels
        self.num_labels = num_labels

        self.MLP = nn.Sequential(
            nn.LayerNorm(in_size),
            nn.Linear(in_size, inter_size),
            nn.GELU(),
            nn.LayerNorm(inter_size),
            nn.Linear(inter_size, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

        self.linear_means = nn.Linear(32, latent_size)
        self.linear_log_var = nn.Linear(32, latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, self.num_labels)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, out_size, inter_size, latent_size, conditional, num_labels):
        super().__init__()

        self.num_labels = num_labels

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.MLP = nn.Sequential(
            nn.Linear(input_size, inter_size),
            nn.GELU(),
            nn.LayerNorm(inter_size),
            nn.Linear(inter_size, out_size),
        )

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
