import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    """ 
    Variational autoencoder class, as per _Auto-Encoding Variational Bayes_. 
        https://arxiv.org/pdf/1312.6114.pdf

    Code written with reference to:
        https://github.com/AntixK/PyTorch-VAE/

    """

    def __init__(self, n_features, n_hidden, n_latent):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent),
            nn.ReLU(),
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )

        self.fc_mu = nn.Linear(n_latent, n_latent)
        self.fc_var = nn.Linear(n_latent, n_latent)

    def encode(self, x):
        x = self.encoder(x)

        # From latent embedding, generate mean and standard deviation.
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, log_var):
        # Render as a random sample from Gaussian distribution.
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)

        # Latent representation.
        z = self.reparameterize(mu, log_var)

        y = self.decode(z)
        return [y, mu, log_var]
