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

    def __init__(self, *dimensions):
        super().__init__()

        n_latent = dimensions[-1]

        encoder_layers = []
        decoder_layers = []
        for i in range(len(dimensions) - 1):
            j = len(dimensions) - (i + 1)
            
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            decoder_layers.append(nn.Linear(dimensions[j], dimensions[j - 1]))

            if i >= len(dimensions) - 2:
                continue

            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.ReLU())
    
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

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


class AE(nn.Module):
    """ 
    Autoencoder class, following _An Empirical Comparison Between Autoencoders_
        https://arxiv.org/pdf/2103.04874.pdf
    """

    def __init__(self, *dimensions):
        super().__init__()

        encoder_layers = []
        decoder_layers = []
        for i in range(len(dimensions) - 1):
            j = len(dimensions) - (i + 1)
            
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            decoder_layers.append(nn.Linear(dimensions[j], dimensions[j - 1]))

            if i >= len(dimensions) - 2:
                continue

            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.ReLU())
    
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y, z


class MixedAE(nn.Module):
    """ 
    Autoencoder class allowing for mixed continuous and categorical data.
    """

    def __init__(self, decoder_spec, *dimensions):
        super().__init__()

        encoder_layers = []
        decoder_layers = []
        for i in range(len(dimensions) - 1):
            j = len(dimensions) - (i + 1)
            
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            decoder_layers.append(nn.Linear(dimensions[j], dimensions[j - 1]))

            if i >= len(dimensions) - 2:
                continue

            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder_body = nn.Sequential(*decoder_layers[:-1])

        n_prefeatures = self.decoder_body[-2].out_features
        
        decoder_heads = [nn.Linear(n_prefeatures, decoder_spec[0])]
        for n_predict in decoder_spec[1:]:
            decoder_heads.append(
                nn.Sequential(
                    nn.Linear(n_prefeatures, n_predict), nn.Softmax()
                )
            )

        self.decoder_heads = nn.ModuleList(decoder_heads)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        z = self.decoder_body(z)

        y = []
        for head in self.decoder_heads:
            y.append(head(z))

        return torch.cat(y, dim=1)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y, z