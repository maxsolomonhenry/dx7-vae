import numpy as np
import torch
import torch.nn.functional as F


def collect_embeddings(model, dataset, device, model_type):
    # Return all embeddings as a 2D numpy matrix.
    model.eval()

    embeddings = []

    with torch.no_grad():
        for x in dataset:
            x = x.to(device)

            if model_type == 'vae':
                # Take means as embeddings.
                y, z, _ = model(x)
            elif model_type == 'ae':
                y, z = model(x)
            elif model_type == 'mixed_ae':
                y, z = model(x)

            embeddings.append(z)

    embeddings = [z.cpu().numpy() for z in embeddings]
    embeddings = np.vstack(embeddings)

    return embeddings


def load_checkpoint(fpath, model, optimizer):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def mixed_loss(x, y):

    # TODO: Update this to not be hard-coded.
    DECODER_SPEC = [
        123, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 32, 2, 2, 8
    ]

    idx = np.array([0] + DECODER_SPEC).cumsum()[1:]

    # Add loss for continuous data.
    mse_loss = F.mse_loss(x[:, :idx[0]], y[:, :idx[0]])

    # Add categorical loss.
    cross_entropy_loss = 0
    for i in range(0, len(idx) - 1):
        idx_in = idx[i]
        idx_out = idx[i + 1]

        cross_entropy_loss += F.cross_entropy(
            y[:, idx_in:idx_out],
            x[:, idx_in:idx_out]
        )

    loss = mse_loss + cross_entropy_loss
    info = {'mse_loss': mse_loss, 'cross_entropy_loss': cross_entropy_loss}

    return loss, info

def print_loss_metrics(train_loss, val_loss):
    print(f"Trn: {train_loss:.8f}\tVal: {val_loss:.8f}\n")


def save_checkpoint(model, optimizer, fpath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, fpath)


def train_one_epoch(model, train_loader, optimizer, device, model_type):
    model.train()
    train_loss = 0.0

    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        if model_type == 'vae':
            y, mu, log_var = model(x)
            loss, _ = vae_loss(x, y, mu, log_var)
        elif model_type == 'ae':
            y, z = model(x)
            loss = F.mse_loss(x, y)
        elif model_type == 'mixed_ae':
            y, z = model(x)
            loss, _ = mixed_loss(x, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Report average loss.
    train_loss /= len(train_loader.dataset)

    return train_loss


def vae_loss(x, y, mu, log_var, kld_weight=0.001):

    reconstruction_loss = F.mse_loss(x, y)
    kullback_liebler_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )

    loss = reconstruction_loss + kld_weight * kullback_liebler_loss
    return loss, {'mse': reconstruction_loss, 'kld': kullback_liebler_loss}


def validate(model, val_loader, device, model_type):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            if model_type == 'vae':
                y, mu, log_var = model(x)
                loss, stats = vae_loss(x, y, mu, log_var)
            elif model_type == 'ae':
                y, z = model(x)
                loss = F.mse_loss(x, y)
            elif model_type == 'mixed_ae':
                y, z = model(x)
                loss, _ = mixed_loss(x, y)
            val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
    return val_loss
