import torch
import torch.nn.functional as F


def load_checkpoint(fpath, model, optimizer):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


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
            y = model(x)
            loss = F.mse_loss(x, y)

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
                y = model(x)
                loss = F.mse_loss(x, y)
            val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
    return val_loss