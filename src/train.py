import torch
import torch.nn.functional as F


def load_checkpoint(fpath, model, optimizer):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def print_loss_metrics(train_loss, val_loss):
    print(f"Trn:   {train_loss['loss']:.4f}\tMSE: {train_loss['mse']:.4f}\tKLD: {train_loss['kld']:.4f}")
    print(f"Val:   {val_loss['loss']:.4f}\tMSE: {val_loss['mse']:.4f}\tKLD: {val_loss['kld']:.4f}\n")


def save_checkpoint(model, optimizer, fpath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, fpath)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = {key: 0.0 for key in ['loss', 'mse', 'kld']}

    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        y, mu, log_var = model(x)
        loss = vae_loss(x, y, mu, log_var)
        loss['loss'].backward()
        optimizer.step()

        for key in loss.keys():
            train_loss[key] += loss[key].item()

    # Report average loss.
    for key in loss.keys():
        train_loss[key] /= len(train_loader.dataset)

    return train_loss


def vae_loss(x, y, mu, log_var, kld_weight=1.0):

    reconstruction_loss = F.mse_loss(x, y)
    kullback_liebler_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )

    loss = reconstruction_loss + kld_weight * kullback_liebler_loss
    return {'loss': loss, 'mse': reconstruction_loss, 'kld': kullback_liebler_loss}


def validate(model, val_loader, device):
    model.eval()
    val_loss = {key: 0.0 for key in ['loss', 'mse', 'kld']}

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            y, mu, log_var = model(x)
            loss = vae_loss(x, y, mu, log_var)

            for key in loss.keys():
                val_loss[key] += loss[key].item()

        for key in loss.keys():
                val_loss[key] /= len(val_loader.dataset)

    return val_loss
