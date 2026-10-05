"""VAE / DB-VAE losses and reparameterization."""

import torch
import torch.nn.functional as F

from config.params import KL_WEIGHT


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=KL_WEIGHT):
    # Define the latent loss using the provided equation for L_{KL}
    # L_{KL} = 0.5 * sum(sigma + mu^2 - 1 - log(sigma))
    # Note: logsigma is passed in, so sigma = exp(logsigma)
    latent_loss = 0.5 * torch.sum(
        torch.exp(logsigma) + mu**2 - 1.0 - logsigma, dim=-1
    )

    # Define the reconstruction loss as the mean absolute pixel-wise
    # difference between the input and reconstruction.
    # We average over channels, height, and width.
    reconstruction_loss = torch.mean(torch.abs(x - x_recon), dim=(1, 2, 3))

    # Define the VAE loss as L_{VAE} = c * L_{KL} + L_{x}
    # We average the batch-wise loss
    vae_loss = torch.mean(kl_weight * latent_loss + reconstruction_loss)

    return vae_loss


def sampling(z_mean, z_logsigma):
    # Generate random noise with the same shape as z_mean, sampled from a standard normal distribution (mean=0, std=1)
    eps = torch.randn_like(z_mean)

    # Define the reparameterization computation!
    # z = mu + exp(0.5 * logsigma) * eps
    z = z_mean + torch.exp(0.5 * z_logsigma) * eps

    return z


def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    # Call the relevant function to obtain VAE loss
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma)

    # Define the classification loss using binary_cross_entropy_with_logits
    classification_loss = F.binary_cross_entropy_with_logits(
        y_logit, y, reduction="none"
    )

    # Use the training data labels to create variable face_indicator:
    #   indicator that reflects which training data are images of faces
    y = y.float()
    face_indicator = (y == 1.0).float()

    # Define the DB-VAE total loss! Use torch.mean to average over all samples
    # L_total = L_y + I_f * L_VAE
    total_loss = torch.mean(classification_loss + face_indicator * vae_loss)

    return total_loss, torch.mean(classification_loss)
