"""Adaptive latent-space resampling for DB-VAE."""

import numpy as np
import torch

from config.params import LATENT_DIM, RESAMPLE_BINS, SMOOTHING_FAC


def get_latent_mu(images, dbvae, device, batch_size=64):
    dbvae.eval()
    all_z_mean = []

    # If images is NumPy, convert once outside the loop
    images_t = torch.from_numpy(images).float()

    with torch.inference_mode():
        for start in range(0, len(images_t), batch_size):
            end = start + batch_size
            batch = images_t[start:end]
            batch = batch.to(device).permute(0, 3, 1, 2)
            # Forward pass on this chunk only
            _, z_mean, _, _ = dbvae(batch)
            all_z_mean.append(z_mean.cpu())

    # Concatenate all partial z_mean
    z_mean_full = torch.cat(all_z_mean, dim=0)  # shape [N, latent_dim]
    mu = z_mean_full.numpy()  # convert to NumPy if needed
    return mu


def get_training_sample_probabilities(
    images,
    dbvae,
    device,
    latent_dim=LATENT_DIM,
    bins=RESAMPLE_BINS,
    smoothing_fac=SMOOTHING_FAC,
):
    """Function that recomputes the sampling probabilities for images within a batch
    based on how they distribute across the training data"""
    print("Recomputing the sampling probabilities")

    # Run the input batch and get the latent variable means
    mu = get_latent_mu(images, dbvae, device)

    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0], dtype=np.float64)

    # consider the distribution for each latent variable
    for i in range(latent_dim):
        latent_distribution = mu[:, i]
        # generate a histogram of the latent distribution
        hist_density, bin_edges = np.histogram(
            latent_distribution, density=True, bins=bins
        )

        # find which latent bin every data sample falls in
        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")

        # Call the digitize function to find which bins in the latent distribution
        # every data sample falls in to
        bin_idx = np.digitize(latent_distribution, bin_edges)

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # invert the density function
        p = 1.0 / (hist_smoothed_density[bin_idx - 1])

        # Normalize all probabilities
        p = p / np.sum(p)

        # Update sampling probabilities by considering whether the newly
        # computed p is greater than the existing sampling probabilities.
        training_sample_p = np.maximum(training_sample_p, p)

    # final normalization
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p
