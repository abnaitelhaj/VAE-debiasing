"""Lightweight training helpers (loss smoothing + optional plots)."""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch


class LossHistory:
    def __init__(self, smoothing_factor=0.99):
        self.smoothing_factor = smoothing_factor
        self.loss = []
        self.smoothed = []

    def append(self, value):
        value = float(value)
        self.loss.append(value)
        if not self.smoothed:
            self.smoothed.append(value)
        else:
            s = self.smoothing_factor * self.smoothed[-1] + (1 - self.smoothing_factor) * value
            self.smoothed.append(s)

    def get(self):
        return list(self.smoothed)


class PeriodicPlotter:
    def __init__(self, sec=2, scale="semilogy"):
        self.sec = sec
        self.scale = scale
        self._last = 0.0

    def plot(self, data):
        now = time.time()
        if now - self._last < self.sec:
            return
        self._last = now
        plt.figure(figsize=(6, 3))
        if self.scale == "semilogy":
            plt.semilogy(data)
        else:
            plt.plot(data)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.show()


def plot_sample(x, y, dbvae, n=4):
    """Show a few inputs and reconstructions (faces only when available)."""
    dbvae.eval()
    with torch.inference_mode():
        y_logit, z_mean, z_logsigma, recon = dbvae(x[:n])
    x_np = x[:n].detach().cpu().numpy()
    recon_np = np.clip(recon.detach().cpu().numpy(), 0, 1)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(np.transpose(x_np[i], (1, 2, 0)))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.transpose(recon_np[i], (1, 2, 0)))
        axes[1, i].axis("off")
    axes[0, 0].set_title("input")
    axes[1, 0].set_title("recon")
    plt.tight_layout()
    plt.show()
