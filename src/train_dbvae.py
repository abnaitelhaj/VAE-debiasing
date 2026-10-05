"""DB-VAE training loop with adaptive resampling."""

import IPython
import torch
import torch.optim as optim
from tqdm import tqdm

from config.params import DBVAE_PARAMS, COMET_DBVAE_PROJECT
from src.experiment import create_experiment
from src.losses import debiasing_loss_function
from src.models import DB_VAE
from src.resampling import get_training_sample_probabilities
from src.util import plot_sample


def train_dbvae(loader, device, params=None, use_comet=True):
    params = params or DBVAE_PARAMS

    experiment = None
    if use_comet:
        experiment = create_experiment(COMET_DBVAE_PROJECT, params)

    dbvae = DB_VAE(params["latent_dim"], device=device).to(device)
    optimizer = optim.Adam(dbvae.parameters(), lr=params["learning_rate"])

    def debiasing_train_step(x, y):
        optimizer.zero_grad()
        y_logit, z_mean, z_logsigma, x_recon = dbvae(x)
        loss, class_loss = debiasing_loss_function(
            x, x_recon, y, y_logit, z_mean, z_logsigma
        )
        loss.backward()
        optimizer.step()
        return loss

    all_faces = loader.get_all_train_faces()

    step = 0
    for i in range(params["num_epochs"]):
        IPython.display.clear_output(wait=True)
        print("Starting epoch {}/{}".format(i + 1, params["num_epochs"]))

        p_faces = get_training_sample_probabilities(all_faces, dbvae, device)

        for j in tqdm(range(loader.get_train_size() // params["batch_size"])):
            (x, y) = loader.get_batch(params["batch_size"], p_pos=p_faces)
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)

            loss = debiasing_train_step(x, y)
            loss_value = loss.detach().cpu().numpy()
            if experiment is not None:
                experiment.log_metric("loss", loss_value, step=step)

            if j % 500 == 0:
                plot_sample(x, y, dbvae)

            step += 1

    if experiment is not None:
        experiment.end()

    return dbvae
