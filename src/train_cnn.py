"""Standard CNN training loop."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.params import CNN_PARAMS, COMET_CNN_PROJECT
from src.experiment import create_experiment
from src.util import LossHistory, PeriodicPlotter


def train_standard_cnn(standard_classifier, loader, device, params=None, use_comet=True):
    params = params or CNN_PARAMS
    loss_fn = nn.BCEWithLogitsLoss()

    experiment = None
    if use_comet:
        experiment = create_experiment(COMET_CNN_PROJECT, params)

    optimizer = optim.Adam(
        standard_classifier.parameters(), lr=params["learning_rate"]
    )
    loss_history = LossHistory(smoothing_factor=0.99)
    plotter = PeriodicPlotter(sec=2, scale="semilogy")
    if hasattr(tqdm, "_instances"):
        tqdm._instances.clear()

    standard_classifier.train()

    def standard_train_step(x, y):
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        optimizer.zero_grad()
        logits = standard_classifier(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return loss

    step = 0
    for epoch in range(params["num_epochs"]):
        for idx in tqdm(range(loader.get_train_size() // params["batch_size"])):
            x, y = loader.get_batch(params["batch_size"])
            loss = standard_train_step(x, y)
            loss_value = loss.detach().cpu().numpy()

            loss_history.append(loss_value)
            plotter.plot(loss_history.get())

            if experiment is not None:
                experiment.log_metric("loss", loss_value, step=step)
            step += 1

    if experiment is not None:
        experiment.end()

    return standard_classifier, loss_history
