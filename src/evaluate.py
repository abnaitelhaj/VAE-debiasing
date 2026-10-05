"""Train-set and demographic evaluation / plots."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from config.params import TRAIN_EVAL_BATCH, DEMOGRAPHIC_KEYS, CHANNELS_LAST
from src.data import get_test_faces


def eval_standard_on_train(standard_classifier, loader, device, n=TRAIN_EVAL_BATCH):
    standard_classifier.eval()
    (batch_x, batch_y) = loader.get_batch(n)
    batch_x = torch.from_numpy(batch_x).float().to(device)
    batch_y = torch.from_numpy(batch_y).float().to(device)

    with torch.inference_mode():
        y_pred_logits = standard_classifier(batch_x)
        y_pred_standard = torch.round(torch.sigmoid(y_pred_logits))
        acc_standard = torch.mean((batch_y == y_pred_standard).float())

    print(
        "Standard CNN accuracy on (potentially biased) training set: {:.4f}".format(
            acc_standard.item()
        )
    )
    return acc_standard.item()


def load_and_plot_test_faces(channels_last=CHANNELS_LAST, keys=None):
    keys = keys or DEMOGRAPHIC_KEYS
    test_faces = get_test_faces(channels_last=channels_last)

    fig, axs = plt.subplots(1, len(keys), figsize=(7.5, 7.5))
    for i, (group, key) in enumerate(zip(test_faces, keys)):
        axs[i].imshow(np.hstack(group).transpose(1, 2, 0))
        axs[i].set_title(key, fontsize=15)
        axs[i].axis("off")
    plt.show()
    return test_faces, keys


def eval_standard_on_demographics(standard_classifier, test_faces, keys, device):
    standard_classifier_probs_list = []

    with torch.inference_mode():
        for x in test_faces:
            x = torch.from_numpy(np.array(x, dtype=np.float32)).to(device)
            logits = standard_classifier(x)
            probs = torch.sigmoid(logits)
            probs = torch.squeeze(probs, dim=-1)
            standard_classifier_probs_list.append(probs.cpu().numpy())

    standard_classifier_probs = np.stack(standard_classifier_probs_list, axis=0)

    xx = range(len(keys))
    yy = standard_classifier_probs.mean(axis=1)
    plt.bar(xx, yy)
    plt.xticks(xx, keys)
    plt.ylim(max(0, yy.min() - np.ptp(yy) / 2.0), yy.max() + np.ptp(yy) / 2.0)
    plt.title("Standard classifier predictions")
    plt.show()

    return standard_classifier_probs


def eval_dbvae_on_demographics(dbvae, test_faces, keys, device, standard_classifier_probs):
    dbvae.to(device)
    dbvae_logits_list = []
    for face in test_faces:
        face = np.asarray(face, dtype=np.float32)
        face = torch.from_numpy(face).to(device)

        with torch.inference_mode():
            logit = dbvae.predict(face)

        dbvae_logits_list.append(logit.detach().cpu().numpy())

    dbvae_logits_array = np.concatenate(dbvae_logits_list, axis=0)
    dbvae_logits_tensor = torch.from_numpy(dbvae_logits_array)
    dbvae_probs_tensor = torch.sigmoid(dbvae_logits_tensor)
    dbvae_probs_array = dbvae_probs_tensor.squeeze(dim=-1).numpy()

    xx = np.arange(len(keys))
    std_probs_mean = standard_classifier_probs.mean(axis=1)
    dbvae_probs_mean = dbvae_probs_array.reshape(len(keys), -1).mean(axis=1)

    plt.bar(xx, std_probs_mean, width=0.2, label="Standard CNN")
    plt.bar(xx + 0.2, dbvae_probs_mean, width=0.2, label="DB-VAE")
    plt.xticks(xx, keys)
    plt.title("Network predictions on test dataset")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

    return dbvae_probs_array


def confidence_vs_accuracy(standard_classifier_probs, dbvae_probs_array, keys):
    db_probs = dbvae_probs_array.reshape(len(keys), -1)

    std_mean = standard_classifier_probs.mean(axis=1)
    db_mean = db_probs.mean(axis=1)

    std_acc = (standard_classifier_probs >= 0.5).mean(axis=1)
    db_acc = (db_probs >= 0.5).mean(axis=1)

    x = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, std_mean, width, label="Standard CNN")
    axes[0].bar(x + width / 2, db_mean, width, label="DB-VAE")
    axes[0].set_title("Average face confidence")
    axes[0].set_ylabel("Probability of face")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].bar(x - width / 2, std_acc, width, label="Standard CNN")
    axes[1].bar(x + width / 2, db_acc, width, label="DB-VAE")
    axes[1].set_title("Face-detection accuracy")
    axes[1].set_ylabel("Fraction classified as face")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=20)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("Group                 Standard accuracy   DB-VAE accuracy")
    for i, key in enumerate(keys):
        print(f"{key:20s} {std_acc[i]:.3f}              {db_acc[i]:.3f}")

    print("\nStandard gap:", std_acc.max() - std_acc.min())
    print("DB-VAE gap:", db_acc.max() - db_acc.min())

    return std_acc, db_acc
