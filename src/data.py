"""Download train_face.h5 and wrap the face / not-face loader + demographic test faces."""

from pathlib import Path

import torch
import mitdeeplearning as mdl

from config.params import CACHE_DIR, TRAIN_FACE_URL, CHANNELS_LAST


def get_training_data_path(cache_dir=None):
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir.joinpath("train_face.h5")


def download_training_data(path_to_training_data=None):
    path_to_training_data = path_to_training_data or get_training_data_path()
    if path_to_training_data.is_file():
        print(f"Using cached training data from {path_to_training_data}")
    else:
        print(f"Downloading training data to {path_to_training_data}")
        torch.hub.download_url_to_file(TRAIN_FACE_URL, path_to_training_data)
    return path_to_training_data


def make_loader(path_to_training_data=None, channels_last=CHANNELS_LAST):
    path_to_training_data = download_training_data(path_to_training_data)
    loader = mdl.lab2.TrainingDatasetLoader(
        path_to_training_data, channels_last=channels_last
    )
    return loader


def get_test_faces(channels_last=CHANNELS_LAST):
    return mdl.lab2.get_test_faces(channels_last=channels_last)
