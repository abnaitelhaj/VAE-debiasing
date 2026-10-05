"""Project hyperparameters and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CNN_CHECKPOINT = CHECKPOINT_DIR / "standard_cnn.pt"
DBVAE_CHECKPOINT = CHECKPOINT_DIR / "dbvae.pt"

CACHE_DIR = Path.home() / ".cache" / "dbvae"
TRAIN_FACE_URL = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
CHANNELS_LAST = False

IMAGE_SIZE = 64
N_FILTERS = 12
LATENT_DIM = 100
IN_CHANNELS = 3

CNN_PARAMS = dict(
    batch_size=32,
    num_epochs=2,
    learning_rate=5e-4,
)

DBVAE_PARAMS = dict(
    batch_size=32,
    learning_rate=5e-4,
    latent_dim=100,
    num_epochs=2,
)

KL_WEIGHT = 0.0005
RESAMPLE_BINS = 10
SMOOTHING_FAC = 0.001

TRAIN_EVAL_BATCH = 5000
DEMOGRAPHIC_KEYS = ["Light Female", "Light Male", "Dark Female", "Dark Male"]

COMET_CNN_PROJECT = "dbvae-cnn"
COMET_DBVAE_PROJECT = "dbvae-train"
