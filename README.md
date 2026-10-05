# DB-VAE — Debiasing Facial Detection

Train a face / not-face CNN, then a debiasing variational autoencoder (DB-VAE) that resamples rare latent regions during training. Evaluate both models across Light/Dark × Female/Male groups.

## Layout

```
config/params.py
src/                  # data, models, losses, resampling, train, evaluate
notebooks/
  01_data.ipynb
  02_train_cnn.ipynb      → checkpoints/standard_cnn.pt
  03_train_dbvae.ipynb    → checkpoints/dbvae.pt
  04_evaluate.ipynb       → load checkpoints, compare
figures/  papers/  reports/  checkpoints/
```

## Setup

```bash
pip install -r requirements.txt
# optional: export COMET_API_KEY=... and set USE_COMET = True in notebooks 02 / 03
```

## Run order

1. `01_data.ipynb` — download + inspect
2. `02_train_cnn.ipynb` — train CNN, save checkpoint
3. `03_train_dbvae.ipynb` — train DB-VAE, save checkpoint
4. `04_evaluate.ipynb` — load both, plot comparison
