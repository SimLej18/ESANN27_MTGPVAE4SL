# MTGP-VAE for Sign Language Analysis

Code of *Multi-Task Gaussian Process VAE for Sign Language Analysis* (ESANN 2027).

## Main results

*(To be written.)*

<!-- Table 1: results/*.json (`just table`). Figures 1-3: figures/*.svg / *.pdf (`just figures`). -->

## Reproducing the results

Every result comes from one `just` command (`just` lists them all). The trained models are in `checkpoints/`, so
Table 1 and the figures can be reproduced without training, on CPU.

### Install

Requires Python 3.14, [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just install             # CPU
just install gpu         # NVIDIA GPU (CUDA 12)
```

### Data

The experiments use LSFB100: the 100 most frequent signs of
[LSFB-ISOL](https://huggingface.co/datasets/ppoitier/lsfb-isol), clips of 8 to 48 frames.
We do not redistribute the poses: `data/lsfb100.csv` lists the clips (split, id, sign, mirror flag) and
`data/signs.txt` the sign order; a script builds the dataset from the HuggingFace release.

```bash
just pull-isol   # LSFB-ISOL poses, ~3 GB, into ~/datasets/lsfb/lsfb-isol (pinned revision)
just data        # data/lsfb100.npz (~370 MB, ~1 min)
just stats       # clips, frames and signers of LSFB100, per split
```

`pull-isol` and `data` take the dataset folder as an argument, e.g. `just pull-isol /data/lsfb-isol` then
`just data /data/lsfb-isol`.

### Table 1 (classification)

```bash
just table                                      # the four evaluations, then the table (results/*.json, ~10 min)
just evaluate checkpoints/mtgp_vae              # or one model at a time
just evaluate checkpoints/sign_gp_vae --posthoc # under a Magma prior fitted on its frozen codes
```

The post-hoc prior of the Sign-GP-VAE is provided (`checkpoints/sign_gp_vae_posthoc.eqx`); without it, `--posthoc`
fits it again (~1 h on a GPU).

### Figures

```bash
just figures   # figures 1-3 of the paper -> figures/*.svg (+ *.pdf with rsvg-convert)
```

### Training

The three models start from the pre-trained VAE (`checkpoints/vae_pretrained`), 25k steps each: Sign-VAE, Sign-GP-VAE
and MTGP-VAE (the same MTGP prior, restricted for the first two, see `mtgpvae/prior.py`). The MTGP-VAE is then
fine-tuned with a tempered classification loss.

```bash
just train-all                 # everything, one run after the other (~30 h on one GPU)
just train sign_vae            # or one model at a time: sign_vae, sign_gp_vae, mtgp_vae
just finetune                  # MTGP-VAE fine-tuning, from checkpoints/mtgp_vae_base
```

### Pre-training the VAE (optional)

The pre-trained VAE is provided (`checkpoints/vae_pretrained`). To pre-train it again, on the unannotated recordings of
[LSFB-CONT](https://huggingface.co/datasets/ppoitier/lsfb-cont) (except those LSFB100 validation clips come from):

```bash
just pull-cont      # LSFB-CONT poses, ~20 GB, into ~/datasets/lsfb/lsfb-cont (pinned revision)
just data-cont      # data/cont_frames.npy (~8 GB); needs data/lsfb100.npz (`just data`)
just pretrain-vae   # 1M steps, ~30 min on a GPU -> checkpoints/vae_pretrained
```

### Repository structure

```
mtgpvae/        the package
  pose.py         skeleton layout (137 MediaPipe landmarks, x / y), left-right mirror
  data.py         LSFB100 clips and LSFB-CONT frames, normalisation, time grid
  vae.py          frame-level body-part VAE
  prior.py        MTGP prior per latent dim; Sign-VAE and Sign-GP-VAE as special cases
  model.py        posterior and training loss (reconstruction, KL, classification, interpolation)
  inference.py    codes, sign posteriors, classification, post-hoc Magma fit, generation
scripts/        one script per command of the justfile
data/           LSFB100 clip list and sign order (the built datasets are not versioned)
checkpoints/    pre-trained VAE, the three trained models, the post-hoc prior
results/        Table 1, one json per model
figures/        figures 1-3 of the paper
```

## License

MIT (see `LICENSE`).
