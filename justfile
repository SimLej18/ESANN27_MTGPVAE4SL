# Use the repo's own environment, whatever UV_PROJECT_ENVIRONMENT says
export UV_PROJECT_ENVIRONMENT := ".venv"
export VIRTUAL_ENV := justfile_directory() / ".venv"

# List recipes
default:
    @just --list

# Install the environment: `just install` (CPU) or `just install gpu` (NVIDIA, CUDA 12)
install device="cpu":
    uv sync {{ if device == "gpu" { "--extra cuda" } else { "" } }}

# Download LSFB-ISOL poses (shards of the 500 most frequent signs, no videos) from HuggingFace, pinned revision
pull-isol dest="~/datasets/lsfb/lsfb-isol":
    uvx --from huggingface_hub hf download ppoitier/lsfb-isol --repo-type dataset --revision 947ee2209969d2fc12fa8f95503fe32906a64941 --local-dir {{dest}} --include "shards/500/*"

# Build data/lsfb100.npz from the local LSFB-ISOL copy
data root="~/datasets/lsfb/lsfb-isol":
    uv run python scripts/make_lsfb100.py --root {{root}}

# Download LSFB-CONT poses (unannotated shards, ~20 GB, no videos), pinned revision. Only needed to pre-train the VAE
pull-cont dest="~/datasets/lsfb/lsfb-cont":
    uvx --from huggingface_hub hf download ppoitier/lsfb-cont --repo-type dataset --revision 64b0c112f59d3710367fb736ef1808f37ff15e28 --local-dir {{dest}} --include "shards/unannotated/*"

# Build data/cont_frames.npy (VAE pre-training frames, ~8 GB) from the local LSFB-CONT copy
data-cont root="~/datasets/lsfb/lsfb-cont":
    uv run python scripts/make_cont.py --root {{root}}

# Pre-train the body-part VAE on LSFB-CONT (1M steps, ~30 min on a GPU) -> checkpoints/vae_pretrained
pretrain-vae:
    uv run python scripts/pretrain_vae.py

# Train one model from the pre-trained VAE: sign_vae, sign_gp_vae or mtgp_vae (25k steps, GPU) -> checkpoints/<name>
train model name=model:
    uv run python scripts/train.py --model {{model}} --name {{name}}

# Fine-tune the MTGP-VAE with a tempered CE (5k steps of a cosine decay set for 15k) -> checkpoints/mtgp_vae
finetune init="checkpoints/mtgp_vae_base":
    uv run python scripts/train.py --model mtgp_vae --init {{init}} --name mtgp_vae --lr 5e-4 --beta 5e-4 --ce-temp 10 --steps 5000 --decay-steps 15000

# All the paper's trainings, one after the other (~30 h on one GPU)
train-all: (train "sign_vae") (train "sign_gp_vae") (train "mtgp_vae" "mtgp_vae_base") finetune

# Classify LSFB100 testing with one checkpoint (flags: --posthoc, --split validation) -> results/<name>_testing.json
evaluate ckpt *flags:
    uv run python scripts/evaluate.py {{ckpt}} {{flags}}

# Table 1 of the paper: the four evaluations, then the table (balanced top-1 / top-5)
table: (evaluate "checkpoints/sign_vae") (evaluate "checkpoints/sign_gp_vae") (evaluate "checkpoints/sign_gp_vae" "--posthoc") (evaluate "checkpoints/mtgp_vae")
    uv run python scripts/evaluate.py --table

# Figures 1-3 of the paper -> figures/*.svg, and *.pdf if rsvg-convert is installed
figures:
    uv run python scripts/figures.py
    if command -v rsvg-convert > /dev/null; then for f in figures/*.svg; do rsvg-convert -f pdf -o "${f%.svg}.pdf" "$f"; done; fi

# Statistics of LSFB100 (clips, frames, signers per split)
stats:
    uv run python scripts/data_stats.py
