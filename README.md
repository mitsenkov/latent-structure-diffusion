# Protein backbone diffusion starter.

This repository contains a compact Colab-friendly workflow for training and sampling a backbone diffusion baseline on padded protein coordinates.

The core pieces are:

1. reusable geometry and diffusion code in `src/`
2. a single notebook that audits the data, trains the model, and samples structures
3. a minimal config file for Colab and private repo setup

## Recommended workflow

Clone the repository, create a virtual environment, and install the package in editable mode:

```bash
git clone git@github.com:<your-github-username>/latent-structure-diffusion.git
cd latent-structure-diffusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

In Colab, clone the repo and install it the same way:

```python
!git clone https://github.com/<your-github-username>/latent-structure-diffusion.git
%cd latent-structure-diffusion
!pip install -q -r requirements.txt
!pip install -q -e .
```

Place the CATH data under `data/raw/` in the Colab runtime or point `CATH_DATA_DIR` at the downloaded folder.

## Repository layout

```text
notebooks/              Notebook entry points.
src/latent_structure_generation/ Reusable parsing, geometry, metrics, and diffusion code.
tests/                  Small tests for core geometry functions.
data/raw/               Local or Colab-only input data. Ignored by git.
data/processed/         Intermediate outputs. Ignored by git.
results/                Figures and metric tables. Ignored by git.
reports/                Decision log and report templates.
```

## Current notebook

Start with `notebooks/protein_backbone_diffusion_v1.ipynb`.
