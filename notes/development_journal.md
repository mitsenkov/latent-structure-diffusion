# Protein Backbone Diffusion Working Notebook

## Current understanding

The repo is centered on a Colab-friendly backbone diffusion baseline for CATH protein chains. The current notebook:

- audits the raw chain tables before training
- normalizes coordinates using train-only statistics
- trains a simple DDPM-style backbone denoiser
- samples structures from noise and renders them

## Data files

The raw data lives in:

- `data/raw/chain_set.jsonl`
- `data/raw/chain_set_splits.json`

The main table contains one row per chain with:

- `seq`
- `coords`
- `num_chains`
- `name`
- `CATH`
- derived `split`

The split file is not just a simple train/validation/test list. It also contains a `cath_nodes` mapping.

## The 96 `unknown` rows

We found 96 rows in `chain_set.jsonl` whose `name` values are not present in the split manifest.

Interpretation:

- they are ordinary chain records with sequences and backbone coordinates
- they are not assigned to train, validation, test, or `cath_nodes`
- the notebook should continue ignoring them for now

This is a data bookkeeping issue, not an immediate modeling issue. The useful conclusion is that the split manifest is not a complete assignment for all rows in the main table.

## Split-file nuance

The `cath_nodes` entry is a dictionary keyed by chain ID. It overlaps heavily with the `test` list.

Important lesson:

- the split file must be parsed carefully
- `test` should not be overwritten by the auxiliary `cath_nodes` bucket
- the notebook now preserves the true train/validation/test split first, then assigns `cath_nodes` only to unassigned records

## The truncation problem

The audit showed that the original fixed-length setting was discarding a lot of protein content.

Observed truncation rates:

- train: 0.411895
- validation: 0.217105
- test: 0.167857

This means the earlier assumption that most proteins would fit cleanly into a single 256-residue window was wrong.

Observed length range:

- some chains are much shorter than 256
- some reach around 500 residues

So a fixed 256-residue tensor window was:

- padding short proteins heavily
- cropping a large fraction of long proteins

## Direction change

The better direction is to keep the fixed-window model, but generate overlapping chunks up front for training.

Current choice:

- chunk long training chains into overlapping 256-residue windows
- track `parent_chain_id`
- track `chunk_id`, `chunk_start`, and `chunk_end`
- keep split membership at the parent-chain level to avoid leakage

This lets us preserve more of the long proteins without redesigning the entire model immediately.

## Dataset heterogeneity

The dataset appears structurally heterogeneous, with variable protein lengths and diverse CATH annotations. Therefore, the initial model should be interpreted as a prototype for learning generic local protein backbone geometry, rather than a fold-family-specific generator. This motivates careful masking, padding/cropping decisions, and evaluation using geometry-based checks rather than expecting fold-specific generation quality.

## Practical takeaway

The previous single-window setup was a workable baseline, but it was not a good approximation of the data distribution.

The next notebook direction should assume:

- protein lengths vary widely
- long chains must be represented by multiple overlapping chunks
- provenance must be preserved so chunks do not leak across splits
