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

## V1 report: Completed baseline diffusion model

V1 is now complete in `notebooks/protein_backbone_diffusion_v1.ipynb`. The saved artifacts are under `results/v1/` after extracting `results/v1-20260606T190918Z-3-001.zip`.

### Motivation

The purpose of V1 was to build the simplest defensible end-to-end diffusion baseline for the project: take protein backbone coordinates, train a model to predict Gaussian noise added at random diffusion timesteps, and sample new backbone-like coordinate tensors starting from noise.

The target was not final biological realism. The target was a working baseline with the right generative framing, strict masking, train/validation/test losses, saved outputs, and protein-aware sanity checks. This gives a concrete model to improve in later versions rather than leaving the project at the data-exploration stage.

### Reasoning and method

The model uses the provided CATH backbone-coordinate data with fixed `B x L x 4 x 3` tensors, where the atom axis is `N, CA, C, O` and `L = 256`. Coordinates are centered per structure, normalized using training-set statistics only, then flattened per residue into 12 coordinate channels.

Data handling was intentionally conservative. The audit kept all examples with valid backbone residues and used masks so padded residues did not affect the loss. Because many training chains are longer than 256 residues, long train chains were expanded into overlapping 256-residue chunks with stride 128. This increased the effective training set from 18,024 filtered train chains to 27,706 train chunks while leaving validation and test as held-out chain-level splits.

The diffusion baseline is a DDPM-style noise-prediction model:

```text
Input: noised normalized backbone coordinates x_t, timestep t, residue mask.
Model target: Gaussian noise epsilon that was added to x_0.
Loss: masked MSE over real residues only.
Sampling: start from Gaussian noise and run the learned reverse process back to coordinates.
```

The V1 denoiser is deliberately simple: a 1D residual convolutional network over residue positions with hidden dimension 256, four residual blocks, sinusoidal timestep embedding projected through an MLP, learned positional embeddings, and a 12-channel coordinate-noise output. It uses a linear 100-step noise schedule, Adam with learning rate `1e-3`, batch size 32, gradient clipping, and saves the best checkpoint by validation loss.

### Results and outputs

Key saved outputs:

| Artifact | Meaning |
|---|---|
| `results/v1/tables/backbone_diffusion_history.csv` | Train, validation, and test denoising losses by epoch. |
| `results/v1/figures/backbone_diffusion_losses.png` | Loss-curve figure. |
| `results/v1/checkpoints/best_backbone_diffusion.pt` | Best validation-loss checkpoint. |
| `results/v1/tables/real_eval_metrics.csv` | Structural metrics for four real validation examples. |
| `results/v1/tables/generated_eval_metrics.csv` | Structural metrics for four generated examples. |
| `results/v1/tables/real_vs_generated_summary_mean.csv` | Mean real-versus-generated structural comparison. |
| `results/v1/tables/filter_summary.csv`, `length_summary.csv`, manifests, and audits | Data filtering, split, length, truncation, and chunking documentation. |

Data summary:

| Split | Kept examples | Median length | Mean length | Truncated fraction |
|---|---:|---:|---:|---:|
| Train | 18,024 | 204 | 218.7 | 0.412 |
| Validation | 608 | 146 | 174.2 | 0.217 |
| Test | 1,120 | 138 | 162.2 | 0.168 |

Training summary:

| Epoch | Train loss | Validation loss | Test loss | Validation x0 RMSE | Test x0 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2104 | 0.1348 | 0.1404 | 0.1879 | 0.1839 |
| 2 | 0.1280 | 0.1132 | 0.1171 | 0.1755 | 0.1742 |
| 3 | 0.1128 | 0.1094 | 0.1111 | 0.1756 | 0.1703 |
| 4 | 0.1044 | 0.0975 | 0.0992 | 0.1651 | 0.1641 |
| 5 | 0.0996 | 0.1010 | 0.1049 | 0.1607 | 0.1627 |

The main result is that the model learns the denoising task: train loss decreases from 0.2104 to 0.0996, validation loss improves from 0.1348 to a best value of 0.0975 at epoch 4, and test loss improves from 0.1404 to 0.0992 at epoch 4 before slightly rising to 0.1049 at epoch 5.

Structural comparison on the first four generated samples:

| Metric | Real mean | Generated mean | Interpretation |
|---|---:|---:|---|
| Adjacent CA distance | 3.813 | 3.165 | Generated CA spacing is too short. |
| Fraction adjacent CA in band | 0.999 | 0.273 | Most generated adjacent CA distances fail the real-like band check. |
| N-CA distance | 1.459 | 1.228 | Generated local bond geometry is compressed. |
| CA-C distance | 1.526 | 1.265 | Generated local bond geometry is compressed. |
| C-O distance | 1.231 | 1.000 | Generated carbonyl geometry is compressed. |
| C-N distance | 1.331 | 1.270 | Closer, but still not fully real-like. |
| Radius of gyration | 16.90 | 6.95 | Generated structures are much too compact. |

### Interpretation

V1 is a valid diffusion baseline and a useful project milestone. It demonstrates the complete pipeline: data audit, masked coordinate normalization, DDPM forward noising, timestep-conditioned denoising, train/validation/test tracking, checkpointing, reverse sampling, and protein-aware comparison against real examples.

The generated samples are not yet biologically convincing. The model has learned a denoising objective, but the sampled backbones collapse toward overly compact geometries with shortened local distances. This is visible in the low generated radius of gyration and the poor adjacent CA band fraction. The most likely causes are the simplicity of the denoiser, the lack of explicit geometry constraints, short training, and the difficulty of learning valid protein geometry from coordinate MSE alone.

This is still a strong V1 outcome because the failure mode is measurable and actionable. The next version should keep the same pipeline but improve sample quality with stronger masking/logging, timestep-wise diagnostics, geometry-aware losses or penalties, and eventually a point-cloud or EGNN-style denoiser that better respects 3D structure.
