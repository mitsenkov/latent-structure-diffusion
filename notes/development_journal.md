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

## V2 report: Diagnostic baseline evaluation

V2 is complete in `notebooks/protein_backbone_diffusion_v2.ipynb`. The extracted artifacts are under `results/v2/`. V2 intentionally kept the same core model and five-epoch training setup as V1, but added a stronger diagnostic layer so the baseline failure mode is supported by distribution-level evidence rather than a few sampled examples.

### Motivation

After V1, the key question was no longer "does the diffusion pipeline run?" The answer was yes. The next question was: "how exactly does the baseline fail, and is the failure consistent enough to justify a model upgrade?"

V2 was designed to answer that question. It adds experiment metadata, fixed-timestep denoising diagnostics, more generated samples, structural metric distributions, and explicit collapse flags. This makes the baseline easier to defend in a report: V1 proves the method is implemented; V2 proves why the simple denoiser is not enough.

### Setup

The V2 run used the same core setup as V1:

| Item | Value |
|---|---:|
| Device | NVIDIA L4 |
| Seed | 42 |
| Max sequence length | 256 |
| Backbone atoms | N, CA, C, O |
| Train chains | 18,024 |
| Train chunks | 27,706 |
| Validation chains | 608 |
| Test chains | 1,120 |
| Timesteps | 100 |
| Epochs | 5 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Model parameters | 1,750,028 |

The run also fixed the V2 config-summary artifact so it writes both CSV and Parquet. The important lesson is that Parquet needs stable column types; therefore the config table now stores `value_text`, `value_int`, and `value_float` instead of one mixed-type `value` column.

### Training results

The training curve is effectively the same as V1, which is expected because the model and training schedule were unchanged.

| Epoch | Train loss | Validation loss | Test loss | Validation x0 RMSE | Test x0 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2104 | 0.1348 | 0.1404 | 0.1879 | 0.1839 |
| 2 | 0.1280 | 0.1132 | 0.1171 | 0.1755 | 0.1742 |
| 3 | 0.1128 | 0.1094 | 0.1111 | 0.1756 | 0.1703 |
| 4 | 0.1044 | 0.0975 | 0.0992 | 0.1651 | 0.1641 |
| 5 | 0.0996 | 0.1010 | 0.1049 | 0.1607 | 0.1627 |

Best validation loss was `0.0975` at epoch 4, with corresponding test loss `0.0992`. Epoch 5 slightly worsened validation and test noise loss, although x0 RMSE continued to improve slightly. This supports treating the checkpointed epoch-4 model as the best V2 baseline model.

### Fixed-timestep diagnostics

V1 only reported losses averaged over randomly sampled timesteps. V2 evaluated fixed timesteps on validation and test splits:

| Split | Timestep | Noise fraction | Noise loss | x0 RMSE |
|---|---:|---:|---:|---:|
| Validation | 1 | 0.01 | 0.7993 | 0.0089 |
| Validation | 5 | 0.05 | 0.2768 | 0.0264 |
| Validation | 10 | 0.10 | 0.1769 | 0.0423 |
| Validation | 25 | 0.25 | 0.1066 | 0.0832 |
| Validation | 50 | 0.50 | 0.0736 | 0.1452 |
| Validation | 75 | 0.75 | 0.0549 | 0.2050 |
| Validation | 100 | 1.00 | 0.0439 | 0.2771 |
| Test | 1 | 0.01 | 0.7994 | 0.0089 |
| Test | 5 | 0.05 | 0.2745 | 0.0263 |
| Test | 10 | 0.10 | 0.1764 | 0.0422 |
| Test | 25 | 0.25 | 0.1071 | 0.0834 |
| Test | 50 | 0.50 | 0.0732 | 0.1449 |
| Test | 75 | 0.75 | 0.0552 | 0.2054 |
| Test | 100 | 1.00 | 0.0435 | 0.2761 |

The useful interpretation is that the reported noise-prediction MSE and x0 reconstruction RMSE tell different stories. Noise loss decreases at later timesteps, but x0 RMSE worsens as the input becomes more heavily noised. Full sampling starts from the high-noise end, so poor clean-coordinate reconstruction in that regime is consistent with the collapsed samples.

This is an important V2 finding: the baseline can optimize the DDPM noise objective, but that does not guarantee valid reverse-sampled protein geometry.

### Broader sample-quality results

V2 expanded generation from the original four V1 samples to 32 validation-shaped samples. The real and generated samples had the same mean residue count, so the comparison is not explained by different lengths:

| Kind | Structures | Mean residues | Mean adjacent CA | CA in-band fraction | Mean N-CA | Mean CA-C | Mean C-O | Mean C-N | Mean radius of gyration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real | 32 | 158.16 | 3.810 | 0.999 | 1.464 | 1.525 | 1.229 | 1.330 | 16.21 |
| Generated | 32 | 158.16 | 3.099 | 0.265 | 1.212 | 1.254 | 1.027 | 1.196 | 7.15 |

The distribution-level result is stronger than the V1 four-sample check. Real structures form tight distributions around expected local backbone distances. Generated structures are much broader and shifted toward compressed local geometry.

The main structural gaps were:

| Metric | Real mean | Generated mean | Generated minus real | Interpretation |
|---|---:|---:|---:|---|
| Adjacent CA distance | 3.810 | 3.099 | -0.711 | Generated adjacent CA spacing is too short. |
| Adjacent CA in-band fraction | 0.999 | 0.265 | -0.734 | Most generated adjacent CA distances are not protein-like. |
| Radius of gyration | 16.21 | 7.15 | -9.07 | Generated samples are globally over-compact. |

The collapse summary makes the failure mode explicit:

| Sample count | Real radius median | Collapse threshold | Collapse fraction | Poor CA-band fraction |
|---:|---:|---:|---:|---:|
| 32 | 16.34 | 8.17 | 0.96875 | 1.0 |

This means 31 of 32 generated samples were flagged as collapsed using the radius threshold, and all 32 failed the adjacent-CA band-quality check.

### Plot interpretation

The V2 plots are one of the most useful outputs. The real data shows tight, interpretable distributions around expected backbone geometry. The generated data shows the opposite: broader, shifted distributions and very low global size. This makes the baseline failure visually obvious.

The plots support the same conclusion as the tables:

```text
The model has learned a denoising objective, but it has not learned the structural constraints needed to sample protein-like backbones.
```

### Conclusion

V2 did not try to make the generator better. It made the baseline diagnosis better.

The main conclusion is:

```text
The simple 1D coordinate denoiser is not structurally adequate. It trains and evaluates cleanly under the DDPM loss, but reverse sampling produces over-compact, locally distorted backbones.
```

This is expected and useful. It means the next step should not be just more epochs on the same model. A longer run might reduce loss slightly, but the V2 evidence points to a representation/objective problem. V3 should keep the same DDPM pipeline and evaluation harness, but replace the denoiser with a more geometry-aware atom-level or point-cloud-style model.

### V2 claim

```text
I expanded the baseline evaluation from a small qualitative sample check to a distribution-level structural diagnosis. The V2 results show that the baseline learns denoising loss but consistently samples collapsed, non-protein-like backbones, motivating a geometry-aware V3 architecture.
```

## V3 plan: Atom-graph denoiser upgrade

V3 has been created as `notebooks/protein_backbone_diffusion_v3.ipynb` by duplicating V2 and preserving the DDPM data pipeline, chain-level train/validation/test splits, train-only normalization, masking, reverse sampling procedure, and V2 structural diagnostics.

The main change is the denoiser. V3 replaces the flattened residue-wise 1D convolutional model with a pragmatic atom-level graph denoiser:

- each `N`, `CA`, `C`, and `O` backbone atom is represented as a node
- node features include noised coordinates, atom-type embeddings, residue-position embeddings, timestep embeddings, and a residue-derived atom mask
- local graph edges follow simple backbone topology: intra-residue `N-CA`, `CA-C`, `C-O`, adjacent-residue `C-N`, and adjacent `CA-CA`
- message passing uses relative coordinate vectors and squared distances, giving the model a stronger local geometry bias without implementing a full EGNN
- the public model interface remains `(B, L, 12) -> (B, L, 12)`, so training loss, sampling, and structural metrics stay compatible with V2

V3 defaults artifacts to `results/v3/` and saves:

- `backbone_diffusion_history`
- `checkpoints/best_backbone_diffusion.pt`
- `v3_run_config_summary`
- `v3_timestep_diagnostics`
- `v3_real_eval_metrics`
- `v3_generated_eval_metrics`
- `v3_real_vs_generated_summary`
- `v3_real_vs_generated_metric_summary`
- `v3_structural_gap_summary`
- `v3_collapse_summary`
- `v3_vs_v2_metric_comparison`, when `results/v2/tables/v2_real_vs_generated_metric_summary.csv` is available
- `v3_vs_v2_collapse_comparison`, when `results/v2/tables/v2_collapse_summary.csv` is available

Expected success criteria remain deliberately modest. V3 should be judged by whether generated radius of gyration moves upward toward the real distribution, adjacent CA distance moves closer to 3.8 A, adjacent CA in-band fraction improves above the V2 value of about 0.265, collapse fraction drops below 31/32, and bond-length metrics move closer to real values.

Local validation performed here:

- `python -m json.tool notebooks/protein_backbone_diffusion_v3.ipynb` passed
- all V3 notebook code cells compile with Python `compile(...)`

Follow-up setup fix: the V3 runtime cell now makes Google Drive mounting optional for PyCharm/IDE execution. `BACKBONE_DIFFUSION_MOUNT_DRIVE=0` skips Drive mounting and disables table writes by default; `BACKBONE_DIFFUSION_SAVE_TABLES=0` disables CSV/Parquet table artifacts explicitly; `BACKBONE_DIFFUSION_SAVE_TABLES=1` re-enables table writes even without Drive. The default artifact root remains repo-local at `results/v3/`.

Local execution was not completed in this workspace because the available Python environment does not have `torch` or `nbformat` installed. The next step is to run the V3 notebook in the same Colab/GPU environment used for V1 and V2, then replace this plan section with the measured V3 training and structural results.
