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

Follow-up setup fix: the V3 runtime cell now makes Google Drive mounting optional for PyCharm/IDE execution. `BACKBONE_DIFFUSION_MOUNT_DRIVE=0` skips Drive mounting and disables table writes by default; `BACKBONE_DIFFUSION_SAVE_TABLES=0` disables CSV/Parquet table artifacts explicitly; `BACKBONE_DIFFUSION_SAVE_TABLES=1` re-enables table writes even without Drive. The artifact root now uses Google Drive when mounted successfully and falls back to repo-local `results/v3/` otherwise.

Local execution was not completed in this workspace because the available Python environment does not have `torch` or `nbformat` installed. V3 was then run in Google Colab on an NVIDIA L4, and the extracted artifacts are now under `results/v3/`.

## V3 report: Atom-graph denoiser evaluation

V3 is complete in `notebooks/protein_backbone_diffusion_v3.ipynb`, with artifacts extracted under `results/v3/`. The uploaded zip was extracted and removed after extraction.

### Setup

V3 preserved the V2 DDPM data pipeline, splits, masking, train-only coordinate normalization, sampling procedure, and diagnostics. The main architecture change was replacing the V2 flattened 1D residual convolutional denoiser with `BackboneAtomGraphDenoiser`, a local atom-graph model.

Key run settings:

| Item | V3 value |
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
| Model type | BackboneAtomGraphDenoiser |
| Hidden dimension | 192 |
| Layers | 4 |
| Model parameters | 1,679,843 |

### Training results

V3 trained substantially slower than V2 and achieved worse noise-prediction losses.

| Version | Best epoch | Best validation loss | Test loss at best epoch | Mean seconds per epoch |
|---|---:|---:|---:|---:|
| V2 | 4 | 0.0975 | 0.0992 | 13.7 |
| V3 | 4 | 0.1373 | 0.1393 | 124.3 |

The V3 epoch time was about 9.1x slower than V2. Best validation loss was about 41% worse than V2. This means the atom-graph denoiser was not a better optimizer for the DDPM noise objective under the same five-epoch schedule.

V3 training history:

| Epoch | Train loss | Validation loss | Test loss | Validation x0 RMSE | Test x0 RMSE | Seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2305 | 0.1577 | 0.1615 | 0.2079 | 0.2026 | 122.3 |
| 2 | 0.1631 | 0.1460 | 0.1507 | 0.1957 | 0.1949 | 124.8 |
| 3 | 0.1560 | 0.1518 | 0.1559 | 0.2028 | 0.1960 | 124.8 |
| 4 | 0.1500 | 0.1373 | 0.1393 | 0.1914 | 0.1907 | 124.8 |
| 5 | 0.1442 | 0.1382 | 0.1391 | 0.1898 | 0.1926 | 124.8 |

### Fixed-timestep diagnostics

V3 improved the very-low-noise timestep-1 loss relative to V2, but was worse at every other checked timestep. The high-noise end matters most for sampling because reverse diffusion starts from noise; at timestep 100, V3 validation x0 RMSE was 0.3409 versus V2's 0.2771.

| Timestep | V2 validation loss | V3 validation loss | V2 validation x0 RMSE | V3 validation x0 RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.7993 | 0.6542 | 0.0089 | 0.0081 |
| 5 | 0.2768 | 0.3716 | 0.0264 | 0.0306 |
| 10 | 0.1769 | 0.2833 | 0.0423 | 0.0535 |
| 25 | 0.1066 | 0.1575 | 0.0832 | 0.1011 |
| 50 | 0.0736 | 0.0957 | 0.1452 | 0.1657 |
| 75 | 0.0549 | 0.0754 | 0.2050 | 0.2402 |
| 100 | 0.0439 | 0.0664 | 0.2771 | 0.3409 |

This supports the visual observation that V3 did not solve reverse-sampling quality. It was especially weaker in the high-noise regime used at the start of generation.

### Structural sample quality

The structural metrics are mixed. V3 improved several mean distances relative to V2, but the generated structures remained far from real proteins and still failed the most important CA-band quality check.

| Metric | Real mean | V2 generated | V3 generated | V3 minus V2 | Interpretation |
|---|---:|---:|---:|---:|---|
| Adjacent CA distance | 3.810 | 3.099 | 3.531 | +0.432 | Better and closer to 3.8 A. |
| Adjacent CA in-band fraction | 0.999 | 0.265 | 0.258 | -0.007 | Slightly worse; still broken. |
| N-CA distance | 1.464 | 1.212 | 1.360 | +0.148 | Better and closer to real. |
| CA-C distance | 1.525 | 1.254 | 1.361 | +0.108 | Better but still short. |
| C-O distance | 1.229 | 1.027 | 1.209 | +0.182 | Much closer to real. |
| C-N distance | 1.330 | 1.196 | 1.233 | +0.037 | Slightly better but still short. |
| Radius of gyration | 16.213 | 7.148 | 8.045 | +0.897 | Better, but still about half of real. |

V3 reduced the collapse flag rate but did not eliminate collapse:

| Version | Samples | Collapse fraction | Collapse count | Poor CA-band fraction | Poor CA-band count |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 0.969 | 31/32 | 1.000 | 32/32 |
| V3 | 32 | 0.625 | 20/32 | 1.000 | 32/32 |

The collapse fraction improved from 31/32 to 20/32, but all generated samples still failed the adjacent-CA band-quality threshold. The generated adjacent-CA in-band fraction was essentially unchanged and slightly worse: 0.258 for V3 versus 0.265 for V2.

### Interpretation

The original qualitative conclusion mostly holds, with nuance. V3 was not a useful architecture upgrade in terms of end-to-end sample quality. It was about 9x slower per epoch, had worse validation/test denoising loss, and generated samples still looked collapsed and non-protein-like. However, the atom-graph representation did move several mean bond metrics and radius of gyration in the right direction, and it reduced the binary collapse count from 31/32 to 20/32.

The key failure is that V3 did not improve the decisive local CA geometry metric. Every generated sample still had poor CA-band quality. This means atom-level local message passing alone is not enough. The model needs an objective or sampling procedure that explicitly rewards valid protein geometry.

### V3 claim

```text
I tested an atom-level local graph denoiser as a pragmatic geometry-aware V3 architecture. It preserved the DDPM pipeline and improved some mean bond-length and compactness metrics, but it trained about 9x slower, achieved worse denoising losses, and still produced structurally invalid samples: 20/32 were collapsed and 32/32 failed the adjacent-CA band-quality check. This suggests the next step should be geometry-augmented denoising loss, not another architecture-only tweak.
```

## V3 interpretation and V3b next-step note

The final Colab outputs confirm the planning direction.

The original roadmap used `V3.1` to mean atom-level representation. That part has already been implemented inside V3: each backbone atom is treated as a node with atom-type, residue-position, timestep, coordinate, and mask features. Therefore, the next follow-up should be called `V3b` or `V3.7` rather than reusing `V3.1`, unless the roadmap is deliberately renumbered.

The working recommendation is not to keep tweaking the V3 architecture itself. If atom nodes plus local message passing do not fix collapse, and training is much slower than V1/V2, then the missing piece is probably not just "better local representation."

The obvious missing ingredient is explicit protein geometry supervision during denoising.

Right now the model is trained mostly to:

```text
predict Gaussian noise correctly under a masked MSE
```

That objective does not directly say:

```text
adjacent CA should be about 3.8 A
N-CA / CA-C / C-O / C-N bonds should be realistic
the chain should not collapse globally
```

So the recommended next step is V3b, not V4 yet.

### V3b proposal: geometry-augmented denoising loss

Use the faster V2 flattened denoiser first, or optionally keep the V3 atom denoiser for comparison, and add auxiliary geometry losses on predicted clean coordinates.

During training the notebook already computes:

```python
x0_pred = predict_x0(x_t, t, pred_noise, schedule['alpha_bars'])
```

That gives an estimated clean backbone. Add geometry terms on `x0_pred`:

```text
total_loss =
    noise_mse
    + lambda_bond * bond_length_loss(x0_pred)
    + lambda_ca * adjacent_ca_loss(x0_pred)
    + optional lambda_rg * radius/compactness_regularizer
```

Start simple:

| Term | Initial weight |
|---|---:|
| Noise prediction loss | 1.0 |
| Backbone bond loss | 0.05 or 0.1 |
| Adjacent CA loss | 0.05 or 0.1 |
| Radius/compactness loss | omit initially |

The most defensible additions are:

| Loss | Target |
|---|---|
| Adjacent CA distance loss | Penalize valid adjacent CA distances away from about 3.8 A. |
| N-CA bond loss | Penalize distances away from about 1.46 A. |
| CA-C bond loss | Penalize distances away from about 1.53 A. |
| C-O bond loss | Penalize distances away from about 1.23 A. |
| Adjacent C-N peptide bond loss | Penalize distances away from about 1.33 A. |

Delay pairwise or radius-of-gyration regularization at first. Radius regularization can be brittle because it risks forcing all generated proteins toward one global size.

### Why V4 should wait

An EGNN gives a better inductive bias:

```text
translation/rotation equivariance
coordinate-aware message passing
better geometric symmetry handling
```

But an EGNN does not automatically enforce protein bond lengths. It can still learn a denoising objective and sample collapsed structures if the loss and sampling setup do not constrain valid geometry.

Geometry losses are useful because they carry forward:

```text
V3b: flattened or atom model + geometry losses
V4: EGNN + same geometry losses
future: guidance/projection/internal-coordinate model
```

Recommended sequence:

1. Complete the V3 report once the final Colab metrics are available.
2. Treat current V3 as a negative or mixed result if the metrics confirm that atom-level local message passing did not improve collapse.
3. Create V3b with geometry-augmented training loss.
4. Use the faster V2 flattened denoiser first, because it trains much faster and can test the geometry-loss hypothesis quickly.
5. If geometry losses improve sample metrics, then V4 EGNN is justified as "same protein-aware objective, stronger equivariant architecture."
6. If geometry losses do not improve sampling, the issue is likely deeper: raw Cartesian DDPM sampling may need guidance, projection, or internal-coordinate generation.

The strongest project story would be:

```text
V2 showed collapse.
V3 showed atom-level local message passing alone did not fix collapse.
V3b added explicit protein geometry losses, which are architecture-independent and can carry into EGNN-style V4.
```

## V3b implementation status: Geometry-augmented V2 baseline

V3b has been created as `notebooks/protein_backbone_diffusion_v3b.ipynb`. It intentionally starts from V2 rather than V3, because V2 is much faster and is the stronger baseline for testing whether explicit geometry losses help.

The notebook preserves the V2 DDPM setup:

- train/validation/test split handling
- train-chain chunking
- masking
- train-only coordinate normalization
- 100-step linear noise schedule
- flattened `BackboneDenoiser`
- reverse sampling procedure
- V2-style fixed-timestep diagnostics and generated structural metrics

The main implementation change is in training. After the model predicts noise, V3b reconstructs:

```python
x0_pred = predict_x0(x_t, t, pred_noise, schedule['alpha_bars'])
```

It then converts `x0_pred` from normalized flattened coordinates back to `B x L x 4 x 3` Angstrom coordinates before measuring geometry. The total objective is:

```text
total_loss =
    1.0 * noise_loss
    + 0.05 * backbone_bond_geometry_loss
    + 0.05 * adjacent_ca_geometry_loss
```

The geometry targets are:

| Loss component | Target |
|---|---:|
| Adjacent CA-CA | 3.80 A |
| N-CA | 1.46 A |
| CA-C | 1.53 A |
| C-O | 1.23 A |
| Adjacent C-N | 1.33 A |

Mask handling:

- intra-residue bond losses use the valid residue mask
- adjacent CA-CA and adjacent C-N losses use `mask[:, :-1] & mask[:, 1:]`
- padded residues do not contribute to geometry losses

V3b uses the same optional Google Drive artifact behavior as V3: when Drive mounts successfully, artifacts go under `MyDrive/latent-structure-generation/results/v3b/`; otherwise the notebook falls back to repo-local `results/v3b/`. `BACKBONE_DIFFUSION_MOUNT_DRIVE=0` skips Drive mounting, `BACKBONE_DIFFUSION_SAVE_TABLES=0` disables table artifact writes, and `BACKBONE_DIFFUSION_SAVE_TABLES=1` re-enables table writes even when Drive mounting is skipped.

V3b saves:

- `backbone_diffusion_history` with total, noise, bond-geometry, adjacent-CA-geometry, and x0 RMSE columns for train/validation/test
- `v3b_run_config_summary`
- `v3b_timestep_diagnostics`
- `v3b_real_eval_metrics`
- `v3b_generated_eval_metrics`
- `v3b_real_vs_generated_metric_summary`
- `v3b_structural_gap_summary`
- `v3b_collapse_summary`
- `v3b_vs_v2_metric_comparison` and `v3b_vs_v2_collapse_comparison` when V2 reference artifacts are available
- `v3b_vs_v3_metric_comparison` and `v3b_vs_v3_collapse_comparison` when V3 reference artifacts are available

Radius-of-gyration loss was deliberately not added. It remains an evaluation metric only, because direct radius regularization may force proteins toward one global size and obscure whether local geometry supervision alone helps.

Local validation performed here:

- `python3 -m json.tool notebooks/protein_backbone_diffusion_v3b.ipynb` passed
- all V3b notebook code cells compile with Python `compile(...)`

The full training run has not yet been executed in this local workspace. The next step is to run V3b in Colab, extract `results/v3b/`, and compare adjacent-CA in-band fraction, adjacent-CA mean, bond means, radius of gyration, collapse fraction, and poor CA-band fraction against V2 and V3.
