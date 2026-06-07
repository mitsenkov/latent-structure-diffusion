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
x0_pred = differentiable_predict_x0(x_t, t, pred_noise, schedule['alpha_bars'])
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

## V3b first run review: Geometry losses were logged but not active

The first V3b Colab run was extracted from `results/v3b-20260607T093726Z-3-001.zip` into `results/v3b/`.

### Training loss behavior

The run used the intended settings:

| Item | Value |
|---|---:|
| Device | NVIDIA L4 |
| Epochs | 5 |
| Batch size | 32 |
| Model | `BackboneDenoiser` |
| Parameters | 1,750,028 |
| `lambda_bond` | 0.05 |
| `lambda_ca` | 0.05 |
| Radius loss | 0.0 |

The history showed that validation total loss decreased from `0.3618` to `0.2059`, while validation noise loss followed the same trajectory as V2:

| Epoch | Val total | Val noise | Val bond geometry | Val adjacent CA geometry | Val x0 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3618 | 0.1348 | 2.4149 | 2.1249 | 0.1879 |
| 2 | 0.2836 | 0.1132 | 1.7408 | 1.6669 | 0.1755 |
| 3 | 0.2615 | 0.1094 | 1.3741 | 1.6692 | 0.1756 |
| 4 | 0.2096 | 0.0975 | 0.6721 | 1.5703 | 0.1651 |
| 5 | 0.2059 | 0.1010 | 0.5360 | 1.5619 | 0.1607 |

At first glance this looked encouraging: the bond geometry loss dropped substantially on validation, and adjacent CA geometry loss improved modestly.

### Structural sample comparison

The generated sample metrics, however, were exactly identical to V2:

| Metric | Real mean | V2 generated | V3 generated | V3b first-run generated |
|---|---:|---:|---:|---:|
| Adjacent CA distance | 3.810 | 3.099 | 3.531 | 3.099 |
| Adjacent CA in-band fraction | 0.999 | 0.265 | 0.258 | 0.265 |
| N-CA distance | 1.464 | 1.212 | 1.360 | 1.212 |
| CA-C distance | 1.525 | 1.254 | 1.361 | 1.254 |
| C-O distance | 1.229 | 1.027 | 1.209 | 1.027 |
| C-N distance | 1.330 | 1.196 | 1.233 | 1.196 |
| Radius of gyration | 16.213 | 7.148 | 8.045 | 7.148 |

Collapse comparison:

| Version | Samples | Collapse fraction | Collapse count | Poor CA-band fraction | Poor CA-band count |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 0.969 | 31/32 | 1.000 | 32/32 |
| V3 | 32 | 0.625 | 20/32 | 1.000 | 32/32 |
| V3b first run | 32 | 0.969 | 31/32 | 1.000 | 32/32 |

The exact equality to V2 is too strong to interpret as a normal experimental outcome. It indicates that V3b did not actually change the trained model relative to V2.

### Root cause

The issue was implementation-level. The shared helper `predict_x0` in `src/latent_structure_generation/backbone_diffusion.py` is decorated with `@torch.no_grad()`. V3b used that helper when computing geometry losses:

```python
x0_pred = predict_x0(x_t, t, pred_noise, schedule['alpha_bars'])
```

That meant:

- geometry losses were computed and logged
- geometry losses contributed numerically to `total_loss`
- but geometry losses could not backpropagate into the denoiser, because `x0_pred` was detached from the computation graph
- only the original noise MSE updated the model

This explains why the V3b first-run noise losses, generated metrics, collapse summary, and sampled structures matched V2 exactly.

### Correction made

The V3b notebook has been corrected to use a local differentiable reconstruction:

```python
def differentiable_predict_x0(x_t, t, pred_noise, alpha_bars):
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
    return (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
```

Training and fixed-timestep diagnostics now call `differentiable_predict_x0(...)` instead of the no-grad shared helper. The corrected notebook was revalidated locally:

- `python3 -m json.tool notebooks/protein_backbone_diffusion_v3b.ipynb` passed
- all V3b notebook code cells compile with Python `compile(...)`

The first V3b run should therefore be treated as an invalid geometry-loss experiment but a useful debugging result. The corrected notebook needs a fresh Colab run before judging whether geometry-augmented denoising improves sample quality.

### Interpretation

The important insight is not that geometry losses failed. They were not active. The correct conclusion is:

```text
V3b first run reproduced V2 because the geometry terms were detached by a no-grad x0 reconstruction helper.
```

The real V3b test remains pending. After rerunning the corrected notebook, the key success metrics are still:

- adjacent CA in-band fraction above the V2/V3 level around `0.26`
- adjacent CA mean moving toward `3.8 A`
- bond means moving toward real values
- radius of gyration moving upward from the V2 value around `7.15`
- collapse fraction dropping below V2 `31/32`, ideally below V3 `20/32`
- poor CA-band count dropping below `32/32`

## V3b corrected run report: Geometry loss changes the model but does not solve continuity

The corrected V3b Colab run was extracted from `results/v3b-20260607T100118Z-3-001.zip` into `results/v3b/`. The zip file was deleted after extraction. This run used the corrected differentiable `x0_pred` reconstruction, so the geometry losses were active during training.

### Training behavior

Compared with V2, corrected V3b was only modestly slower, not V3-level slow:

| Version | Mean epoch seconds | Best epoch | Best validation noise loss | Test noise at best epoch |
|---|---:|---:|---:|---:|
| V2 | 13.7 | 4 | 0.0975 | 0.0992 |
| V3 | 124.3 | 4 | 0.1373 | 0.1393 |
| V3b corrected | 16.6 | 5 | 0.1499 | 0.1538 |

The geometry terms clearly affected optimization. Unlike the incorrect first run, corrected V3b no longer reproduced V2 losses or samples exactly. However, the cost was a substantially worse noise-prediction loss. Validation noise loss ended at `0.1499`, worse than both V2 and V3.

Corrected V3b history:

| Epoch | Val total | Val noise | Val bond geometry | Val adjacent CA geometry | Val x0 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3665 | 0.1809 | 1.5838 | 2.1285 | 0.2049 |
| 2 | 0.3244 | 0.1671 | 1.4070 | 1.7394 | 0.1937 |
| 3 | 0.3374 | 0.2140 | 1.0467 | 1.4212 | 0.2072 |
| 4 | 0.2537 | 0.1617 | 0.6714 | 1.1701 | 0.1865 |
| 5 | 0.2267 | 0.1499 | 0.6751 | 0.8619 | 0.1804 |

The validation bond and adjacent-CA geometry losses did decrease, which confirms the added objective is now active. The model is being pushed toward shorter-range geometric targets, but that did not translate into valid generated chains.

### Generated structure comparison

Corrected V3b made some local means better, but worsened the decisive chain-continuity and collapse indicators.

| Metric | Real mean | V2 generated | V3 generated | V3b corrected | V3b vs V2 |
|---|---:|---:|---:|---:|---:|
| Adjacent CA distance | 3.810 | 3.099 | 3.531 | 3.332 | +0.233 |
| Adjacent CA in-band fraction | 0.999 | 0.265 | 0.258 | 0.152 | -0.113 |
| N-CA distance | 1.464 | 1.212 | 1.360 | 1.284 | +0.072 |
| CA-C distance | 1.525 | 1.254 | 1.361 | 1.342 | +0.089 |
| C-O distance | 1.229 | 1.027 | 1.209 | 1.037 | +0.010 |
| C-N distance | 1.330 | 1.196 | 1.233 | 1.541 | +0.345 |
| Radius of gyration | 16.213 | 7.148 | 8.045 | 6.873 | -0.275 |

Interpretation:

- Adjacent CA mean improved relative to V2, moving from `3.099` to `3.332`, but it remains below the real value around `3.81`.
- The adjacent CA in-band fraction got worse, dropping from `0.265` to `0.152`. This matches the visual observation of more discontinuous chains: average spacing moved in the right direction, but the distribution is less consistently in the acceptable band.
- N-CA and CA-C improved relative to V2, while C-O barely improved.
- Adjacent C-N overshot badly to `1.541` versus the real `1.330`, suggesting the simple geometry objective is pulling parts of the backbone unevenly.
- Radius of gyration fell from `7.148` to `6.873`, so corrected V3b is slightly more collapsed by this metric, not less.

Collapse comparison:

| Version | Samples | Collapse fraction | Collapse count | Poor CA-band fraction | Poor CA-band count |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 0.969 | 31/32 | 1.000 | 32/32 |
| V3 | 32 | 0.625 | 20/32 | 1.000 | 32/32 |
| V3b corrected | 32 | 0.969 | 31/32 | 1.000 | 32/32 |

Corrected V3b did not reduce collapse count or poor CA-band count. It remained at the V2 failure level for both binary summaries.

### Interpretation

Corrected V3b is a useful negative result. It proves that geometry losses can change the learned model, because generated metrics are no longer identical to V2. But the particular implementation, weights, and raw Cartesian DDPM sampling setup did not improve the main failure mode.

The most important observation is the split between mean-distance improvement and band-quality degradation:

```text
V3b improved average adjacent CA distance, but worsened the fraction of adjacent CA distances in the valid band.
```

That is consistent with visually discontinuous chains. A scalar mean can move toward the target while the distribution remains broad or multimodal. For protein generation, the in-band fraction is more important than the mean.

The corrected V3b result suggests:

- simple unweighted target-distance penalties on `x0_pred` are not enough
- geometry losses can fight the DDPM noise objective and worsen denoising loss
- stronger local objectives may need robust/clipped losses, timestep weighting, or sampling-time projection/guidance
- architecture still matters: V3 had better radius and collapse than V3b despite worse noise loss

### Recommended next step

Do not spend many more epochs on this exact V3b configuration. The validation geometry losses were still improving, but the generated samples show the objective is not targeting the right failure cleanly.

Better follow-up options:

1. Add distribution-aware diagnostics for generated adjacent CA distances, not just means.
2. Try a gentler geometry objective:
   - lower weights such as `lambda_bond = 0.01`, `lambda_ca = 0.01`
   - or apply geometry loss only at mid/late denoising timesteps
   - or use Huber/soft-clipped distance loss to avoid overcorrecting bad predictions
3. Consider sampling-time geometry guidance or projection rather than training-only penalties.
4. Move to V4 EGNN only with the lesson that architecture and objective both matter.

### V3b corrected claim

```text
I tested a corrected geometry-augmented DDPM objective on the faster V2 denoiser. The geometry terms were active and changed the generated samples, improving some average local distances, but they worsened adjacent-CA band quality, did not reduce collapse, and degraded denoising loss. This suggests that naive x0 bond-distance penalties are insufficient; future work should use more stable geometry objectives, sampling guidance/projection, or combine protein-aware losses with a stronger equivariant architecture.
```

## V3b safer geometry objective refactor

After the corrected V3b run, the notebook was refactored for a gentler follow-up experiment rather than continuing with the aggressive plain-MSE geometry setup.

The new V3b configuration keeps the same fast V2 denoiser and same output path, but changes the geometry objective:

| Setting | Previous corrected V3b | Refactored V3b |
|---|---:|---:|
| Bond geometry weight | 0.05 | 0.01 |
| Adjacent CA geometry weight | 0.05 | 0.01 |
| Geometry loss type | Squared error | Smooth L1 / Huber |
| Smooth L1 beta | n/a | 0.5 |
| Geometry timestep range | all timesteps | `t <= 50` |
| Radius loss | 0.0 | 0.0 |

Rationale:

- the previous active geometry losses degraded validation noise loss too much
- high-noise `x0_pred` estimates are unstable, so precise bond penalties are only applied at lower/mid timesteps
- Smooth L1 should reduce the impact of extreme early distance errors
- lower weights should preserve more of the V2 denoising behavior while still giving geometry a signal

The notebook now also saves `v3b_adjacent_ca_distribution_summary`, because the previous corrected run showed that adjacent CA mean can improve while the adjacent CA in-band fraction gets worse. The next run should judge the distribution and in-band fraction, not just the mean.

Local validation after this refactor:

- `python3 -m json.tool notebooks/protein_backbone_diffusion_v3b.ipynb` passed
- all V3b notebook code cells compile with Python `compile(...)`

## V3b stable-geometry run report: modest improvement over V2

The safer refactored V3b run was extracted from `results/v3b-20260607T102254Z-3-001.zip` into `results/v3b/`. This is the current canonical V3b result.

### Setup

This run used the gentler geometry objective:

| Setting | Value |
|---|---:|
| Model | `BackboneDenoiser` |
| Parameters | 1,750,028 |
| Epochs | 5 |
| Batch size | 32 |
| `lambda_bond` | 0.01 |
| `lambda_ca` | 0.01 |
| Geometry loss type | Smooth L1 |
| Smooth L1 beta | 0.5 |
| Geometry active timesteps | `t <= 50` |
| Radius loss | 0.0 |

### Training behavior

The stable V3b objective preserved the denoising baseline much better than the aggressive geometry-loss run.

| Version | Mean epoch seconds | Best epoch | Best validation noise loss | Test noise at best epoch |
|---|---:|---:|---:|---:|
| V2 | 13.7 | 4 | 0.0975 | 0.0992 |
| V3 | 124.3 | 4 | 0.1373 | 0.1393 |
| V3b aggressive | 16.6 | 5 | 0.1499 | 0.1538 |
| V3b stable | 17.2 | 4 | 0.1005 | 0.1024 |

The stable V3b run is only slightly slower than V2 and keeps validation/test noise loss close to V2. This is a much better optimization tradeoff than the aggressive corrected V3b run.

Stable V3b history:

| Epoch | Val total | Val noise | Val bond geometry | Val adjacent CA geometry | Val x0 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1444 | 0.1351 | 0.2909 | 0.6303 | 0.1919 |
| 2 | 0.1188 | 0.1127 | 0.2252 | 0.3860 | 0.1744 |
| 3 | 0.1171 | 0.1112 | 0.1796 | 0.4092 | 0.1720 |
| 4 | 0.1050 | 0.1005 | 0.1688 | 0.2819 | 0.1674 |
| 5 | 0.1052 | 0.1007 | 0.1592 | 0.2889 | 0.1635 |

### Generated structure comparison

Stable V3b improved the targeted local trace metrics modestly relative to V2 and the aggressive V3b run.

| Metric | Real mean | V2 generated | V3 generated | V3b aggressive | V3b stable |
|---|---:|---:|---:|---:|---:|
| Adjacent CA distance | 3.810 | 3.099 | 3.531 | 3.332 | 3.219 |
| Adjacent CA in-band fraction | 0.999 | 0.265 | 0.258 | 0.152 | 0.300 |
| N-CA distance | 1.464 | 1.212 | 1.360 | 1.284 | 1.279 |
| CA-C distance | 1.525 | 1.254 | 1.361 | 1.342 | 1.310 |
| C-O distance | 1.229 | 1.027 | 1.209 | 1.037 | 1.016 |
| C-N distance | 1.330 | 1.196 | 1.233 | 1.541 | 1.206 |
| Radius of gyration | 16.213 | 7.148 | 8.045 | 6.873 | 7.303 |

Relative to V2:

- adjacent CA mean improved from `3.099` to `3.219`
- adjacent CA in-band fraction improved from `0.265` to `0.300`
- radius of gyration improved from `7.148` to `7.303`
- N-CA and CA-C means moved closer to real
- C-N avoided the aggressive run's overshoot and stayed near V2

The adjacent CA distribution summary confirms the improvement:

| Kind | Mean adjacent CA | Median adjacent CA | CA in-band mean | CA in-band median |
|---|---:|---:|---:|---:|
| Real | 3.810 | 3.806 | 0.999 | 1.000 |
| V3b stable generated | 3.219 | 3.205 | 0.300 | 0.305 |

For generated samples, the adjacent-CA in-band interquartile range was roughly `0.240` to `0.353`, and the 95th percentile reached `0.406`. That is still far from protein-like, but it is a real improvement over V2's mean `0.265`.

### Collapse summary

Stable V3b did not improve the binary collapse or poor-CA-band counts:

| Version | Samples | Collapse fraction | Collapse count | Poor CA-band fraction | Poor CA-band count |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 0.969 | 31/32 | 1.000 | 32/32 |
| V3 | 32 | 0.625 | 20/32 | 1.000 | 32/32 |
| V3b stable | 32 | 0.969 | 31/32 | 1.000 | 32/32 |

This means the visual improvement is not strong enough to cross the current collapse threshold. Stable V3b is less bad than V2 on mean radius and CA-band fraction, but it still mostly samples compact structures.

### Interpretation

Stable V3b is the first V3b result that genuinely supports the geometry-loss idea. It is not a solution, but it improves the right local metrics without destroying the DDPM objective.

The key lesson is:

```text
Geometry supervision helps only when it is gentle and applied where x0 predictions are stable enough.
```

Compared with aggressive V3b, the safer objective:

- restored noise loss close to V2
- improved CA in-band fraction instead of damaging it
- improved radius of gyration slightly
- avoided the adjacent C-N overshoot

Compared with V3, the result is mixed:

- V3 still has better adjacent CA mean, radius of gyration, and collapse count
- stable V3b has better CA in-band fraction and is about 7x faster per epoch
- both still fail the poor-CA-band criterion for all generated samples

### Recommended next step

The stable V3b result is worth extending. The next run should increase training length before changing the architecture again.

Recommended follow-up:

1. Run stable V3b for `10` epochs with the same settings.
2. Compare epoch-5 and epoch-10 generated metrics, especially:
   - adjacent CA in-band fraction
   - radius of gyration
   - collapse count
   - bond means
   - validation noise loss
3. If 10 epochs improves CA in-band and radius without hurting noise loss, use stable V3b as the best fast baseline.
4. If 10 epochs plateaus, then consider either sampling-time projection/guidance or V4 EGNN.

### V3b stable claim

```text
I refined the V3b geometry objective using lower weights, Smooth L1 distance penalties, and timestep gating. This stable geometry-loss variant preserved V2-like denoising loss while modestly improving adjacent CA in-band fraction, adjacent CA mean, radius of gyration, and selected bond lengths. It still did not fix global collapse or poor CA-band failures, but it shows that careful geometry supervision can move the generator in the right direction without the severe degradation seen in the aggressive V3b run.
```

## V3b stable 100-epoch run report: the first clearly useful generation improvement

The stable V3b notebook was then rerun for `100` epochs with the same objective:

```text
lambda_bond = 0.01
lambda_ca = 0.01
geometry loss = Smooth L1
geometry active for t <= 50
```

This produced the strongest result so far.

### Training behavior

The model continued improving well beyond 5 epochs. The best validation total loss occurred at epoch `86`.

| Run | Epochs | Best epoch | Best val total | Best val noise | Test noise at best |
|---|---:|---:|---:|---:|---:|
| V2 | 5 | 4 | n/a | 0.0975 | 0.0992 |
| V3 | 5 | 4 | n/a | 0.1373 | 0.1393 |
| V3b stable | 5 | 4 | 0.1050 | 0.1005 | 0.1024 |
| V3b stable long | 100 | 86 | 0.0699 | 0.0688 | 0.0742 |

This is important. The long V3b run no longer just preserves the DDPM objective; it surpasses the earlier noise-loss baselines on held-out data.

One evaluation caveat matters here: the notebook saves the best-validation checkpoint, but the sampling/evaluation section still uses the in-memory model after the final epoch unless the checkpoint is explicitly reloaded. So the structural metrics extracted from this run most likely correspond to the epoch-100 model, while the best validation row was epoch `86`. Sampling from the saved best checkpoint may be slightly better than the current exported metrics.

### Structural sample comparison

The 100-epoch run improved the most important structural indicators sharply relative to V2, V3, and the 5-epoch stable V3b run.

| Metric | Real mean | V2 | V3 | V3b stable 5 ep | V3b stable 100 ep |
|---|---:|---:|---:|---:|---:|
| Adjacent CA mean | 3.810 | 3.099 | 3.531 | 3.219 | 3.162 |
| Adjacent CA in-band fraction | 0.999 | 0.265 | 0.258 | 0.300 | 0.728 |
| N-CA distance | 1.464 | 1.212 | 1.360 | 1.279 | 1.151 |
| CA-C distance | 1.525 | 1.254 | 1.361 | 1.310 | 1.190 |
| C-O distance | 1.229 | 1.027 | 1.209 | 1.016 | 0.964 |
| C-N distance | 1.330 | 1.196 | 1.233 | 1.206 | 1.269 |
| Radius of gyration | 16.213 | 7.148 | 8.045 | 7.303 | 7.855 |

The best signs are:

- adjacent CA in-band fraction jumped from `0.300` at 5 epochs to `0.728` at 100 epochs
- radius of gyration rose from `7.303` to `7.855`
- collapse count dropped from `31/32` to `19/32`
- poor CA-band count dropped from `32/32` to `18/32`

This is the first result where the binary structural summaries improved strongly rather than just the means.

### Adjacent-CA distribution interpretation

The adjacent-CA distribution summary is especially informative:

| Kind | Mean adjacent CA | Median adjacent CA | CA in-band mean | CA in-band median |
|---|---:|---:|---:|---:|
| Real | 3.810 | 3.806 | 0.999 | 1.000 |
| V3b stable 100 ep generated | 3.162 | 3.268 | 0.728 | 0.760 |

This resolves the ambiguity from earlier runs. The mean adjacent CA distance alone is not spectacular, but the chain-level in-band fraction is now much better. That matches the visual observation of more coherent helices and less tangled local structure.

The tradeoff is that some bond means drifted away from the ideal values, especially `N-CA`, `CA-C`, and `C-O`. So the model appears to be learning a better C-alpha trace and less collapsed topology before fully recovering atom-level bond geometry.

### Collapse summary

| Version | Samples | Collapse fraction | Collapse count | Poor CA-band fraction | Poor CA-band count |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 0.969 | 31/32 | 1.000 | 32/32 |
| V3 | 32 | 0.625 | 20/32 | 1.000 | 32/32 |
| V3b stable 5 ep | 32 | 0.969 | 31/32 | 1.000 | 32/32 |
| V3b stable 100 ep | 32 | 0.594 | 19/32 | 0.562 | 18/32 |

This is the strongest evidence that the long stable V3b run is materially better. It slightly beats V3 on collapse count while dramatically outperforming V3 on poor-CA-band count.

### Interpretation

The 100-epoch run changes the project picture.

The earlier conclusion was:

```text
Geometry supervision is only a mild local improvement.
```

The new conclusion is:

```text
Stable geometry supervision plus longer training materially improves sampled backbone continuity and reduces collapse.
```

That does not mean the problem is solved. Global compactness is still poor relative to real proteins, and several bond means remain too short. But the generator is no longer just producing uniformly broken local geometry. It is now clearly recovering a meaningful amount of backbone-like structure.

This also answers the earlier epoch question: for this stable V3b setup, `5` epochs was too short to reveal the real effect. The geometry-aware model needed much longer training to turn local supervision into visible structural gains.

### What the model currently penalizes

The current V3b objective penalizes only a small protein-backbone subset of distances:

- intra-residue `N-CA`
- intra-residue `CA-C`
- intra-residue `C-O`
- inter-residue `C-N`
- adjacent `CA-CA`

It does **not** penalize:

- all pairwise atom distances
- nonlocal CA-CA distances
- radius of gyration directly
- clashes directly
- torsion angles directly

So the improvements seen here came from a very limited local-geometry objective.

### Recommended next step

Do not jump to V4 yet. The current evidence says the stable V3b line still has headroom.

Best next experiments:

1. Keep the same stable V3b objective and test a modestly longer run window with checkpointed sampling summaries, for example every `10` epochs.
2. Add a lightweight global anti-collapse term only after preserving the current local-trace gains.
3. Prefer a gentle global regularizer or sampling-time guidance over an aggressive radius loss.

The most defensible next modification would be one of:

- a weak collapse penalty based on radius or spread, with a very small weight
- a weak nonlocal CA-distance consistency term
- sampling-time geometry guidance/projection

I would avoid changing the architecture immediately, because the current objective is finally showing that it can work.

### V3b stable 100-epoch claim

```text
The stable geometry-augmented V3b model became substantially better when trained longer. At 100 epochs it preserved strong denoising performance, raised adjacent CA in-band fraction from 0.265 to 0.728, improved radius of gyration, reduced collapse from 31/32 to 19/32, and reduced poor CA-band failures from 32/32 to 18/32. The model still collapses globally and does not yet match real bond geometry, but it is now recovering meaningful backbone-like structure, including visible alpha-helical segments.
```

### V3c implementation note: gentle anti-collapse term

V3c should start from the stable V3b objective rather than changing the architecture again. The only new training term is a tiny lower-bound penalty on CA radius of gyration computed from `x0_pred`.

The point of V3c is narrow:

- improve radius of gyration enough to discourage obvious collapse
- reduce collapse count and poor CA-band count
- preserve the V3b local-geometry gains as much as possible

The anti-collapse term should remain soft. It should do nothing once a structure is above a conservative radius threshold, and it should stay small enough that validation noise loss and adjacent CA in-band fraction are not destroyed.

## 2026-06-07 V3c 100-epoch result review

V3c was run for 100 epochs with the stable V3b local-geometry objective plus a weak anti-collapse term.

Configuration:

```text
epochs: 100
timesteps: 100
lambda_bond: 0.01
lambda_ca: 0.01
lambda_collapse: 0.0015
geometry_max_timestep: 50
collapse_radius_min: 8.755 A
best epoch: 94
best validation total loss: 0.07349
best validation noise loss: 0.07206
test noise loss at reported best epoch: 0.07625
```

### Generated structural metrics

| Metric | Real | V2 | V3 | V3b 100 ep | V3c 100 ep |
|---|---:|---:|---:|---:|---:|
| Adjacent CA mean | 3.810 | 3.099 | 3.531 | 3.162 | 3.382 |
| Adjacent CA in-band fraction | 0.999 | 0.265 | 0.258 | 0.728 | 0.781 |
| Radius of gyration | 16.213 | 7.148 | 8.045 | 7.855 | 7.650 |
| N-CA mean | 1.464 | 1.212 | 1.360 | 1.151 | 1.257 |
| CA-C mean | 1.525 | 1.254 | 1.361 | 1.190 | 1.319 |
| C-O mean | 1.229 | 1.027 | 1.209 | 0.964 | 1.053 |
| C-N mean | 1.330 | 1.196 | 1.233 | 1.269 | 1.289 |

Collapse summary:

| Version | Sample count | Collapse count | Collapse fraction | Poor CA-band count | Poor CA-band fraction |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 31 | 0.969 | 32 | 1.000 |
| V3 | 32 | 20 | 0.625 | 32 | 1.000 |
| V3b 100 ep | 32 | 19 | 0.594 | 18 | 0.562 |
| V3c 100 ep | 32 | 26 | 0.812 | 17 | 0.531 |

### Interpretation

V3c is a mixed result.

The good news is that local backbone validity improved again. Compared with the 100-epoch V3b result, V3c moved adjacent CA distances closer to the 3.8 A target, raised adjacent CA in-band fraction from `0.728` to `0.781`, and moved all tracked bond means closer to real values. The per-sample distribution is also encouraging: median CA-band fraction is about `0.788`, and the best generated samples reach `0.988`, which means some samples are now locally close to valid backbone traces.

The bad news is that the anti-collapse term did not reduce global collapse. Radius of gyration decreased from `7.855` in V3b to `7.650` in V3c, still far below the real mean of `16.213`. Collapse count worsened from `19/32` to `26/32`. This means the current weak radius lower-bound loss is not enough to make samples globally expanded, even though optimization itself looks healthy.

The most important conclusion is:

```text
V3c improves local geometry but does not solve global collapse.
```

This suggests the anti-collapse term is either too weak, applied too indirectly, or mismatched to the sampling failure mode. It should not be presented as a clean improvement over V3b yet. The strongest current result remains the stable 100-epoch V3b/V3c family as evidence that geometry supervision helps, with V3b better on collapse and V3c better on local distance quality.

### Recommended next experiment

Do not jump straight to a large architecture change based only on this run. The immediate next experiment should isolate global guidance more carefully:

```text
Keep the stable V3c setup, but test a stronger or better-targeted anti-collapse mechanism while monitoring whether CA-band quality is preserved.
```

Practical options:

- raise `lambda_collapse` cautiously, for example from `0.0015` to `0.003` or `0.005`
- apply the collapse penalty over a wider timestep range rather than only low-noise geometry timesteps
- try a nonlocal CA-distance spread term instead of, or alongside, radius-of-gyration hinge loss
- keep best-checkpoint sampling enabled so the generated metrics are tied to the best validation model

The risk remains that a radius penalty can expand structures unnaturally. Therefore, success should be judged by multiple metrics together, not radius alone: CA-band fraction, bond means, collapse count, poor CA-band count, and visual continuity all need to move in the right direction.

## 2026-06-07 V3c aggressive anti-collapse review

The next V3c run increased the anti-collapse setting:

```text
lambda_bond: 0.01
lambda_ca: 0.01
lambda_collapse: 0.01
geometry_max_timestep: 75
epochs: 100
best epoch: 82
best validation total loss: 0.07084
best validation noise loss: 0.06940
test noise loss at reported best epoch: 0.07251
```

The extracted outputs are stored separately under:

```text
results/v3c_aggressive_lambda001/
```

### Important comparability caveat

This run may not be a clean from-scratch comparison. Epoch 1 already starts at a low train total loss of about `0.081`, whereas the previous clean V3c run started around `0.232`. That strongly suggests the notebook may have continued from an already-trained in-memory model or checkpointed state rather than starting from a fresh initialization.

Therefore this result is still useful as an intervention test, but it should not be treated as a clean apples-to-apples V3c rerun unless the runtime was restarted and the notebook was executed from the top.

### Generated structural metrics

| Metric | Real | V3b 100 ep | V3c weak collapse | V3c aggressive collapse |
|---|---:|---:|---:|---:|
| Adjacent CA mean | 3.810 | 3.162 | 3.382 | 2.845 |
| Adjacent CA in-band fraction | 0.999 | 0.728 | 0.781 | 0.595 |
| Radius of gyration | 16.213 | 7.855 | 7.650 | 7.525 |
| N-CA mean | 1.464 | 1.151 | 1.257 | 1.008 |
| CA-C mean | 1.525 | 1.190 | 1.319 | 1.041 |
| C-O mean | 1.229 | 0.964 | 1.053 | 0.837 |
| C-N mean | 1.330 | 1.269 | 1.289 | 1.223 |

Collapse summary:

| Version | Collapse count | Collapse fraction | Poor CA-band count | Poor CA-band fraction |
|---|---:|---:|---:|---:|
| V3b 100 ep | 19/32 | 0.594 | 18/32 | 0.562 |
| V3c weak collapse | 26/32 | 0.812 | 17/32 | 0.531 |
| V3c aggressive collapse | 24/32 | 0.750 | 24/32 | 0.750 |

### Interpretation

The aggressive anti-collapse run had a real effect, but not the desired measured effect.

Training loss looked numerically strong, with lower best validation noise loss than the previous weak-collapse V3c run. However, the generated structures became worse by the structural metrics that matter most. Adjacent CA mean moved away from the target, CA-band fraction dropped from `0.781` to `0.595`, and all tracked bond lengths became too short. Radius of gyration also decreased slightly rather than increasing.

The visual impression of more relaxed or opened-up chains may reflect local rearrangement or less visually tangled samples, but the metric summary says the model did not actually produce globally expanded protein-like backbones. The generated radius remains about `7.5 A`, far below the real value around `16.2 A`.

This result suggests that simply increasing `lambda_collapse` is not enough, and may interfere with local geometry when combined with a wider geometry timestep window.

### Working conclusion

```text
The aggressive V3c setting is not a clear improvement. It may alter the visual character of samples, but it worsens adjacent CA validity, bond lengths, poor CA-band count, and radius of gyration.
```

The best current evidence remains:

```text
Stable local geometry supervision is useful.
The current radius-hinge anti-collapse term is not solving global collapse.
```

### Recommended response

For a clean next check, restart the runtime and run from the top with one controlled change at a time:

```text
Option A:
lambda_collapse = 0.005
geometry_max_timestep = 50

Option B:
lambda_collapse = 0.005
geometry_max_timestep = 75
```

Avoid combining a large collapse weight and a wider timestep range until a clean intermediate setting shows real improvement in both radius and CA-band quality.

The next more principled anti-collapse experiment should probably move away from only radius-of-gyration hinge loss. A weak nonlocal CA-distance spread objective may better discourage collapsed folds without forcing every chain toward one global radius.

## 2026-06-07 V3c moderate anti-collapse review

A cleaner intermediate V3c run was completed with:

```text
lambda_bond: 0.01
lambda_ca: 0.01
lambda_collapse: 0.005
geometry_max_timestep: 50
epochs: 100
timesteps: 100
learning_rate: 0.001
best epoch: 89
best validation total loss: 0.07304
best validation noise loss: 0.07172
test noise loss at reported best epoch: 0.07277
```

The extracted outputs are stored separately under:

```text
results/v3c_mid_lambda0005_t50/
```

Unlike the aggressive `lambda_collapse=0.01, geometry_max_timestep=75` run, this appears to be a clean from-scratch run. Epoch 1 starts with high loss:

```text
epoch 1 train total loss: 0.2328
epoch 1 validation total loss: 0.1434
```

That pattern matches the earlier clean runs and does not look like continuation from an already-trained in-memory model.

### Training behavior

The optimization itself looks normal. Loss falls rapidly during the first few epochs and then improves more slowly:

```text
best validation total loss: 0.07304 at epoch 89
best validation noise loss: 0.07172 at epoch 89
```

The validation geometry losses fluctuate in later epochs, but not in a catastrophic way. This does not look like learning-rate instability. The more important issue is that the anti-collapse loss is still usually tiny or zero on validation batches, so the radius-hinge term is not reliably steering the generated samples away from collapse.

### Generated structural metrics

| Metric | Real | V3b 100 ep | V3c weak collapse | V3c moderate collapse | V3c aggressive collapse |
|---|---:|---:|---:|---:|---:|
| Adjacent CA mean | 3.810 | 3.162 | 3.382 | 3.205 | 2.845 |
| Adjacent CA in-band fraction | 0.999 | 0.728 | 0.781 | 0.704 | 0.595 |
| Radius of gyration | 16.213 | 7.855 | 7.650 | 7.416 | 7.525 |
| N-CA mean | 1.464 | 1.151 | 1.257 | 1.189 | 1.008 |
| CA-C mean | 1.525 | 1.190 | 1.319 | 1.243 | 1.041 |
| C-O mean | 1.229 | 0.964 | 1.053 | 0.991 | 0.837 |
| C-N mean | 1.330 | 1.269 | 1.289 | 1.217 | 1.223 |

Collapse summary:

| Version | Collapse count | Collapse fraction | Poor CA-band count | Poor CA-band fraction |
|---|---:|---:|---:|---:|
| V3b 100 ep | 19/32 | 0.594 | 18/32 | 0.562 |
| V3c weak collapse, 0.0015/t50 | 26/32 | 0.812 | 17/32 | 0.531 |
| V3c moderate collapse, 0.005/t50 | 26/32 | 0.812 | 19/32 | 0.594 |
| V3c aggressive collapse, 0.01/t75 | 24/32 | 0.750 | 24/32 | 0.750 |

### Interpretation

This run is a partial improvement only if compared against the original V2/V3 failures. It is much better than V2 and V3 on adjacent CA in-band fraction:

```text
V2 CA in-band: 0.265
V3 CA in-band: 0.258
V3c moderate CA in-band: 0.704
```

However, against the stronger 100-epoch V3b and weak-collapse V3c baselines, the moderate-collapse run is not a clear improvement.

Compared with V3b 100 epoch:

```text
adjacent CA mean improves slightly: 3.162 -> 3.205
bond means improve for N-CA, CA-C, and C-O
CA in-band worsens: 0.728 -> 0.704
radius worsens: 7.855 -> 7.416
collapse worsens: 19/32 -> 26/32
poor CA-band worsens: 18/32 -> 19/32
```

Compared with weak-collapse V3c:

```text
adjacent CA mean worsens: 3.382 -> 3.205
CA in-band worsens: 0.781 -> 0.704
radius worsens: 7.650 -> 7.416
collapse stays bad: 26/32 -> 26/32
poor CA-band worsens: 17/32 -> 19/32
```

The visual samples may still appear more relaxed or less tangled because the loss changes the character of generated chains. But the measured radius of gyration did not increase, and collapse count did not drop. Therefore the numeric evidence does not support calling this a successful anti-collapse improvement.

### Working conclusion

```text
The moderate V3c anti-collapse setting is a clean and useful negative result.
It confirms that simply increasing the radius-hinge loss from 0.0015 to 0.005 does not fix global collapse.
```

This does not invalidate the geometry-loss direction. The V3b/V3c family still shows that local geometry supervision can greatly improve adjacent CA validity relative to V2/V3. But it does suggest that the current radius-of-gyration hinge is not the right main mechanism for global structure.

### Recommendation

Stop spending major time tuning this exact V3c radius-hinge mechanism. The next step should be V4 EGNN, using the V3b/V3c lessons:

```text
keep best-checkpoint sampling
keep geometry losses on x0_pred
keep the structural metric suite
replace the flattened denoiser with a coordinate-aware/equivariant graph denoiser
```

The V3c story is now useful for the report: local losses help, simple global radius regularization does not reliably solve collapse, and this motivates moving to an architecture with a stronger geometric inductive bias.

## V4 implementation update: EGNN-style coordinate-aware denoiser

V4 has now been scaffolded as:

```text
notebooks/protein_backbone_diffusion_v4.ipynb
```

with artifacts directed to:

```text
results/v4/
```

### What changed

V4 starts from the clean V3b notebook structure, not V3c. The retained pieces are the same ones that made V3b a stable comparison point:

- the chunked train / chain-level validation / test data pipeline
- train-only coordinate normalization
- `B x L x 4 x 3` backbone representation with residue masking
- 100-step linear DDPM schedule
- geometry-augmented loss on `x0_pred`
- history tables, fixed-timestep diagnostics, generated-structure metrics, collapse summaries, and comparison tables

The main model change is architectural. The flattened denoiser has been replaced by `BackboneCoordinateEGNNDenoiser` in `src/latent_structure_generation/backbone_diffusion.py`.

### Current V4 model design

The V4 denoiser is intentionally small and sparse:

- nodes are backbone atoms `N, CA, C, O`
- internal coordinates stay as `B x L x 4 x 3`
- the public DDPM interface still uses flattened `(B, L, 12)` tensors so the V3b training and sampling code stays compatible
- node features use atom identity, learned residue position embeddings, normalized residue index, timestep embedding, and node-mask conditioning
- sparse edges include:
  - intra-residue `N-CA`
  - intra-residue `CA-C`
  - intra-residue `C-O`
  - adjacent-residue `C-N`
  - adjacent `CA-CA`
  - optional nonlocal `CA-CA` sequence-offset edges with defaults `+/-8`, `+/-16`, and `+/-32`
- EGNN-style blocks update hidden states from node features plus squared distances, and update coordinates through learned scalar weights on relative vectors
- predicted noise is read out as the masked coordinate delta between refined node coordinates and the input noisy coordinates

This keeps the model coordinate-aware and equivariant in the coordinate-update path without introducing dense all-pairs edges.

### Loss and training choices carried forward from V3b

The V4 notebook preserves the stable V3b objective exactly:

```text
noise_mse
+ 0.01 * bond_geometry_loss(x0_pred)
+ 0.01 * adjacent_ca_geometry_loss(x0_pred)
```

with:

```text
geometry_loss_type = Smooth L1
geometry_loss_beta = 0.5
geometry_max_timestep = 50
no radius-of-gyration anti-collapse loss
```

The notebook now also reloads the saved best-validation checkpoint before final sampling and evaluation, so the reported V4 sample metrics will match the checkpoint-selection rule.

### Intended comparison framing

The V4 report should be framed narrowly and defensibly:

- V2: the flattened DDPM trains but collapses
- V3: local atom-graph message passing alone was not enough
- V3b: explicit local geometry losses improve local backbone realism substantially
- V3c: simple radius-hinge anti-collapse loss does not reliably solve global collapse
- V4: test whether a coordinate-aware sparse EGNN-style denoiser can preserve the V3b local-geometry gains while improving radius of gyration and collapse counts

### Validation status in this workspace

Local execution here is still limited by the environment:

- `torch` is not installed in the available Python interpreter
- `nbformat` is not installed either

So the completed local checks were:

- notebook JSON rewrite completed cleanly
- code-cell compilation check is possible without execution
- `python -m json.tool` validation should be run on the final notebook

The missing runtime smoke checks in this workspace are:

- instantiate the V4 model with `torch`
- run one forward pass on a small batch
- compute one loss
- run one tiny sampling pass

Those should be done in the notebook runtime before trusting V4 training results.

## V4 troubleshooting update: make the EGNN a real DDPM predictor

After the first V4 smoke runs, the most likely failure mode was no longer "EGNNs are a bad idea." It was much narrower:

- the original V4 hidden state did not ingest raw noisy coordinates directly
- hidden-state updates depended mainly on static node metadata plus squared distances
- the final DDPM prediction was only `coords_refined - coords_input`
- that coordinate path had also been intentionally stabilized with bounded small updates and a zero-initialized final coordinate head

So V4 began training very close to an all-zero noise predictor and had no strong direct regression path analogous to the V3b flattened denoiser's explicit output head. That is a plausible explanation for the flat losses around `0.82` to `0.83`, high geometry losses, and visually dead samples.

### Concrete V4 fixes now applied

The shared module `src/latent_structure_generation/backbone_diffusion.py` and `notebooks/protein_backbone_diffusion_v4.ipynb` were updated with a minimal, high-value patch set:

- keep EGNN-style coordinate-aware message passing
- inject raw noisy node coordinates and coordinate norms into the initial node-state features
- keep the coordinate-refinement path, but stop using it as the only output
- add an explicit per-node noise head that predicts `(x, y, z)` noise from:
  - final hidden state
  - input noisy coordinates
  - learned coordinate residual
- combine:
  - hidden-state noise prediction
  - coordinate residual path
- relax the exact-zero coordinate-head init to a tiny random init so the coordinate branch is still stable but not perfectly dead
- reduce default nonlocal CA sequence-offset edges from `+/-8, +/-16, +/-32` to `+/-8, +/-16`
- reduce default V4 learning rate from `1e-3` to `5e-4`
- add debugging metrics to the training history and fixed-timestep diagnostics:
  - predicted noise RMS
  - `x_t` RMS
  - `x0_pred` RMS
  - near-zero prediction fraction
  - coordinate residual RMS
  - node-head RMS
  - per-layer coordinate-update RMS summary

### Why this is the right next test

This preserves what matters for comparability:

- same data splits
- same chunking
- same normalization
- same DDPM schedule
- same V3b objective
- no V3c radius loss
- same sampling and evaluation structure

But it removes the most obvious architectural handicap:

- V3b had an explicit supervised path from model internals to predicted noise
- the old V4 effectively forced all supervision through tiny coordinate displacements
- the patched V4 now has a direct noise head while still letting the EGNN coordinate branch contribute structural reasoning

So the next short run should answer a cleaner question:

- does the architecture learn once the prediction path is no longer bottlenecked?

### Practical next run

The notebook is now set up so the next V4 rerun should be done from training onward with the updated architecture and defaults:

- `BACKBONE_DIFFUSION_MODEL_HIDDEN_DIM=192` by default
- `BACKBONE_DIFFUSION_MODEL_NUM_LAYERS=4` by default
- `BACKBONE_DIFFUSION_SEQUENCE_OFFSET_EDGES=8,16` by default
- `BACKBONE_DIFFUSION_LEARNING_RATE=5e-4` by default
- `BACKBONE_DIFFUSION_GRAD_CLIP_NORM=0.5` by default

The first thing to inspect after rerunning is not just validation total loss. It is whether:

- train total starts decreasing in the first few epochs
- validation noise stops staying flat
- predicted noise RMS is non-trivial
- near-zero fraction is not stuck near `1.0`
- bond and adjacent-CA losses begin moving down

If those move in the right direction, V4 becomes a meaningful architecture experiment instead of a failed smoke run.
