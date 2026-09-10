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

## V4 troubleshooting continuation: what the debugging actually showed

After the first architecture patch, the next phase of V4 work was not "train once and declare success." It became a structured troubleshooting pass to separate three possibilities:

- the EGNN implementation is broken
- the EGNN implementation works, but the output parameterization is still too weak
- the implementation is basically viable, but full-run optimization is much harder than tiny-subset fitting

### First post-patch full-run behavior

The early post-patch full runs still looked discouraging.

One representative run with a more conservative setup showed:

```text
lr = 5e-4
grad_clip_norm = 0.5
sequence_offset_edges = (8, 16)
```

and produced:

```text
Epoch 01:
train total 0.75168
val total 0.72620
val noise 0.65217
val bond 4.1500
val CA 3.2528
train pred rms 0.3140
val pred rms 0.3257
val near-zero 0.005

Epoch 05:
train total 0.74087
val total 0.73755
val noise 0.66599
val bond 4.0166
val CA 3.1384
train pred rms 0.3203
val pred rms 0.3132
val near-zero 0.005
```

Interpretation:

- the model was no longer dead-zero
- near-zero fraction was already low, so the explicit output head had fixed the worst bottleneck
- but predicted-noise RMS was still stuck around `0.31` to `0.33`
- losses barely moved

So the old failure mode changed. V4 was no longer "not producing anything." It was producing a weak, low-amplitude, underpowered prediction.

### More aggressive full-run hyperparameters

A stronger full configuration was tried:

```text
lr = 1e-3
grad_clip_norm = 1.0
hidden_dim = 256
sequence_offset_edges = (8, 16, 32)
```

That made epochs slower, around `105s` on A100, but still did not materially fix the weak-learning pattern. The key observation was that `pred rms` stayed pinned near `0.32`, which argued against "just raise LR" as the main answer.

This led to a more disciplined debugging step:

- stop spending full-run compute
- test whether the model can overfit a tiny subset at all

### Overfit-debug mode was added

To make this practical, V4 gained an env-controlled overfit-debug mode in the notebook loader section. The purpose was:

- train on a tiny subset of train chunks
- optionally evaluate on that same train subset
- see whether V4 can overfit the denoising task and the geometry-augmented objective at all

There was some Colab friction while testing this:

- notebook-file changes in the local repo did not automatically affect the already-open Colab runtime
- shell-style `export` commands did not reliably propagate to later Python cells in Colab
- eventually the simplest reliable path was to hardcode the overfit settings in the notebook cell for diagnosis

Once overfit-debug was definitely active, epoch time dropped to about `1.0` to `1.3s`, confirming that the loader was really using the tiny subset.

### Noise-only overfit result

With geometry losses disabled:

```text
lambda_bond = 0.0
lambda_ca = 0.0
```

the tiny-subset run showed clear learning.

Representative behavior over 30 epochs:

- train total/noise dropped from about `0.997` to about `0.523`
- validation-on-train-subset noise dropped from about `0.993` to about `0.523`
- train predicted-noise RMS rose from about `0.05` to about `0.53`
- validation predicted-noise RMS rose to about `0.57`
- near-zero fraction stayed very low

Interpretation:

- V4 can learn the DDPM noise task
- the explicit hidden-state output head is working
- the architecture is not fundamentally broken

### Geometry-augmented overfit result

The next key test was whether the original V3b geometry objective was itself learnable in V4 overfit mode.

Using:

```text
lambda_bond = 0.01
lambda_ca = 0.01
sequence_offset_edges = (8, 16)
lr = 1e-3
grad_clip_norm = 1.0
```

the tiny-subset run also learned meaningfully.

Representative pattern:

- train total fell from about `1.11` to about `0.65`
- validation total fell from about `1.12` to about `0.64`
- validation noise fell from about `0.99` to about `0.58`
- predicted-noise RMS climbed from near zero to around `0.5` to `0.6`
- geometry losses were noisy, but not frozen

Interpretation:

- the geometry-augmented objective is not impossible for V4
- V4 can fit both denoising and geometry terms on a tiny subset
- therefore the old flatlined full-run behavior is much more likely a scaling / throughput / optimization-at-dataset-scale issue than a basic correctness bug

This was the reassuring evidence that the V4 redesign was not wasted work.

### First more-promising full-data signal

After turning overfit debug back off and running a lighter full-data configuration:

```text
lr = 1e-3
grad_clip_norm = 1.0
sequence_offset_edges = (8, 16)
lambda_bond = 0.01
lambda_ca = 0.01
```

the first full epoch produced a much healthier signal than the earlier flat runs:

```text
Epoch 01:
train total 0.25810
val total 0.15741
val noise 0.14786
val bond 0.2244
val CA 0.7307
train pred rms 0.8629
val pred rms 0.9068
val near-zero 0.001
epoch time 195.4s on L4
```

This no longer looks like the old weak-amplitude regime:

- predicted-noise RMS is now high rather than pinned near `0.32`
- noise loss is much lower
- geometry losses are much smaller
- near-zero fraction is tiny

The main problem at this stage is no longer obvious model failure. The main problem is runtime:

- one epoch on L4 took about `195s`

So the working interpretation became:

- V4 now appears viable enough to justify continued training
- the immediate bottleneck is wall-clock throughput, not the original learning bottleneck
- A100 is likely the right hardware for meaningful V4 iteration from this point onward

### Practical conclusions from the troubleshooting phase

The V4 debugging trail now supports a much more defensible story:

1. The original coordinate-delta-only prediction path was too constrained.
2. Adding a direct hidden-state noise head and raw-coordinate conditioning fixed the dead-output problem.
3. Tiny-subset overfit showed that:
   - the denoising task is learnable
   - the geometry-augmented objective is also learnable
4. Full-run difficulties therefore should not be interpreted as "V4 is broken."
5. The current remaining issue is training efficiency and stability at full scale, not obvious architectural invalidity.

### What should be monitored next

For the next more stable V4 runs, the most important metrics to watch are:

- train total loss
- validation total loss
- validation noise loss
- train and validation predicted-noise RMS
- near-zero prediction fraction
- bond geometry loss
- adjacent CA geometry loss

The reassuring pattern to look for is:

- predicted-noise RMS stays healthy rather than collapsing
- near-zero fraction stays very low
- train loss continues moving down
- validation noise does not flatten in the old `0.65` to `0.67` band
- geometry losses stay much smaller than in the original failed V4 smoke runs

If that pattern persists, V4 should be treated as a computationally expensive but now credible architecture test, not a failed branch.

## V4 first-iteration conclusion from the A100 full run

The first full V4 iteration that should be treated as the real reference run is the A100 run archived as:

```text
results/v4-20260607T212644Z-3-001.zip
```

This run used the now-stable patched V4 setup:

```text
hidden_dim = 192
num_layers = 4
sequence_offset_edges = (8, 16)
lr = 1e-3
grad_clip_norm = 1.0
lambda_bond = 0.01
lambda_ca = 0.01
geometry_loss_beta = 0.5
geometry_max_timestep = 50
no radius loss
device = NVIDIA A100-SXM4-80GB
```

### Training outcome

This run is the point where V4 should no longer be described as "not learning." It clearly learned.

Best epoch by validation total loss was epoch `28` with:

```text
train total = 0.1130
train noise = 0.1041
val total = 0.1011
val noise = 0.0990
val bond = 0.1135
val CA = 0.0921
test total = 0.1042
test noise = 0.1022
```

Important diagnostic values at the best epoch:

```text
train pred_noise_rms = 0.9450
val pred_noise_rms = 0.9448
test pred_noise_rms = 0.9423
train near_zero_fraction = 0.000843
val near_zero_fraction = 0.000843
test near_zero_fraction = 0.000881
```

Interpretation:

- the old low-amplitude failure mode is gone
- the model is no longer stuck near zero predictions
- the explicit hidden-state noise head is active and stable
- V4 can train end to end on the full dataset under the V3b objective

This is the main success of the first V4 troubleshooting iteration.

### Structural sample outcome

The sample-quality tables show that V4 materially improved on V3b in local geometry and collapse behavior.

Generated-sample summary for 32 validation-shaped masks:

```text
mean adjacent CA = 3.7403
adjacent CA in-band fraction = 0.8587
mean N-CA = 1.3903
mean CA-C = 1.4421
mean C-O = 1.1716
mean C-N = 1.4004
mean radius of gyration = 8.3477
collapse count = 10 / 32
poor CA-band count = 5 / 32
```

Real reference summary for the same mask set:

```text
mean adjacent CA = 3.8104
adjacent CA in-band fraction = 0.9986
mean radius of gyration = 16.2135
```

### Direct comparison against V3b

Compared with the current V3b reference:

```text
V3b mean adjacent CA = 3.1619
V4  mean adjacent CA = 3.7403

V3b adjacent CA in-band fraction = 0.7276
V4  adjacent CA in-band fraction = 0.8587

V3b mean radius of gyration = 7.8554
V4  mean radius of gyration = 8.3477

V3b collapse count = 19 / 32
V4  collapse count = 10 / 32

V3b poor CA-band count = 18 / 32
V4  poor CA-band count = 5 / 32
```

This is the most important scientific outcome of the first V4 iteration:

- V4 did not beat V3b on validation denoising loss
- but V4 did beat V3b clearly on the downstream structural metrics that mattered most in earlier analysis
- especially:
  - adjacent CA spacing
  - CA-band quality
  - collapse frequency

So V4 is not the best pure denoising model yet, but it is already the strongest architecture tried so far for sample geometry and anti-collapse behavior.

### Trade-off summary

The first V4 iteration established a real trade-off:

- V3b remains the stronger loss baseline
- V4 is much slower and more expensive to train
- but V4 produces substantially better local geometry and less collapse in generated samples

This means V4 is now a credible branch worth continuing, not a failed experiment.

For clarity: the best trained checkpoint in the V4 family remains the plain V4 A100 reference run. V4a and V4c are downstream diagnostics / sampling-guidance branches built on that line, not replacement training baselines that have already surpassed it.

### Honest limitations

The first V4 iteration is promising, but it is not final.

Remaining limitations:

- V4 denoising loss still trails V3b
- generated radius of gyration is still far below real structures
- global compactness is improved, but not solved
- V4 is computationally expensive:
  - around `75s` per epoch on A100
  - compared with much cheaper V3b training

So the correct conclusion is not "V4 solved the task." The correct conclusion is:

- the EGNN-style coordinate-aware denoiser is now validated as a meaningful architecture direction
- the first full V4 run successfully converted earlier debugging work into a real structural improvement over V3b
- the next V4 work should focus on improving efficiency and preserving the new geometry gains while trying to close the remaining denoising-loss gap

### Bottom-line claim for the notebook

The first V4 troubleshooting iteration succeeded.

It demonstrated that:

1. the original V4 failure was not fundamental
2. the patched V4 architecture can train stably on the full dataset
3. V4 materially improves structural sample quality over the V3b baseline
4. V4 is now strong enough to justify further iteration as a real candidate model rather than a speculative branch

## V4 follow-up decision: do not keep `(8, 16, 32)` as the default edge set

After the first successful V4 reference run with:

```text
sequence_offset_edges = (8, 16)
```

the most obvious next test was to add one more longer-range sparse CA edge:

```text
sequence_offset_edges = (8, 16, 32)
```

The motivation was reasonable:

- V4 already improved local geometry and collapse behavior
- the remaining weakness was still global compactness
- adding one more sparse nonlocal offset was the cleanest low-complexity test of whether extra sequence-range context would help

However, the early results argue against keeping this as the new default.

### Observed early run behavior

The `(8, 16, 32)` run produced:

```text
Epoch 01:
train total 0.26366
val total 0.16007
val noise 0.15002
val bond 0.2263
val CA 0.7781
train pred rms 0.8601
val pred rms 0.9154
epoch time 218.5s

Epoch 02:
train total 0.17130
val total 0.15090
val noise 0.14287
val bond 0.2017
val CA 0.6016
train pred rms 0.9153
val pred rms 0.9193
epoch time 218.9s

Epoch 03:
train total 0.16294
val total 0.14713
val noise 0.13976
val bond 0.1908
val CA 0.5459
train pred rms 0.9187
val pred rms 0.9289
epoch time 218.7s
```

### Comparison against the `(8, 16)` reference run

The key point is not that `(8, 16, 32)` failed completely. It did not.

The key point is that:

- the optimization behavior is in roughly the same regime as the successful `(8, 16)` run
- but runtime is dramatically worse

Reference `(8, 16)` A100 run:

- around `75s` per epoch

Observed `(8, 16, 32)` A100 run:

- around `219s` per epoch

So adding the `32` offset made the model roughly three times slower, without immediate evidence of a comparably large quality gain.

### Practical conclusion

At this stage, `(8, 16, 32)` does not look like the right default trade-off for V4.

The current working decision should be:

- keep `(8, 16)` as the practical V4 default edge set
- treat `(8, 16, 32)` as useful negative evidence about scaling cost
- only revisit richer offset sets later if there is a much stronger reason or if runtime becomes less important

This is an important outcome because it prevents the V4 branch from drifting into "more edges must be better" thinking.

The evidence so far suggests:

- a small amount of sparse nonlocal context helps
- too much additional fixed sequence-offset connectivity becomes expensive very quickly
- the current V4 value comes from improved structural quality at a still-manageable training cost
- `(8, 16)` is therefore the better default for continuing iteration

## V4a early archive inspection: diagnostics scaffold present, but not yet a real anti-collapse experiment

Reviewed archive:

- `results/v4a-20260607T232548Z-3-001.zip`

### What the archive clearly shows

The V4a notebook scaffolding is present:

- saved training history
- saved timestep diagnostics
- saved V4a run config
- saved small preview real/generated evaluation tables

However, the archive does **not** yet contain the full guided-sampling comparison artifacts that V4a was intended to produce.

No files corresponding to guided conditions or pooled nonlocal-compactness summaries were present in the zip. In practice, this means the run stopped after the training / preview stage rather than completing the actual anti-collapse sampling comparison.

### Configuration mismatch versus the working V4 reference

The saved V4a run config shows that this run was **not** using the settled practical V4 baseline defaults.

Saved V4a config from the archive:

- `model_hidden_dim = 256`
- `model_num_layers = 4`
- `model_sequence_offset_edges = 8,16,32`
- `learning_rate = 1e-3`
- `grad_clip_norm = 1.0`
- `lambda_bond = 0.01`
- `lambda_ca = 0.01`
- `geometry_loss_beta = 0.5`
- `geometry_max_timestep = 50`
- `use_nonlocal_ca_guidance_default = False`
- `guidance_threshold_angstrom_default = 8.0`
- `guidance_scale_default = 0.0`

This matters because the successful practical V4 reference was:

- `hidden_dim = 192`
- `sequence_offset_edges = (8, 16)`

So this archive represents a heavier `(256, 8/16/32)` training variant plus preview evaluation, not a clean "V4 baseline + V4a guidance" experiment.

### Preview metrics from the archive

The preview generated-vs-real mean table showed:

- generated mean adjacent CA: `3.483`
- generated adjacent-CA in-band fraction: `0.511`
- generated radius of gyration: `7.737`

Real preview mean table showed:

- real mean adjacent CA: `3.813`
- real adjacent-CA in-band fraction: `0.999`
- real radius of gyration: `16.901`

These preview generated metrics are poor and strongly compacted. They are also noticeably worse than the full successful V4 reference run.

### Why this does not invalidate V4

This archive should not be interpreted as "V4a disproved the V4 progress".

Instead, the more accurate interpretation is:

- the run only covered the training + preview stage
- the guidance default was effectively off (`guidance_scale_default = 0.0`)
- the model config drifted away from the practical V4 reference
- the archive therefore does not answer the intended V4a question about whether sampling-time nonlocal CA guidance helps collapse

### Working conclusion

The main value of this archive is diagnostic:

- V4a notebook plumbing for additional compactness analysis is present
- but this specific run is not yet the decisive anti-collapse experiment

The next clean V4a attempt should preserve the working V4 reference first:

- `hidden_dim = 192`
- `sequence_offset_edges = (8, 16)`
- same V4 objective and optimizer settings

Then V4a should differ only by:

- extra compactness diagnostics
- explicit baseline-vs-guided sampling conditions
- saved nonlocal-distance / close-contact comparison tables

Until that is done, this archive should be treated as an early V4a smoke run rather than a conclusive collapse-mitigation result.

## V4a first full guided-sampling comparison: guidance effect is real but extremely small

Reviewed archive:

- `results/v4a-20260608T001606Z-3-001.zip`

This is the first V4a archive that actually contains the intended guided-sampling outputs:

- `v4a_guidance_comparison_summary`
- `v4a_nonlocal_sequence_separation_summary`
- `v4a_global_diagnostics_*`
- per-condition `v4a_condition_*_eval_metrics`

### Configuration used

Saved run config:

- `model_hidden_dim = 192`
- `model_sequence_offset_edges = 8,16`
- `learning_rate = 1e-3`
- `grad_clip_norm = 1.0`
- `lambda_bond = 0.01`
- `lambda_ca = 0.01`
- `guidance_sequence_separation_default = 8`
- `guidance_threshold_angstrom_default = 10.0`
- `guidance_scale_default = 6e-4`
- `guidance_start_timestep_default = 75`
- `guidance_end_timestep_default = 10`

So the V4a analysis finally ran on the intended practical V4 baseline architecture.

### Training-side status

The denoiser itself still looks healthy:

- best validation total loss about `0.1375`
- best validation noise about `0.1329`
- test total about `0.1386`
- test noise about `0.1340`
- validation/test predicted-noise RMS about `0.95`

This again supports the interpretation that the remaining issue is not training collapse of the denoiser, but global structure collapse during sampling.

### Baseline versus guided compactness results

The strongest result from this archive is also the most sobering:

- the guided conditions moved the global compactness metrics in the desired direction
- but the effect size is extremely small

Baseline (`baseline_v4`) summary:

- collapse count: `29/32`
- poor adjacent-CA-band count: `32/32`
- radius of gyration mean: `7.8017`
- radius of gyration median: `7.8272`
- max pairwise CA distance mean: `27.7737`
- end-to-end CA distance mean: `15.3426`
- fraction nonlocal CA pairs below `8A`: `0.310184`
- fraction nonlocal CA pairs below `10A`: `0.487422`

Strongest guided condition tested (`guided_thr10_scale1e-3`):

- collapse count: `29/32`
- poor adjacent-CA-band count: `32/32`
- radius of gyration mean: `7.8064`
- radius of gyration median: `7.8352`
- max pairwise CA distance mean: `27.7807`
- end-to-end CA distance mean: `15.3478`
- fraction nonlocal CA pairs below `8A`: `0.309481`
- fraction nonlocal CA pairs below `10A`: `0.486625`

So the direction is technically correct:

- radius increases slightly
- end-to-end distance increases slightly
- max pairwise CA distance increases slightly
- nonlocal close-contact fractions decrease slightly

But the magnitude is tiny:

- collapse count unchanged
- poor adjacent-CA-band count unchanged
- nonlocal compactness metrics improve only at the third decimal place

### Interpretation

This archive is useful because it rules out one important possibility:

- a small sampling-time nonlocal CA repulsion does **not** produce a meaningful anti-collapse effect by itself

That does not mean the V4a idea is wrong. It means the current intervention is too weak, too indirect, or both.

The most likely explanations are:

1. the current guidance scales are still too small relative to the DDPM update magnitude
2. the repulsion term is too diffuse because it averages over many nonlocal pairs
3. the reverse process may already be committed to compact structures before the current guidance meaningfully changes the trajectory
4. radius/global extent is still underconstrained at training time, so sampling-time guidance alone has limited leverage

### Practical conclusion for next iteration

This result does **not** justify running more epochs of the same training setup just to solve collapse.

Why:

- the guidance experiment reuses the checkpoint
- the failure mode is still in generated global structure
- the current evidence suggests the bottleneck is intervention strength/logic, not lack of denoiser convergence

So the better next step is:

- do **not** prioritize more epochs first
- do **not** change the core V4 training hyperparameters again yet
- instead, strengthen or redesign the collapse intervention itself

### Recommended next direction

The cleanest next tests should be one of:

1. a much stronger sampling-time guidance sweep

- try substantially larger scales than `1e-3`
- potentially test `3e-3` or even `1e-2` very carefully
- possibly start guidance earlier in the reverse process

2. a more targeted global penalty

- instead of a weak averaged nonlocal repulsion over many pairs
- use a sharper penalty on the worst close-contact tail
- or target per-structure compactness statistics more directly

3. only after that, consider a new training-time global loss

- but avoid simply resurrecting the old V3c radius hinge unchanged
- if a training-time intervention is added, it should be more diagnostic and global-distance-aware than a crude radius threshold alone

### Working judgment

V4a first-iteration result:

- excellent diagnostic progress
- clear evidence that the existing anti-collapse guidance is too weak to matter materially
- no evidence yet that more epochs of the same setup would solve the collapse problem

So the next iteration should be a **stronger or sharper collapse intervention**, not just longer training.

## V4a aggressive upper-bound sweep: stronger guidance finally moves collapse, but only modestly

Reviewed archive:

- `results/v4a-20260608T003204Z-3-001.zip`

This run finally tested a genuinely more aggressive sampling-time anti-collapse sweep:

- baseline
- `guided_thr10_scale1e-3`
- `guided_thr12_scale3e-3`
- `guided_thr14_scale1e-2`

Default run config:

- `hidden_dim = 192`
- `sequence_offset_edges = (8, 16)`
- `guidance_threshold_angstrom_default = 12.0`
- `guidance_scale_default = 3e-3`
- `guidance_start_timestep_default = 75`
- `guidance_end_timestep_default = 10`

### Training-side status remains healthy

The denoiser is still behaving well:

- best validation total about `0.1361`
- best validation noise about `0.1319`
- test total about `0.1370`
- test noise about `0.1328`

So again, the remaining limitation is not obvious training instability.

### Stronger guidance does finally matter

This is the first V4a result where the stronger sampling-time guidance produces a clearly visible movement in the global compactness metrics.

Baseline (`baseline_v4`) summary:

- collapse count: `26/32`
- poor adjacent-CA-band count: `32/32`
- radius of gyration mean: `7.9069`
- radius of gyration median: `7.9997`
- max pairwise CA distance mean: `28.7562`
- end-to-end CA distance mean: `15.7018`
- fraction nonlocal CA pairs below `8A`: `0.301575`
- fraction nonlocal CA pairs below `10A`: `0.476021`

Most aggressive condition (`guided_thr14_scale1e-2`) summary:

- collapse count: `20/32`
- poor adjacent-CA-band count: `32/32`
- radius of gyration mean: `8.0617`
- radius of gyration median: `8.1115`
- max pairwise CA distance mean: `29.0326`
- end-to-end CA distance mean: `15.8969`
- fraction nonlocal CA pairs below `8A`: `0.283823`
- fraction nonlocal CA pairs below `10A`: `0.454770`

So compared with the earlier weak sweep, the stronger settings finally produce a nontrivial global expansion signal:

- collapse count improves from `26` to `20`
- radius of gyration increases by about `0.155`
- max pairwise CA distance increases by about `0.276`
- end-to-end CA distance increases by about `0.195`
- nonlocal close-contact fractions drop by a visible amount

### But the effect is still limited

Even with the strongest tested condition:

- `20/32` structures are still flagged as collapsed
- `32/32` still fail the adjacent-CA-band quality threshold in this V4a evaluation
- the nonlocal distance distribution is still far from real structures

Real reference at sequence separation `> 8`:

- fraction nonlocal CA pairs below `8A`: `0.015950`
- fraction nonlocal CA pairs below `10A`: `0.043257`
- pooled nonlocal CA distance mean: `23.793`

Best guided condition:

- fraction nonlocal CA pairs below `8A`: `0.295817`
- fraction nonlocal CA pairs below `10A`: `0.468807`
- pooled nonlocal CA distance mean: `10.720`

So although guidance is helping, the model is still producing structures that are globally far too compact.

### Interpretation

This run clarifies the situation:

1. the current V4a guidance mechanism is capable of pushing structures outward
2. guidance scale was indeed previously too low
3. but even a much stronger repulsion does not come close to fixing the compactness gap

That strongly suggests something else is also at play:

- the reverse process has a structural prior toward compact states that guidance alone only partially counteracts
- the current repulsion energy is still too diffuse because it averages over many nonlocal pairs
- and/or the denoiser itself has not learned a sufficiently realistic global shape prior

### Practical next-step conclusion

This run argues **against** simply increasing epochs first.

Why:

- the denoiser is already stable
- stronger guidance can move the samples
- but the remaining compactness gap is still large even with aggressive scales

So the next improvement should probably **not** be "same model, more epochs".

The more promising next directions are:

1. sharpen the guidance logic

- penalize the worst close-contact tail more aggressively instead of averaging over all too-close pairs
- consider a barrier-style or quantile-focused energy

2. consider adding a training-time global loss

- not the old crude V3c radius hinge unchanged
- but something that explicitly teaches a less compact nonlocal distance distribution

3. use the current aggressive sweep as an upper-bound probe

- it shows the mechanism can work
- but also shows that simple scale increases alone are unlikely to fully solve collapse

### Working judgment

The data no longer supports "guidance scale is definitely too low" as the whole story.

A stronger scale helped, but not enough.

So the next real improvement should come from **changing the shape of the anti-collapse objective/intervention**, not just turning the same knob further.

## V4a more aggressive sweep: stronger guidance now gives a noticeable collapse reduction

Reviewed archive:

- `results/v4a-20260608T003204Z-3-001.zip`

This run increased the sweep further:

- `guided_thr10_scale1e-3`
- `guided_thr12_scale3e-3`
- `guided_thr14_scale1e-2`

### Core result

This is the first V4a run where the strongest guided condition produces a clearly nontrivial collapse improvement, not just a third-decimal movement.

Baseline (`baseline_v4`):

- collapse count: `26/32`
- radius of gyration mean: `7.9069`
- radius of gyration median: `7.9997`
- max pairwise CA distance mean: `28.7562`
- end-to-end CA distance mean: `15.7018`
- fraction adjacent CA in band mean: `0.5851`
- bond-target MAE mean: `0.1341`
- fraction nonlocal CA pairs below `8A`: `0.301575`
- fraction nonlocal CA pairs below `10A`: `0.476021`

Strongest condition (`guided_thr14_scale1e-2`):

- collapse count: `20/32`
- radius of gyration mean: `8.0617`
- radius of gyration median: `8.1115`
- max pairwise CA distance mean: `29.0326`
- end-to-end CA distance mean: `15.8969`
- fraction adjacent CA in band mean: `0.5858`
- bond-target MAE mean: `0.1328`
- fraction nonlocal CA pairs below `8A`: `0.283823`
- fraction nonlocal CA pairs below `10A`: `0.454770`

### What improved

Compared with baseline:

- collapse count improved from `26` to `20`
- radius of gyration mean improved by about `0.155`
- max pairwise CA distance mean improved by about `0.276`
- end-to-end CA distance mean improved by about `0.195`
- fraction nonlocal CA pairs below `8A` improved by about `0.0178`
- fraction nonlocal CA pairs below `10A` improved by about `0.0213`

That is finally a visible anti-collapse effect.

### What did not improve enough

Even with the strongest condition:

- `20/32` chains are still classified as collapsed
- the compactness gap versus real structures remains huge
- the model is still massively overpopulating short nonlocal CA distances

Real reference at sequence separation `> 8`:

- fraction nonlocal CA pairs below `8A`: `0.015950`
- fraction nonlocal CA pairs below `10A`: `0.043257`
- pooled nonlocal CA distance mean: `23.793`

Best guided V4a condition:

- fraction nonlocal CA pairs below `8A`: `0.295817`
- fraction nonlocal CA pairs below `10A`: `0.468807`
- pooled nonlocal CA distance mean: `10.720`

So the anti-collapse guidance is now clearly helping, but the generative distribution is still far from realistic global geometry.

### Important interpretation

This run strongly suggests that two things are true at once:

1. stronger guidance was necessary
2. stronger guidance alone is still not sufficient

That means something deeper is at play than just "the scale was too low".

Most likely:

- the model prior itself still prefers overly compact structures
- the current guidance energy is still too averaged and diffuse
- a nontrivial fraction of the reverse trajectory may already be locked into compact states before this correction can fully unwind them

### What this implies for next iteration

This result does not justify simply increasing epochs first.

It also weakens the case for endlessly increasing the same guidance scale, because:

- stronger guidance now clearly works
- but even at `14A / 1e-2` the improvement is only partial

So the next step should probably be:

- keep the insight that stronger guidance helps
- but change the intervention **shape**, not only the magnitude

The most promising next move is:

- replace the current averaged repulsion with a sharper close-contact penalty
- for example, focus on the worst nonlocal contacts or the lower-distance tail rather than averaging all too-close pairs together

Only after trying that should a new training-time global loss be considered.

### Working judgment

This V4a sweep is important because it shows the anti-collapse guidance is not a dead end.

However, it is also strong evidence that the current formulation is too blunt.

The next improvement should come from a **more targeted anti-collapse energy**, not from more epochs of the same setup.

## V4c implementation note: tail-focused anti-collapse guidance and current notebook bottleneck

The next notebook branch was split out as V4c rather than continuing to mutate V4a in place. The reason was methodological rather than cosmetic:

- V4 should remain the working architecture/training reference
- V4a should remain the first global-diagnostics plus simple-guidance branch
- V4c should isolate the next intervention: a sharper sampling-time anti-collapse energy

V4c therefore keeps the same working V4 training setup and checkpoint-loading path, but replaces the old averaged nonlocal repulsion with a tail-focused guidance option.

### V4c intervention that was added

The key new idea in V4c is that the old guidance energy was likely too diffuse. It averaged too many nonlocal close-contact violations together, so even large guidance scales only weakly emphasized the worst compactness failures.

V4c therefore adds a new guidance mode:

- `tail_topk_barrier`

Conceptually this does the following:

- compute nonlocal CA-CA close-contact violations
- keep only the positive violations
- focus only on the worst fraction of them, rather than all of them equally
- apply a steeper barrier-style penalty to that worst-contact tail

The intent is to make the sampling correction act more like "push apart the worst illegal contacts first" instead of "slightly raise the average spacing of all too-close nonlocal pairs."

This is a more targeted anti-collapse intervention than the V4a mean-hinge style guidance.

### Current V4c practical issue

The newest blocker is not obviously model instability. It looks like a notebook-level memory problem.

On the VM currently being used, system RAM reached about `11.5 / 12.7 GB` and the notebook effectively could not get through the first full evaluation cycle cleanly. The important observation is that this appears to be **CPU RAM pressure**, not primarily GPU memory exhaustion.

Why that diagnosis is plausible:

- V4c does not only train the denoiser
- it also evaluates multiple guided sampling conditions in one pass
- for each condition it keeps generated coordinates, per-structure diagnostic tables, and pooled nonlocal distance arrays
- it then recomputes and concatenates additional pairwise-distance summaries across conditions

So the most likely cause is that the notebook is holding too much evaluation state in memory at once.

This is especially plausible because V4c currently does all of the following together:

- baseline sampling
- multiple guided-condition sampling runs
- full global compactness diagnostics
- pooled nonlocal CA distance summaries
- per-condition saved tables
- aggregate comparison tables

That is a poor fit for a small `~13 GB` RAM VM even if the GPU itself is acceptable.

### Important interpretation of the RAM issue

This should not be misread as:

- "the V4 model is too big"
- "the EGNN is broken again"
- "we necessarily need a better GPU first"

The evidence so far points more toward:

- the V4/V4a/V4c checkpoint path is workable
- the anti-collapse experimentation is now bottlenecked by notebook evaluation design and memory usage

An H100 would help throughput, but it would not by itself fix an avoidable CPU-memory blowup from storing too many per-condition tensors and pooled distance arrays simultaneously.

### Concrete suggestions to fix the V4c notebook bottleneck

The first fixes should be pragmatic and low-risk:

1. reduce sample count for guidance comparison

- default `V4C_SAMPLE_COUNT` should be cut substantially for triage
- for example, from `32` down to `8` or `12`

2. reduce sampling batch size

- lower `V4C_SAMPLE_BATCH_SIZE`
- for example, from `8` to `4`

3. reduce the default number of guidance conditions

- do not run a large sweep by default on a small VM
- baseline plus one representative sharp-tail condition is enough for the next quick decision

4. stop storing full generated coordinates for every condition unless explicitly needed

- compute condition metrics
- save condition outputs
- release large tensors before moving on

5. avoid retaining pooled nonlocal distance arrays for every condition at once

- summarize them per condition
- write the summary
- free the raw arrays

6. if needed, set notebook dataloader workers conservatively on small VMs

- worker count is probably not the main bottleneck
- but reducing worker/process overhead is still sensible on low-RAM machines

### Working judgment on V4c so far

V4c is still the right conceptual next step.

The previous V4a results already showed:

- stronger guidance can reduce collapse somewhat
- but average repulsion alone is too blunt

So testing a sharper tail-focused barrier is justified.

The current obstacle is mostly operational:

- V4c needs to be made lighter and more sequential in how it evaluates conditions

That should be done before drawing scientific conclusions from failed or incomplete V4c runs on constrained hardware.

## One-day plan for the final project window

There is effectively one productive day left, so the goal should not be "explore everything." It should be to produce one clean, defensible final result and one clean next-step recommendation.

### Main objective for the final day

Produce a compact, reproducible V4c result that answers:

- does tail-focused guidance improve collapse metrics more than the V4a mean-guidance baseline
- without obviously destroying local backbone quality

That is a concrete and reportable question.

### Recommended plan

1. simplify V4c so it can run reliably on available hardware

- reduce default sample count
- reduce default sample batch size
- reduce default condition sweep to:
  - `baseline_v4`
  - one legacy mean-guidance comparison
  - one representative tail-guidance comparison

2. rerun V4c from the existing good V4 checkpoint rather than retraining

- no more architecture retraining unless absolutely necessary
- the remaining question is sampling-time collapse control

3. evaluate only the most decision-relevant outputs

- collapse count
- radius of gyration
- max pairwise CA distance
- end-to-end CA distance
- fraction nonlocal CA pairs below `8A`
- fraction nonlocal CA pairs below `10A`
- adjacent CA in-band fraction
- bond-target MAE

4. choose one final conclusion

Either:

- tail-focused guidance is a real improvement over mean guidance and should be presented as the best current anti-collapse intervention

or:

- even sharper sampling-time guidance only partially helps, which supports the conclusion that a future training-time nonlocal/global objective is needed

Both conclusions are scientifically usable. The failure mode is now measurable and the intervention path is clear.

### What should not be done in the final day

- do not restart broad architecture tuning
- do not spend the last day rerunning many-epoch training just to chase validation loss
- do not add a large new training-time loss unless V4c becomes completely unworkable

### Best-case end-of-project outcome

The best realistic final outcome is:

- V4 remains the validated architecture improvement over V3b for structural sample quality
- V4a shows that simple nonlocal guidance helps only weakly
- V4c shows whether a sharper tail-focused guidance improves collapse more meaningfully
- the report can then conclude with a clear and defendable next-step recommendation:
  - future work should target global nonlocal compactness more directly, likely with a sharper sampling-time control or a training-time nonlocal structural objective
## Review: stop, diagnose, and restart V4d with a clearer collapse hypothesis.

This review was added after stepping back from the V4, V4a, and V4c collapse-control attempts. The purpose is to separate what has genuinely improved from what is still failing, so the next experiment is methodical rather than another blind hyperparameter tweak.

### Current model framing.

The current V4 line is best described as a DDPM-style backbone-coordinate diffusion model with an EGNN-style coordinate-aware denoiser. The diffusion process is still the generative framework: clean backbone coordinates are noised, the model predicts the added noise, and sampling starts from Gaussian noise before iteratively denoising. The EGNN-style network is the denoiser inside that framework.

### What V4 fixed.

V4 is not a failed branch. It is the strongest structural branch so far.

Relative to the 100-epoch stable V3b reference, V4 improved the generated structural metrics that mattered most for local sample quality:

| Metric | V3b 100 ep | V4 reference |
|---|---:|---:|
| Adjacent Cα mean | 3.162 Å | 3.740 Å |
| Adjacent Cα in-band fraction | 0.728 | 0.859 |
| Radius of gyration | 7.855 Å | 8.348 Å |
| Collapse count | 19/32 | 10/32 |
| Poor Cα-band count | 18/32 | 5/32 |

The important interpretation is that V4 improved the sample geometry despite not having the best pure denoising loss. V3b remains stronger on validation noise loss, but V4 produces better generated structures. Therefore, downstream structural metrics are more informative than noise loss alone for this project.

### What remains wrong.

The remaining failure is now more specific. Earlier versions had both broken local trace geometry and global collapse. V4 substantially improves local Cα continuity and bond-like geometry, but the samples are still globally too compact.

The real validation-shaped structures have a mean radius of gyration around 16.2 Å, whereas V4 generated samples remain around 8.35 Å. This means V4 sits close to the collapse threshold even when the adjacent Cα trace looks much better.

This suggests the model has learned a better local backbone but has not learned the correct nonlocal/global spatial distribution.

### Why the current objective is incomplete.

The current local geometry objective penalises:

- intra-residue `N-CA` distance.
- intra-residue `CA-C` distance.
- intra-residue `C-O` distance.
- adjacent-residue `C-N` distance.
- adjacent `CA-CA` distance.

These terms are useful, but they mainly teach local backbone continuity and local stereochemical plausibility. They do not directly teach the model how far apart sequence-distant residues should be in 3D space.

This is the core missing signal. The model can make a locally plausible chain that is still globally over-compressed.

### Diagnosis versus fix.

Distance maps and contact maps should first be used as diagnostics, not immediately as a new dense training target.

For generated samples, it is not fair to expect one generated contact map to match one exact validation contact map, because the model is sampling unconditionally with validation-shaped masks rather than reconstructing a specific validation fold. The fairer question is whether generated samples have distance/contact-map statistics that resemble real structures of similar length.

Useful diagnostics now include:

- Cα distance maps for a few real and generated examples.
- Cα contact maps at 8 Å and 10 Å.
- Nonlocal contact fractions for sequence separations such as `|i-j| > 8`, `>16`, and `>32`.
- Reverse-trajectory compactness summaries to see when collapse appears during sampling.

These diagnostics should make the remaining failure visually and quantitatively clear: generated structures overpopulate short nonlocal Cα distances.

### Why not use full contact-map training immediately.

A full contact-map loss is not the best next step. Contact maps bin continuous distances into contact/non-contact labels, which loses useful information. A pair at 9 Å and a pair at 30 Å are both non-contacts at an 8 Å threshold, but they mean very different things for global fold extent.

A full distance-map loss is more informative but may be expensive and over-constraining if applied densely across all residue pairs.

The more controlled next step is sparse nonlocal Cα distance supervision.

### V4d hypothesis.

V4d should keep the V4 architecture and stable local geometry objective, then add a small sparse nonlocal Cα distance loss during training.

The intended training objective is:

```text
noise_mse
+ 0.01 * local_bond_geometry_loss
+ 0.01 * adjacent_ca_geometry_loss
+ lambda_nonlocal_ca * sparse_nonlocal_ca_distance_loss
```

The new sparse nonlocal term should:

- compare predicted clean `x0_pred` Cα-Cα distances against true clean `x0` Cα-Cα distances.
- use only valid residue pairs.
- use only sequence-distant pairs, initially `|i-j| > 8`.
- sample a limited number of pairs per batch, for example 512 to 2048.
- use Smooth L1 loss rather than plain MSE.
- start with a small weight, for example `lambda_nonlocal_ca = 0.002`.
- apply mainly at mid/high timesteps, for example `t >= 25`, because global structure formation is a mid/high-noise problem.

This is more principled than a radius hinge because it does not force every protein toward one generic radius. It teaches the model the nonlocal distance pattern of the actual clean training example.

### What V4d should prove or disprove.

V4d should answer one focused question:

```text
Can sparse nonlocal Cα distance supervision reduce global over-collapse while preserving the local geometry gains of V4?
```

The primary comparison should be V4d versus V4, not only V4d versus V3b.

Success would mean:

- collapse count drops below V4's 10/32.
- radius of gyration increases above V4's approximately 8.35 Å.
- nonlocal Cα close-contact fractions decrease.
- adjacent Cα in-band fraction remains close to the V4 value of approximately 0.86.
- bond means do not degrade severely.
- validation/test noise loss remains stable enough to trust the model.

A negative result would still be useful if it shows that sparse nonlocal supervision either fails to move global compactness or damages local geometry. That would support a future shift towards sharper sampling-time guidance or a richer global structure prior.

### Immediate coding direction.

The next notebook should be `protein_backbone_diffusion_v4d.ipynb`, copied from V4. It should preserve the V4 denoiser and local geometry losses, set `results/v4d/` as the artifact output, add sparse nonlocal Cα distance supervision to the training and fixed-timestep evaluation loops, and add reverse-trajectory collapse diagnostics plus lightweight Cα distance/contact-map figures.

This is the most defensible next experiment because it directly targets the remaining failure mode rather than increasing local geometry losses, adding sequence conditioning, or continuing to tune radius penalties blindly.

## Review: V4d sparse nonlocal Cα supervision.

At this stage, the most useful decision is to stop and separate what has been fixed from what remains unsolved.

The current best modelling line is the V4 family: a DDPM-style backbone-coordinate diffusion model with an EGNN-style coordinate-aware denoiser. V4 was a major improvement over the earlier V3b geometry-aware baseline. It preserved much stronger local backbone geometry, improved adjacent Cα spacing, reduced poor Cα-band failures, and reduced collapse compared with the previous flattened denoiser. However, V4 still remained globally over-compact: generated structures had radius of gyration around 8 Å, compared with around 16 Å for real validation structures.

This motivated V4d. The purpose of V4d was not to redesign the architecture again, but to test whether the remaining collapse problem could be addressed more directly with sparse nonlocal Cα distance supervision. The idea was that local geometry losses teach the model how to form a continuous backbone trace, but they do not teach the model enough about the distribution of distances between residues far apart in sequence. V4d therefore preserved the V4 local geometry objective and added a small additional Smooth L1 loss comparing predicted clean Cα–Cα distances against true clean Cα–Cα distances for randomly sampled nonlocal residue pairs.

The 200-epoch overfit-debug run was useful as a stress test. It showed that the nonlocal Cα objective is active and learnable. The model could strongly reduce the nonlocal distance loss and push structures out of the collapsed regime. However, the same run also showed an important failure mode: if the nonlocal objective dominates on a tiny repeated subset, the model can over-expand structures and destroy local backbone continuity. The generated structures became globally spread out, but adjacent Cα distances and backbone bond lengths became unrealistic. This proved that the intervention has power, but also that it needs to be balanced carefully.

The full-data 30-epoch V4d run was therefore the more important experiment. It trained normally: validation total and noise losses decreased, predicted-noise RMS stayed healthy, and the model did not collapse into a trivial predictor. The local geometry terms improved strongly during training, especially the adjacent Cα loss and bond-length loss. In contrast, the sparse nonlocal Cα loss decreased only modestly. This suggests that the nonlocal supervision was active, but too weak or too indirect to dominate the learned reverse process.

Structurally, V4d was much better than the earlier V3b baseline, but it did not beat the current V4 reference. It preserved strong local backbone quality and reduced poor Cα-band failures, but it did not improve global compactness relative to V4. Radius of gyration remained around 8 Å rather than moving toward the real mean around 16 Å, and collapse count was worse than the V4 reference. The generated structures were not too stretched out; they were still under-expanded compared with real structures.

The reverse-trajectory diagnostics helped clarify the failure mode. During sampling, radius of gyration and adjacent Cα quality improved as denoising progressed. The model did not appear to collapse late. Instead, it started from a globally compact configuration and gradually cleaned up local geometry, but plateaued at a radius that was still much smaller than real proteins. This supports the interpretation that the low-noise local geometry regime is good at repairing backbone continuity, but does not create realistic global extent by itself.

The key conclusion from V4d is therefore mixed but valuable. Sparse nonlocal Cα distance supervision is conceptually well-motivated and learnable, but the current weak implementation does not overcome the compact global prior of the generator. Increasing the nonlocal signal too much can also damage local geometry, as shown by the overfit-debug run. The next improvement should therefore not be another blind architecture change or an aggressive radius penalty. It should be a controlled test of stronger, better-timed global supervision, ideally applied mainly in the high-noise or mid-noise regime where global structure is being formed, while keeping the low-noise regime focused on local backbone refinement.

For the final write-up, V4 remains the strongest current model. V4d should be presented as a thoughtful follow-up experiment: it diagnosed the remaining collapse problem as a nonlocal/global geometry issue, tested a targeted intervention, and showed that weak sparse nonlocal supervision alone is insufficient. This is still useful scientific progress because it narrows the next-step hypothesis: future work should target global compactness more directly, but without sacrificing local backbone validity.

## Review: V4e and V4f contact-tail experiments.

After V4d showed that exact sparse nonlocal Cα distance matching had limited benefit, I tested two softer contact-tail objectives to target the remaining collapse problem more directly.

V4e replaced exact nonlocal distance matching with a soft nonlocal contact-tail loss. Instead of asking the model to match exact Cα–Cα distances between sequence-distant residues, V4e compared soft contact fractions at 8 Å and 10 Å. This was inspired by the idea that collapse is visible as an excess of short nonlocal contacts, rather than every long-range distance being equally wrong. V4e did move the nonlocal compactness metrics in the right direction: mean nonlocal Cα distance increased slightly, short nonlocal contact fractions decreased, and collapse count improved relative to the weaker V4d run. However, this came at a clear cost to local backbone continuity. Adjacent Cα in-band fraction dropped substantially, and poor Cα-band failures increased. Therefore, V4e supported the diagnosis that collapse is a global/nonlocal problem, but it did not provide a better final model than V4.

V4f then refined the contact-tail idea into a one-sided excess-contact loss. The aim was to penalise the model only when it produced too many short nonlocal Cα contacts relative to the real structure, rather than symmetrically matching contact fractions. This was intended to avoid punishing legitimate tertiary contacts. However, the generated samples were worse than V4 and worse than V4e. The excess-contact loss remained very small during training, and it did not transfer into better free sampling. Collapse count increased, adjacent Cα in-band fraction fell, and all generated samples failed the poor Cα-band criterion.

The combined lesson from V4d, V4e, and V4f is that training-time global/nonlocal losses have some conceptual leverage, but they are not transferring cleanly into better iterative samples. Exact nonlocal distance matching is too fold-specific and blunt; symmetric contact-tail matching moves the right metrics but damages local geometry; one-sided excess-contact matching is too weak in practice. The failure mode now appears less like “the loss is missing one more term” and more like “the free sampling process needs to be steered or constrained directly”.

This motivates the next experiment: V4g should return to the best V4-style training objective and test sampling-time guidance/projection. Rather than adding another training loss, V4g will keep the trained denoiser fixed and compare standard reverse diffusion against lightweight sampling-time interventions that encourage length-aware global expansion while preserving local adjacent Cα geometry.

## Review: V4f and V4g final intervention experiments.

After V4e showed that a soft nonlocal contact-tail objective could move some global compactness metrics but damaged local Cα continuity, I tested V4f as a more targeted refinement. V4f used a one-sided excess-contact loss, penalising generated structures only when they produced too many short nonlocal Cα contacts relative to real structures. The aim was to avoid penalising legitimate tertiary contacts while still discouraging global collapse.

The V4f result was a useful negative experiment. The model trained normally, but the excess-contact loss remained very small and did not transfer into better free sampling. Generated structures remained globally compact, collapse count worsened, and local adjacent Cα quality fell below the V4 reference. This suggested that the issue was no longer simply choosing the right nonlocal training loss. Across V4d, V4e, and V4f, training-time global/nonlocal losses had some conceptual leverage, but they did not reliably improve the iterative reverse-sampling process.

This motivated V4g, which changed the intervention point from training time to sampling time. Instead of adding another global loss to the training objective, V4g returned to the stronger V4-style denoiser objective and tested lightweight sampling-time projection. This was inspired by the broader principle behind guided and conditioned diffusion methods: after training a denoiser, the sampling trajectory can be steered towards desired structural properties.

V4g trained cleanly with the local V4 objective and then compared standard sampling against weak, medium, and strong projection conditions. The projection applied two simple corrections during reverse diffusion: a gentle radius-based expansion to discourage global collapse, and a later local Cα projection to preserve adjacent Cα spacing. This directly tested whether the remaining collapse problem was better addressed during sampling rather than through yet another training-time nonlocal loss.

The V4g results were the strongest intervention result so far. The medium projection setting gave the best balance. It substantially reduced collapse count, increased radius of gyration, increased mean nonlocal Cα distance, and reduced the fraction of short nonlocal Cα contacts, while preserving or slightly improving adjacent Cα in-band quality. This was the first intervention that clearly improved global compactness without the local-geometry trade-off seen in V4d, V4e, and V4f.

V4g does not fully solve protein generation. Even the medium projection samples remain more compact than real validation structures, and the approach is heuristic rather than a learned generative prior. However, it provides a clear and useful conclusion: the trained V4 denoiser can produce good local backbone geometry, but the free sampling trajectory is biased towards overly compact structures. Sampling-time projection can partially correct this bias more effectively than the nonlocal training losses tested here.

For the final write-up, I will treat V4g with medium projection as the strongest practical result. The plain V4/V4g denoiser remains the core learned model, while the V4g projection experiment is reported as a lightweight sampling-time intervention that improves global compactness while preserving local backbone quality.


## Independent review (2026-06-09): the terminal-SNR root cause and a final-day intervention menu

This section is a second-opinion review added on the last day before the final write-up. It reads the saved diagnostics and the code directly rather than the version narrative above. Its purpose is to name a root cause the V1–V4g trajectory did not identify, and to lay out a ranked set of interventions that can still be tried today.

### The missed root cause: non-zero terminal SNR

The forward process uses the textbook linear DDPM schedule (`beta` 1e-4 → 2e-2 in `src/latent_structure_generation/backbone_diffusion.py::create_noise_schedule`), but it is run with `T = 100`. That schedule was designed for `T = 1000`. At `T = 100` the cumulative `alpha_bar_T = 0.364`, so `sqrt(alpha_bar_T) = 0.60`. In plain terms: even the *noisiest* training step still contains about 60 percent of the clean structure. The terminal signal-to-noise ratio is about 0.57, not ~0.

This matters because every sampler starts generation from pure `N(0, I)` noise — a distribution the model never saw in training, where the noisiest example always retained 60 percent signal. The mismatch bakes in a contraction. Working the epsilon-prediction arithmetic at the terminal step, the first reverse step reconstructs `x0 ≈ 0.335 · x_T`. The prior cloud has Rg ≈ 20.3 Å (per-axis std ≈ [11.5, 11.5, 12.1] recovered from the checkpoint), so the first denoised structure already has Rg ≈ 20.3 × 0.335 ≈ 6.8 Å. The saved reverse-trajectory diagnostic starts at 6.46 Å and only climbs to ~8.2 Å. That match is the signature: roughly half the global collapse is manufactured by the schedule before the model expresses any learned preference.

A one-line sanity check would have caught it: print `alpha_bar_T` (or terminal SNR) and confirm it is ~0, or plot a real structure noised to `t = T` and notice the fold is still visible. This is a well-known silent footgun (see Lin et al. 2023, "Common Diffusion Noise Schedules and Sample Steps are Flawed"), so it is a lesson rather than a failing — but it is the single highest-leverage thing to fix.

### Why this re-explains the whole V4d–V4g story

The trajectory above repeatedly added global/nonlocal losses at training time (V4d exact distances, V4e contact tails, V4f one-sided excess contacts) and found they "did not transfer" to free sampling. The terminal-SNR bug explains why, for two compounding reasons:

1. The sampler always begins in a contracted basin (Rg ~6.5 Å). No training-time loss changes where sampling *starts*; it can only nudge the trajectory afterward, and the trajectory never had enough reverse "room" to re-expand to ~16 Å.
2. Those nonlocal losses were applied mainly at mid/high noise, where the epsilon-model's `x0` estimate is least reliable — and, under this schedule, additionally contracted. The gradient they supplied was therefore weak and biased.

So the conclusion "free sampling needs to be steered directly" (which motivated V4g) was reasonable given the evidence, but the deeper cause was upstream in the data pipeline. V4g's projection guidance worked precisely because it forcibly re-expands the contracted sample at sampling time — it is treating the symptom of the schedule bug, not the cause.

One correction to the record while here: the saved `v4g_guidance_comparison_summary` table actually shows the **weak** projection setting giving the best balance (highest adjacent-CA in-band ≈ 0.885, lowest poor-CA-band count), not medium. The reviews above state medium is best. For the final write-up, weak is the more honest pick.

### The primary fix (implemented as V4h)

Switch to a cosine schedule (Nichol & Dhariwal 2021), which drives `alpha_bar_T` to ≈ 2.4e-7 (terminal SNR ≈ 0) without the divide-by-zero that an exact-zero epsilon-prediction setup would hit. `T = 100` is kept so the schedule *shape* is the only change, which keeps the fix attributable. This lives in `notebooks/protein_backbone_diffusion_v4h.ipynb`, otherwise a copy of V4g, writing to `results/v4h/`. Cosine and the full Lin et al. zero-terminal-SNR rescale solve the same problem; cosine is chosen because it does not also require switching to v-prediction and changing the sampler. Success criteria: mean generated Rg moving off ~8 Å toward ≥12–14 Å, collapse fraction falling, adjacent-CA in-band holding ≥0.85.

### Additional interventions for the final day, ranked by benefit / risk / time

The cosine fix removes the *manufactured* half of the collapse. The remaining gap (isotropic prior, no global supervision that transfers, limited receptive field) is what the items below target. They are tiered so the safe, cheap ones can be stacked today and the risky ones are explicitly flagged as future work.

#### Tier 1 — safe to do today (cheap, low risk, stack freely)

1. **More reverse steps with cosine (`TIMESTEPS = 250`).** *Why:* cosine's end-of-schedule betas are large (last few ≈ 0.55, 0.75, 0.999); more steps smooth the early reverse trajectory and give the sampler more room to expand. *Where:* the `TIMESTEPS` constant in the schedule cell of V4h; retrain (cosine is defined for any T). *Effort:* one constant + retrain. *Risk:* low. *Expected:* modest extra Rg and smoother samples on top of the cosine fix.

2. **EMA of model weights for sampling.** *Why:* an exponential moving average of parameters is a standard, near-free diffusion quality boost and reduces sample-to-sample variance. *Where:* maintain `ema_state` in the training loop (`run_epoch` caller in the training cell), update each step with decay ≈ 0.999, and load EMA weights before the sampling/diagnostic cells. *Effort:* ~15 lines. *Risk:* low. *Expected:* small but reliable improvement in sample cleanliness.

3. **Stack V4g weak projection on top of cosine.** *Why:* the cosine fix and the sampling-time projection are complementary — one removes the artificial contraction, the other corrects residual compactness. *Where:* already present in the guidance-comparison cell; run the `weak` condition against the cosine-trained checkpoint. *Effort:* none beyond running the cell. *Risk:* none; it is also the guaranteed fallback number. *Expected:* best end-to-end Rg of the available options.

4. **Re-center and clamp `x0` during sampling.** *Why:* now that high-`t` `x0` estimates can be large (dividing by a tiny `sqrt(alpha_bar_t)`), a per-step center-of-mass re-centering plus a generous magnitude clamp on the predicted `x0` (a dynamic-thresholding analog) prevents drift and rare blow-ups. *Where:* the reverse loop in `sample_backbone` / the notebook samplers. *Effort:* a few lines. *Risk:* low. *Expected:* robustness, not a metric jump.

5. **Cheap diagnostics that strengthen the writeup.** Add (a) a log Rg vs log N scaling plot comparing the generated slope to the real polymer exponent (~0.4, the Chroma scaling law) — this shows directly whether the model learned global scale; (b) real-vs-generated distogram/contact-map overlays; (c) a cosine-vs-linear A/B table of Rg, collapse fraction, and in-band. *Why:* these turn "I fixed a bug" into measured evidence. *Effort:* low (most helpers already exist). *Risk:* none.

#### Tier 2 — high value, do at most ONE if time remains (moderate effort/risk)

6. **Low-noise pairwise/distogram Cα loss (FrameDiff `L_2D` style).** *Why:* this is the principled version of what V4d/e/f attempted but mistimed. Apply a Smooth L1 loss on the full (or subsampled) pairwise Cα distance matrix of the predicted clean structure versus the true clean structure, gated to **low** timesteps only (e.g. `t ≤ 20`), where `x0_pred` is now reliable after the schedule fix. Unlike a scalar Rg penalty, a distogram supervises the entire global shape; unlike the V4d high-noise version, it acts where the gradient is trustworthy. *Where:* mirror `adjacent_ca_geometry_loss` in the training cell, build it from `x0_pred_to_angstrom_coords`, gate with a low-`t` mask, weight ≈ 0.01–0.05. *Effort:* ~30–40 lines. *Risk:* medium (start with a small weight to avoid the over-expansion seen in the V4d overfit run). *Expected:* the most likely single lever to push Rg from ~12–14 toward real.

7. **Self-conditioning (RFdiffusion / Chen et al. 2022).** *Why:* feeding the model its own previous `x0` prediction is used in essentially every strong structure-diffusion system and materially stabilizes global geometry across reverse steps. *Where:* extend `BackboneCoordinateEGNNDenoiser.forward` to accept an optional prior-`x0` channel, and in training pass it with 50 percent probability (zeros otherwise). *Effort:* moderate (model + training loop). *Risk:* medium; more surface area than item 6. *Expected:* steadier trajectories and better global extent, but only attempt if item 6 is already done or skipped.

#### Tier 3 — do NOT start today (high value, high risk; list as future work)

8. **Spatial / k-NN edges in the EGNN graph.** Adding edges between spatial neighbours of the current `x_t` (not just fixed sequence offsets 8, 16) gives the denoiser a receptive field that can form tertiary contacts — the principled fix for the limited receptive field. But rebuilding the graph from coordinates each forward pass, with correct masking and equivariance, is too much to debug safely on the last day.

9. **Chroma-style correlated polymer prior.** Sampling the initial noise from a correlated Gaussian whose covariance encodes chain connectivity (Rg ~ N^0.4) injects the right global scale at initialization and is arguably the deepest fix for the isotropic prior. But it must be applied consistently in both `q_sample` (training) and the sampler, so it is all-or-nothing and high risk for today.

### What NOT to do on the final day

Do not train past ~40 epochs — validation loss plateaued by epoch ~25 and the teacher-forced `x0_pred_rms` was already flat at the correct scale, so more epochs will not fix free-sampling collapse. Do not switch to `(4, 8, 16, 32)` sequence-offset edges — the notes already showed ~3× cost with no gain. Do not add more scalar radius/Rg penalties — V4d–V4f showed they do not transfer; prefer the low-noise distogram loss instead. Do not keep enlarging the guidance candidate count — that is cherry-picking and hides collapse rather than fixing it. Spend the day instead on the cosine retrain (V4h), then stack Tier 1 items, and add item 6 only if time allows.

### If collapse still persists after V4h

Frame it precisely and the work still reads as strong. The final write-up can state: local geometry is genuinely learned (adjacent Cα 3.74 vs 3.81 Å, in-band ≈ 0.86); the global collapse was traced to a non-zero terminal SNR (`alpha_bar_T = 0.36` at `T = 100`) with a quantitative prediction (0.335 contraction → 6.8 Å predicted vs 6.46 Å observed); the cosine schedule removes the manufactured contraction; and the remaining gap is attributable to the isotropic prior and the absence of transferable global supervision, with a concrete roadmap (low-noise distogram loss, self-conditioning, spatial edges, correlated prior) drawn from FrameDiff, RFdiffusion, and Chroma. A clear diagnosis with a measured partial fix is worth more than an unexplained lucky number.

## Review: V4g, V4h, and V4i.

V4g was the strongest practical result before the diffusion-schedule investigation. It kept the V4 EGNN-style denoiser and local geometry objective, then added sampling-time projection/guidance. The best V4g setting was the medium projection condition, which improved global compactness compared with baseline V4-style sampling while preserving adjacent Cα quality. This made V4g medium projection the best empirical model at that stage.

A late review then identified a more foundational issue in the diffusion setup. V4g used `TIMESTEPS = 100` with a standard linear DDPM beta schedule. With only 100 steps, the terminal cumulative signal level remained high: `alpha_bar_T` was around `0.36`, so `sqrt(alpha_bar_T)` was around `0.60`. This means the noisiest training examples still contained substantial clean structural signal. However, sampling starts from pure Gaussian noise. This created a mismatch between the noisiest structures seen during training and the starting point used during generation.

V4h tested a correction to this issue. It preserved the V4-style model and local geometry objective, but replaced the old 100-step linear schedule with a cosine-style schedule that drove the terminal signal level much closer to zero. This was technically motivated: the goal was to make the final training timestep more consistent with pure-noise sampling.

The full V4h terminal sampler did not work. Starting from `t=100` produced physically exploded structures rather than plausible protein backbones. Radius of gyration became extremely large, adjacent Cα distances became impossible, and structure visualisation failed. Fixed-timestep diagnostics suggested why: at near-zero `alpha_bar`, the epsilon-prediction formula for reconstructing `x0_pred` becomes numerically fragile, so small prediction errors can be amplified into enormous coordinate errors. V4h therefore fixed the under-noising issue, but overcorrected into an unstable high-noise endpoint for this 100-step epsilon-prediction setup.

To understand whether V4h was completely unusable or only failing at the extreme endpoint, I added a t-start sampling diagnostic. This tested reverse sampling from `t_start = 100, 90, 75, 50` without retraining. The diagnostic showed a clear pattern. Sampling from `t_start=100` still exploded. Sampling from `t_start=50` was too collapsed because the model had too little high-noise trajectory left to build global structure. Sampling from `t_start=75` gave acceptable local geometry but remained too compact. Sampling from `t_start=90` gave the best balance: the structures were much closer to real global scale, with radius of gyration around 14 Å, acceptable adjacent Cα in-band fraction, lower short nonlocal-contact fractions, and no collapse in the small diagnostic sample.

This suggests that the V4h denoiser did learn useful high-noise behaviour, but the final near-zero-SNR endpoint was too unstable to use directly. V4i is therefore the next clean variant: it keeps the V4h cosine schedule but promotes truncated-start sampling from `t_start=90` as the default generation procedure.

V4i should be interpreted honestly. It is not a perfect standard DDPM sampler from the full terminal prior, because it avoids the unstable `t=100` endpoint. However, it is a useful and scientifically motivated diagnostic/generation variant. It tests whether the corrected schedule can produce more realistic global structure when the numerically unstable endpoint is skipped. If the `t_start=90` result holds at a larger sample count, V4i may become the best final sampling result, especially because it substantially improves global compactness compared with V4g medium projection while retaining acceptable local Cα geometry.
