# Decision log.

## V0.

Date: YYYY-MM-DD.

Decision: Start with C-alpha extraction, distance matrices, graph construction, and validation metrics before training any model.

Reason: This keeps the project biologically grounded from the beginning. Even a simple generative baseline needs structural checks.

Trade-off: V0 does not prove generative ability by itself, but it prevents later models from producing unvalidated outputs.

## V1.

Date: YYYY-MM-DD.

Decision: Use a distance-matrix VAE as the first sampleable generative baseline.

Reason: Pairwise distances remove arbitrary rotation and translation, making the baseline easier to interpret than raw XYZ coordinate generation.

Trade-off: A generated distance matrix does not guarantee a physically valid protein backbone.

## V2.

Date: YYYY-MM-DD.

Decision: Use a C-alpha graph EGNN denoising model as the main geometry-aware upgrade.

Reason: EGNNs operate directly on 3D coordinates while respecting geometric structure more naturally than a plain MLP.

Trade-off: A denoising model is not automatically a full diffusion model unless a proper noise schedule and sampling procedure are implemented.
