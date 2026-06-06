"""Backbone diffusion baseline utilities for V1.

This module keeps the notebook thin: data preparation, diffusion math,
the denoiser, evaluation utilities, and PDB conversion live here so the
Colab notebook can stay readable and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")
ATOM_TYPES_37: tuple[str, ...] = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)
ATOM_INDEX_37 = {name: i for i, name in enumerate(ATOM_TYPES_37)}

RESTYPES = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)
RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)


@dataclass(frozen=True)
class BackboneNormalizationStats:
    """Train-only coordinate normalisation statistics."""

    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class Protein:
    """Minimal protein container for PDB export.

    Adapted from the earlier Colab notebook helper in
    archive/vae_egnn_runway/notebooks/00_v0_data_validation.ipynb.
    """

    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    chain_index: np.ndarray
    b_factors: np.ndarray


def _chain_end(atom_index: int, end_resname: str, chain_name: str, residue_index: int) -> str:
    """Return a PDB TER line."""

    chain_end = "TER"
    return f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}"


def to_pdb(prot: Protein) -> str:
    """Convert a Protein object to a PDB string.

    Adapted from archive/vae_egnn_runway/notebooks/00_v0_data_validation.ipynb.
    """

    restypes = list(RESTYPES) + ["X"]
    res_1to3 = lambda r: RESTYPE_1TO3.get(restypes[r], "UNK")

    pdb_lines: list[str] = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > len(RESTYPES)):
        raise ValueError("Invalid aatypes.")

    chain_ids: dict[int, str] = {}
    for i in np.unique(chain_index):
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[int(i)] = PDB_CHAIN_IDS[int(i)]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = int(chain_index[0]) if len(chain_index) else 0

    for i in range(aatype.shape[0]):
        if last_chain_index != int(chain_index[i]):
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(int(aatype[i - 1])),
                    chain_ids[int(chain_index[i - 1])],
                    int(residue_index[i - 1]),
                )
            )
            last_chain_index = int(chain_index[i])
            atom_index += 1

        res_name_3 = res_1to3(int(aatype[i]))
        for atom_name, pos, mask_value, b_factor in zip(
            ATOM_TYPES_37, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask_value < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            occupancy = 1.00
            element = atom_name[0]
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{'':>1}"
                f"{res_name_3:>3} {chain_ids[int(chain_index[i])]:>1}"
                f"{int(residue_index[i]):>4}{'':>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{float(b_factor):>6.2f}          "
                f"{element:>2}{'':>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    if len(aatype):
        pdb_lines.append(
            _chain_end(
                atom_index,
                res_1to3(int(aatype[-1])),
                chain_ids[int(chain_index[-1])],
                int(residue_index[-1]),
            )
        )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"


def backbone_coords_to_protein(coords: torch.Tensor, mask: torch.Tensor) -> Protein:
    """Convert one padded backbone tensor into a Protein object for PDB export."""

    if coords.ndim != 3 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (L, 4, 3).")
    if mask.ndim != 1:
        raise ValueError("mask must have shape (L,).")

    coords_np = coords.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    n_res = coords_np.shape[0]

    atom_positions = np.zeros((n_res, len(ATOM_TYPES_37), 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, len(ATOM_TYPES_37)), dtype=np.float32)
    for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
        atom_positions[:, ATOM_INDEX_37[atom_name], :] = coords_np[:, atom_idx, :]
        atom_mask[:, ATOM_INDEX_37[atom_name]] = mask_np.astype(np.float32)

    return Protein(
        atom_positions=atom_positions,
        aatype=np.zeros((n_res,), dtype=np.int32),
        atom_mask=atom_mask,
        residue_index=np.arange(1, n_res + 1, dtype=np.int32),
        chain_index=np.zeros((n_res,), dtype=np.int32),
        b_factors=np.ones((n_res, len(ATOM_TYPES_37)), dtype=np.float32),
    )


def flatten_backbone(coords: torch.Tensor) -> torch.Tensor:
    """Flatten backbone coordinates.

    Parameters
    ----------
    coords:
        Tensor with shape ``(B, L, 4, 3)``.

    Returns
    -------
    Tensor with shape ``(B, L, 12)``.
    """

    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3).")
    return coords.reshape(coords.shape[0], coords.shape[1], 12)


def unflatten_backbone(coords_flat: torch.Tensor) -> torch.Tensor:
    """Reverse ``flatten_backbone``.

    Parameters
    ----------
    coords_flat:
        Tensor with shape ``(B, L, 12)``.

    Returns
    -------
    Tensor with shape ``(B, L, 4, 3)``.
    """

    if coords_flat.ndim != 3 or coords_flat.shape[-1] != 12:
        raise ValueError("coords_flat must have shape (B, L, 12).")
    return coords_flat.reshape(coords_flat.shape[0], coords_flat.shape[1], 4, 3)


def _mask_expand(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    """Expand a residue mask to the requested tensor rank."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, L).")
    if ndim == 3:
        return mask.unsqueeze(-1)
    if ndim == 4:
        return mask.unsqueeze(-1).unsqueeze(-1)
    raise ValueError(f"Unsupported ndim for mask expansion: {ndim}")


def centre_coordinates(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Centre each structure around the mean of real residues only.

    Parameters
    ----------
    coords:
        Tensor with shape ``(B, L, 4, 3)``.
    mask:
        Tensor with shape ``(B, L)`` where ``1`` marks real residues.

    Returns
    -------
    Tensor with the same shape as ``coords``. Padded residues remain zero.
    """

    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3).")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, L).")

    mask_atoms = mask.unsqueeze(-1).unsqueeze(-1).float()
    denom = (mask.sum(dim=1, keepdim=True) * coords.shape[2]).view(-1, 1, 1, 1).clamp_min(1.0)
    centre = (coords * mask_atoms).sum(dim=(1, 2), keepdim=True) / denom
    centred = (coords - centre) * mask_atoms
    return centred


def apply_coordinate_normalisation(
    coords: torch.Tensor,
    mask: torch.Tensor,
    stats: BackboneNormalizationStats,
) -> torch.Tensor:
    """Apply train-derived normalisation to coordinates.

    The inputs should already be centred with :func:`centre_coordinates`.
    Padded residues are kept at zero.
    """

    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3).")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, L).")

    mean = stats.mean.to(device=coords.device, dtype=coords.dtype).view(1, 1, 1, 3)
    std = stats.std.to(device=coords.device, dtype=coords.dtype).view(1, 1, 1, 3).clamp_min(1e-6)
    mask_atoms = mask.unsqueeze(-1).unsqueeze(-1).float()
    return ((coords - mean) / std) * mask_atoms


def invert_coordinate_normalisation(coords_norm: torch.Tensor, stats: BackboneNormalizationStats) -> torch.Tensor:
    """Invert :func:`apply_coordinate_normalisation`."""

    if coords_norm.ndim != 4 or coords_norm.shape[-2:] != (4, 3):
        raise ValueError("coords_norm must have shape (B, L, 4, 3).")
    mean = stats.mean.to(device=coords_norm.device, dtype=coords_norm.dtype).view(1, 1, 1, 3)
    std = stats.std.to(device=coords_norm.device, dtype=coords_norm.dtype).view(1, 1, 1, 3).clamp_min(1e-6)
    return coords_norm * std + mean


def create_noise_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Create a standard linear DDPM schedule.

    Returns 1-indexed tensors with a dummy zero entry at position 0 so that
    timestep ``t`` can be used directly as an index.
    """

    if timesteps <= 0:
        raise ValueError("timesteps must be positive.")

    betas_1d = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    alphas_1d = 1.0 - betas_1d
    alpha_bars_1d = torch.cumprod(alphas_1d, dim=0)
    alpha_bars_prev_1d = torch.cat([torch.ones(1, device=device), alpha_bars_1d[:-1]])
    posterior_variance_1d = betas_1d * (1.0 - alpha_bars_prev_1d) / (1.0 - alpha_bars_1d).clamp_min(1e-20)
    posterior_variance_1d[0] = 0.0

    zeros = torch.zeros(1, dtype=torch.float32, device=device)
    ones = torch.ones(1, dtype=torch.float32, device=device)
    schedule = {
        "betas": torch.cat([zeros, betas_1d]),
        "alphas": torch.cat([ones, alphas_1d]),
        "alpha_bars": torch.cat([ones, alpha_bars_1d]),
        "alpha_bars_prev": torch.cat([ones, alpha_bars_prev_1d]),
        "posterior_variance": torch.cat([zeros, posterior_variance_1d]),
    }
    return schedule


def sample_timesteps(batch_size: int, timesteps: int, device: torch.device) -> torch.Tensor:
    """Sample diffusion timesteps uniformly from ``[1, timesteps]``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive.")
    return torch.randint(1, timesteps + 1, (batch_size,), device=device, dtype=torch.long)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion step ``q(x_t | x_0)``."""

    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have the same shape.")
    if t.ndim != 1:
        raise ValueError("t must have shape (B,).")
    if alpha_bars.ndim != 1:
        raise ValueError("alpha_bars must be a 1D tensor.")

    alpha_bar_t = alpha_bars[t].view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def sinusoidal_timestep_embedding(t: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""

    if t.ndim != 1:
        raise ValueError("t must have shape (B,).")
    half_dim = embedding_dim // 2
    device = t.device
    exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
    exponent = -math.log(10000.0) * exponent / max(half_dim - 1, 1)
    angles = t.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ResidualConvBlock(nn.Module):
    """Lightweight 1D residual block used by the denoiser."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        x = self.activation(x + residual)
        x = x * mask.unsqueeze(1).float()
        return x


class BackboneDenoiser(nn.Module):
    """Simple DDPM denoiser for padded backbone coordinates."""

    def __init__(
        self,
        max_length: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embedding_dim: int = 128,
    ):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.input_proj = nn.Linear(12 + 1, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, 12)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict noise for a batch of flattened backbone coordinates."""

        if x_t.ndim != 3 or x_t.shape[-1] != 12:
            raise ValueError("x_t must have shape (B, L, 12).")
        if mask.ndim != 2:
            raise ValueError("mask must have shape (B, L).")
        if x_t.shape[:2] != mask.shape:
            raise ValueError("x_t and mask must agree on batch and length dimensions.")
        if x_t.shape[1] > self.max_length:
            raise ValueError(
                f"Sequence length {x_t.shape[1]} exceeds max_length={self.max_length}. "
                "Increase the model max_length or crop the dataset."
            )

        mask_f = mask.float()
        pos_ids = torch.arange(x_t.shape[1], device=x_t.device).unsqueeze(0).expand(x_t.shape[0], -1)
        time_emb = sinusoidal_timestep_embedding(t, self.time_embedding_dim)
        time_emb = self.time_mlp(time_emb).unsqueeze(1)

        h = torch.cat([x_t, mask_f.unsqueeze(-1)], dim=-1)
        h = self.input_proj(h) + self.pos_emb(pos_ids) + time_emb
        h = h.transpose(1, 2)
        h = h * mask_f.unsqueeze(1)
        for block in self.blocks:
            h = block(h, mask)
        h = h.transpose(1, 2)
        out = self.output_proj(h) * mask_f.unsqueeze(-1)
        return out


def masked_noise_mse(pred_noise: torch.Tensor, true_noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean squared error over valid residues only."""

    if pred_noise.shape != true_noise.shape:
        raise ValueError("pred_noise and true_noise must have the same shape.")
    if pred_noise.ndim != 3 or pred_noise.shape[-1] != 12:
        raise ValueError("Noise tensors must have shape (B, L, 12).")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, L).")

    mask_expanded = mask.unsqueeze(-1).expand_as(pred_noise).float()
    denom = mask_expanded.sum().clamp_min(1.0)
    return ((pred_noise - true_noise).pow(2) * mask_expanded).sum() / denom


def masked_coordinate_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked RMSE over valid coordinates only."""

    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape.")
    mask_expanded = mask.unsqueeze(-1).expand_as(pred).float()
    denom = mask_expanded.sum().clamp_min(1.0)
    return torch.sqrt(((pred - target).pow(2) * mask_expanded).sum() / denom)


@torch.no_grad()
def predict_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_noise: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct x0 from the model's noise prediction."""

    alpha_bar_t = alpha_bars[t].view(-1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
    return (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)


@torch.no_grad()
def sample_backbone(
    model: nn.Module,
    schedule: dict[str, torch.Tensor],
    shape: tuple[int, int, int],
    mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample flattened backbone coordinates using reverse diffusion."""

    if len(shape) != 3 or shape[2] != 12:
        raise ValueError("shape must be (B, L, 12).")
    batch_size, seq_len, _ = shape
    if device is None:
        device = next(model.parameters()).device

    x = torch.randn(shape, device=device)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
    else:
        mask = mask.to(device=device, dtype=torch.float32)

    betas = schedule["betas"].to(device)
    alphas = schedule["alphas"].to(device)
    alpha_bars = schedule["alpha_bars"].to(device)
    posterior_variance = schedule["posterior_variance"].to(device)
    timesteps = int(betas.shape[0] - 1)

    for timestep in range(timesteps, 0, -1):
        t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        pred_noise = model(x, t, mask)
        alpha_t = alphas[timestep]
        beta_t = betas[timestep]
        alpha_bar_t = alpha_bars[timestep]
        mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)

        if timestep > 1:
            noise = torch.randn_like(x)
            variance = posterior_variance[timestep]
            x = mean + torch.sqrt(variance.clamp_min(1e-20)) * noise
        else:
            x = mean

        x = x * mask.unsqueeze(-1)

    return x


def extract_backbone_from_coords_dict(coords_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone coordinates and a residue mask from a raw coords dict."""

    missing = [atom for atom in BACKBONE_ATOMS if atom not in coords_dict]
    if missing:
        raise KeyError(f"Missing backbone atoms in coords dict: {missing}")

    arrays = [np.asarray(coords_dict[atom], dtype=np.float32) for atom in BACKBONE_ATOMS]
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Backbone atom arrays disagree on length: {sorted(lengths)}")
    n_res = arrays[0].shape[0]
    coords = np.zeros((n_res, 4, 3), dtype=np.float32)
    residue_mask = np.ones((n_res,), dtype=bool)

    for atom_idx, arr in enumerate(arrays):
        if arr.shape != (n_res, 3):
            raise ValueError(f"Backbone atom {BACKBONE_ATOMS[atom_idx]!r} has unexpected shape {arr.shape}")
        finite = np.isfinite(arr).all(axis=-1)
        coords[:, atom_idx, :] = np.where(np.isfinite(arr), arr, 0.0)
        residue_mask &= finite

    residue_mask &= np.isfinite(coords).all(axis=(1, 2))
    coords[~residue_mask] = 0.0
    return coords, residue_mask


def pad_or_crop_backbone(
    coords: np.ndarray,
    residue_mask: np.ndarray,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Pad or crop a backbone to a fixed length."""

    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if coords.ndim != 3 or coords.shape[1:] != (4, 3):
        raise ValueError("coords must have shape (L, 4, 3).")
    if residue_mask.ndim != 1:
        raise ValueError("residue_mask must have shape (L,).")

    n_res = coords.shape[0]
    truncated = n_res > max_length
    coords_fixed = np.zeros((max_length, 4, 3), dtype=np.float32)
    mask_fixed = np.zeros((max_length,), dtype=bool)
    n_copy = min(n_res, max_length)
    coords_fixed[:n_copy] = coords[:n_copy]
    mask_fixed[:n_copy] = residue_mask[:n_copy]
    return coords_fixed, mask_fixed, truncated


class BackboneDataset(Dataset):
    """Padded backbone coordinate dataset built from the provided CATH dataframe."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        max_length: int = 256,
        record_id_column: str = "name",
    ):
        self.dataframe = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.split = split
        self.max_length = max_length
        self.record_id_column = record_id_column
        if self.dataframe.empty:
            raise ValueError(f"No rows found for split={split!r}.")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        coords, residue_mask = extract_backbone_from_coords_dict(row["coords"])
        coords, residue_mask, truncated = pad_or_crop_backbone(coords, residue_mask, self.max_length)
        seq_value = row["seq"] if "seq" in row.index else ""
        return {
            "record_id": row[self.record_id_column],
            "split": row["split"],
            "coords": torch.tensor(coords, dtype=torch.float32),
            "mask": torch.tensor(residue_mask.astype(np.float32), dtype=torch.float32),
            "length": len(seq_value),
            "real_length": int(residue_mask.sum()),
            "truncated": bool(truncated),
        }


def collate_backbone_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate backbone examples into batched tensors and metadata."""

    coords = torch.stack([item["coords"] for item in batch], dim=0)
    mask = torch.stack([item["mask"] for item in batch], dim=0)
    return {
        "record_id": [item["record_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "coords": coords,
        "mask": mask,
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "real_length": torch.tensor([item["real_length"] for item in batch], dtype=torch.long),
        "truncated": torch.tensor([item["truncated"] for item in batch], dtype=torch.bool),
    }


def _ca_coordinates(coords: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CA coordinates and the corresponding residue mask."""

    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3).")
    ca = coords[:, :, 1, :]
    return ca, mask.bool()


def backbone_structure_summary(coords: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute a compact structural summary for one structure or a batch."""

    if coords.ndim == 4:
        coords = coords[0]
        mask = mask[0]
    if coords.ndim != 3 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (L, 4, 3).")
    if mask.ndim != 1:
        raise ValueError("mask must have shape (L,).")

    mask_bool = mask.bool()
    real_coords = coords[mask_bool]
    if real_coords.numel() == 0:
        return {
            "n_residues": 0.0,
            "mean_adjacent_ca": float("nan"),
            "fraction_adjacent_ca_in_band": float("nan"),
            "mean_n_ca": float("nan"),
            "mean_ca_c": float("nan"),
            "mean_c_o": float("nan"),
            "mean_c_n": float("nan"),
            "radius_of_gyration": float("nan"),
        }

    ca = coords[:, 1, :]
    valid_adjacent = mask_bool[1:] & mask_bool[:-1]
    adjacent_ca = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1)
    adjacent_values = adjacent_ca[valid_adjacent]

    n_ca = torch.linalg.norm(coords[:, 0, :] - coords[:, 1, :], dim=-1)[mask_bool]
    ca_c = torch.linalg.norm(coords[:, 1, :] - coords[:, 2, :], dim=-1)[mask_bool]
    c_o = torch.linalg.norm(coords[:, 2, :] - coords[:, 3, :], dim=-1)[mask_bool]

    if mask_bool.sum() > 1:
        c_n = torch.linalg.norm(coords[:-1, 2, :] - coords[1:, 0, :], dim=-1)
        c_n = c_n[valid_adjacent]
    else:
        c_n = torch.tensor([], device=coords.device, dtype=coords.dtype)

    centred_ca = ca[mask_bool] - ca[mask_bool].mean(dim=0, keepdim=True)
    rg = torch.sqrt((centred_ca.pow(2).sum(dim=-1)).mean())
    band = (adjacent_values >= 3.5) & (adjacent_values <= 4.1)

    return {
        "n_residues": float(mask_bool.sum().item()),
        "mean_adjacent_ca": float(adjacent_values.mean().item()) if len(adjacent_values) else float("nan"),
        "fraction_adjacent_ca_in_band": float(band.float().mean().item()) if len(adjacent_values) else float("nan"),
        "mean_n_ca": float(n_ca.mean().item()) if len(n_ca) else float("nan"),
        "mean_ca_c": float(ca_c.mean().item()) if len(ca_c) else float("nan"),
        "mean_c_o": float(c_o.mean().item()) if len(c_o) else float("nan"),
        "mean_c_n": float(c_n.mean().item()) if len(c_n) else float("nan"),
        "radius_of_gyration": float(rg.item()),
    }


def build_normalization_stats(coords: torch.Tensor, mask: torch.Tensor) -> BackboneNormalizationStats:
    """Compute train-only mean and standard deviation over real atom coordinates."""

    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3).")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, L).")

    real_coords = coords[mask.bool()].reshape(-1, 3)
    if real_coords.numel() == 0:
        raise ValueError("Cannot compute normalisation stats from an empty mask.")
    mean = real_coords.mean(dim=0)
    std = real_coords.std(dim=0, unbiased=False).clamp_min(1e-6)
    return BackboneNormalizationStats(mean=mean.detach().cpu(), std=std.detach().cpu())


def structure_validity_report(coords: torch.Tensor, mask: torch.Tensor) -> pd.DataFrame:
    """Return a simple per-structure sanity table for a batch."""

    rows = []
    for index in range(coords.shape[0]):
        row = backbone_structure_summary(coords[index], mask[index])
        row["index"] = index
        rows.append(row)
    return pd.DataFrame(rows)
