"""Input/output helpers for structure files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser

SUPPORTED_SUFFIXES = {".cif", ".mmcif", ".pdb", ".ent"}


def collect_structure_files(data_dir: str | Path) -> list[Path]:
    """Return supported structure files from a directory, recursively."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    return sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)


def _get_parser(path: Path):
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    if path.suffix.lower() in {".pdb", ".ent"}:
        return PDBParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path.suffix}")


def extract_atom_table(
    path: str | Path,
    chain_id: str | None = None,
    atom_names: tuple[str, ...] = ("N", "CA", "C", "O"),
) -> pd.DataFrame:
    """Extract selected atoms from a PDB or mmCIF file into a dataframe.

    The returned table has one row per atom. It is intentionally simple so it can
    be adapted quickly if the input data uses a custom format.
    """
    path = Path(path)
    parser = _get_parser(path)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())

    rows: list[dict[str, object]] = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue_position, residue in enumerate(chain, start=1):
            hetflag, auth_seq_id, insertion_code = residue.id
            if hetflag.strip():
                continue
            residue_name = residue.get_resname()
            for atom_name in atom_names:
                if atom_name not in residue:
                    continue
                atom = residue[atom_name]
                x, y, z = atom.coord.astype(float)
                rows.append(
                    {
                        "structure_id": path.stem,
                        "path": str(path),
                        "chain_id": chain.id,
                        "residue_index": residue_position,
                        "auth_seq_id": auth_seq_id,
                        "insertion_code": insertion_code.strip() or "",
                        "residue_name": residue_name,
                        "atom_name": atom_name,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )
    return pd.DataFrame(rows)


def ca_coordinates(atom_table: pd.DataFrame, chain_id: str | None = None) -> pd.DataFrame:
    """Return C-alpha rows from an atom table."""
    table = atom_table[atom_table["atom_name"] == "CA"].copy()
    if chain_id is not None:
        table = table[table["chain_id"] == chain_id].copy()
    return table.sort_values(["chain_id", "residue_index"]).reset_index(drop=True)


def atom_availability_summary(atom_table: pd.DataFrame) -> pd.DataFrame:
    """Summarise backbone atom availability per structure and chain."""
    if atom_table.empty:
        return pd.DataFrame()
    summary = (
        atom_table.assign(present=True)
        .pivot_table(
            index=["structure_id", "chain_id"],
            columns="atom_name",
            values="present",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    summary.columns.name = None
    return summary
