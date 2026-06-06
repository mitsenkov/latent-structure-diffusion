"""Build a small SH2-domain mock dataset for early protein structure generative modelling experiments.

This script uses InterPro domain coordinates to identify SH2 regions and AlphaFold DB
mmCIF files as the source structures. The cropped mmCIF files are intended for V0 data
validation and V1 distance-matrix VAE experiments.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import gemmi


LOGGER = logging.getLogger(__name__)
INTERPRO_BASE_URL = "https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/InterPro/{interpro_accession}/?page_size={page_size}"
AFDB_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}?include_complexes=false"
DEFAULT_INTERPRO_ACCESSION = "IPR000980"
DEFAULT_PAGE_SIZE = 20
DEFAULT_SLEEP_SECONDS = 3.0
DEFAULT_TIMEOUT_SECONDS = 60.0


def _to_int(value: Any) -> int | None:
    """Return an integer if ``value`` can be parsed cleanly, otherwise ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> bool:
    """Best-effort conversion to bool."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes", "y"}
    return bool(value)


def _first_present(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return the first present value from ``mapping`` for the provided ``keys``."""

    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _polite_sleep(seconds: float) -> None:
    """Sleep for the requested duration when it is greater than zero."""

    if seconds > 0:
        time.sleep(seconds)


def fetch_interpro_domains(interpro_accession: str, page_size: int) -> dict[str, Any]:
    """Fetch the first page of InterPro protein records for an InterPro accession."""

    url = INTERPRO_BASE_URL.format(
        interpro_accession=interpro_accession,
        page_size=page_size,
    )
    response = requests.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    _polite_sleep(DEFAULT_SLEEP_SECONDS)
    return response.json()


def parse_interpro_results(payload: dict[str, Any], interpro_accession: str) -> list[dict[str, Any]]:
    """Parse InterPro payload into one record per domain fragment."""

    raw_results = payload.get("results") or payload.get("data") or []
    parsed: list[dict[str, Any]] = []

    for result in raw_results:
        metadata = result.get("metadata") or {}
        entries = result.get("entries") or []
        accession = metadata.get("accession")
        protein_name = metadata.get("name")
        protein_length = _to_int(metadata.get("length"))
        source_database = metadata.get("source_database")
        organism = (
            (metadata.get("source_organism") or {}).get("scientificName")
            if isinstance(metadata.get("source_organism"), dict)
            else None
        )
        in_alphafold = _to_bool(metadata.get("in_alphafold"))

        if not entries:
            parsed.append(
                {
                    "uniprot_accession": accession,
                    "protein_name": protein_name,
                    "source_database": source_database,
                    "protein_length": protein_length,
                    "organism": organism,
                    "in_alphafold": in_alphafold,
                    "interpro_accession": interpro_accession,
                    "domain_index": None,
                    "domain_start": None,
                    "domain_end": None,
                    "domain_length": None,
                    "fragment_status": None,
                    "error_message": "No InterPro entries found for protein.",
                }
            )
            continue

        domain_index = 0
        for entry in entries:
            entry_accession = entry.get("accession") or interpro_accession
            locations = entry.get("entry_protein_locations") or []
            for location in locations:
                fragments = location.get("fragments") or []
                for fragment in fragments:
                    domain_index += 1
                    start = _to_int(fragment.get("start"))
                    end = _to_int(fragment.get("end"))
                    status = fragment.get("dc-status") or fragment.get("dc_status") or fragment.get("status")
                    domain_length = None
                    if start is not None and end is not None and end >= start:
                        domain_length = end - start + 1

                    parsed.append(
                        {
                            "uniprot_accession": accession,
                            "protein_name": protein_name,
                            "source_database": source_database,
                            "protein_length": protein_length,
                            "organism": organism,
                            "in_alphafold": in_alphafold,
                            "interpro_accession": entry_accession,
                            "domain_index": domain_index,
                            "domain_start": start,
                            "domain_end": end,
                            "domain_length": domain_length,
                            "fragment_status": status,
                            "error_message": None,
                        }
                    )

    return parsed


def filter_domain_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only AFDB-available continuous fragments with valid coordinates and length."""

    filtered: list[dict[str, Any]] = []
    for record in records:
        start = record.get("domain_start")
        end = record.get("domain_end")
        domain_length = record.get("domain_length")
        fragment_status = record.get("fragment_status")
        if not record.get("in_alphafold"):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end < start:
            continue
        if domain_length is None or domain_length < 30:
            continue
        if isinstance(fragment_status, str) and fragment_status.upper() != "CONTINUOUS":
            continue
        if fragment_status is None:
            continue
        filtered.append(record)
    return filtered


def fetch_afdb_prediction(uniprot_accession: str) -> dict[str, Any] | None:
    """Fetch the first AlphaFold DB prediction object for a UniProt accession."""

    url = AFDB_PREDICTION_URL.format(uniprot_accession=uniprot_accession)
    response = requests.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    if response.status_code == 404:
        _polite_sleep(DEFAULT_SLEEP_SECONDS)
        return None
    response.raise_for_status()
    payload = response.json()
    _polite_sleep(DEFAULT_SLEEP_SECONDS)
    if isinstance(payload, list):
        return payload[0] if payload else None
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"][0] if payload["results"] else None
        return payload
    return None


def download_file(url: str, output_path: Path, overwrite: bool = False) -> Path:
    """Download a file from ``url`` to ``output_path`` unless it already exists."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return output_path

    response = requests.get(url, timeout=DEFAULT_TIMEOUT_SECONDS, stream=True)
    response.raise_for_status()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(output_path)
    _polite_sleep(DEFAULT_SLEEP_SECONDS)
    return output_path


def _residue_seqid_num(residue: gemmi.Residue) -> int | None:
    """Return the residue seqid number if available."""

    try:
        return int(residue.seqid.num)
    except Exception:
        return None


def _score_chain_for_range(chain: gemmi.Chain, start: int, end: int) -> tuple[int, int]:
    """Score a chain by overlap with the requested residue range."""

    overlap = 0
    polymer_residues = 0
    for residue in chain:
        seqnum = _residue_seqid_num(residue)
        if seqnum is None:
            continue
        polymer_residues += 1
        if start <= seqnum <= end:
            overlap += 1
    return overlap, polymer_residues


def _select_best_chain(structure: gemmi.Structure, start: int, end: int) -> gemmi.Chain | None:
    """Select the chain with the best overlap for the given residue range."""

    if len(structure) == 0:
        return None
    model = structure[0]
    chains = [chain for chain in model]
    if not chains:
        return None

    scored: list[tuple[int, int, int, gemmi.Chain]] = []
    for index, chain in enumerate(chains):
        overlap, polymer_residues = _score_chain_for_range(chain, start, end)
        scored.append((overlap, polymer_residues, -index, chain))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if scored[0][0] <= 0:
        return None
    return scored[0][3]


def crop_mmcif_to_residue_range(
    input_cif: Path,
    output_cif: Path,
    start: int,
    end: int,
) -> dict[str, Any]:
    """Crop a mmCIF file to a residue range and return crop metadata."""

    structure = gemmi.read_structure(str(input_cif))
    selected_chain = _select_best_chain(structure, start, end)
    if selected_chain is None:
        return {
            "selected_chain_id": None,
            "n_residues_cropped": 0,
            "n_atoms_cropped": 0,
            "crop_status": "FAILED",
            "error_message": "No chain overlaps the requested domain range.",
        }

    cropped_structure = gemmi.Structure()
    cropped_structure.name = structure.name
    cropped_model = gemmi.Model("1")
    cropped_chain = gemmi.Chain(selected_chain.name)

    for residue in selected_chain:
        seqnum = _residue_seqid_num(residue)
        if seqnum is None or seqnum < start or seqnum > end:
            continue
        cropped_chain.add_residue(residue.clone())

    n_residues = sum(1 for _ in cropped_chain)
    n_atoms = sum(len(residue) for residue in cropped_chain)
    if n_residues == 0 or n_atoms == 0:
        return {
            "selected_chain_id": selected_chain.name,
            "n_residues_cropped": n_residues,
            "n_atoms_cropped": n_atoms,
            "crop_status": "FAILED",
            "error_message": "Cropped structure has zero residues or atoms.",
        }

    cropped_model.add_chain(cropped_chain)
    cropped_structure.add_model(cropped_model)
    cropped_structure.setup_entities()
    cropped_structure.assign_label_seq_id()

    output_cif.parent.mkdir(parents=True, exist_ok=True)
    doc = cropped_structure.make_mmcif_document()
    doc.write_file(str(output_cif))

    return {
        "selected_chain_id": selected_chain.name,
        "n_residues_cropped": n_residues,
        "n_atoms_cropped": n_atoms,
        "crop_status": "SUCCESS",
        "error_message": None,
    }


def _extract_prediction_metadata(prediction: dict[str, Any]) -> dict[str, Any]:
    """Extract useful fields from an AlphaFold DB prediction object."""

    return {
        "cif_url": _first_present(prediction, ["cifUrl", "cif_url"]),
        "model_created_date": _first_present(prediction, ["modelCreatedDate", "model_created_date"]),
        "latest_version": _first_present(prediction, ["latestVersion", "latest_version"]),
        "global_metric_value": _first_present(prediction, ["globalMetricValue", "global_metric_value"]),
        "prediction_uniprot_accession": _first_present(
            prediction,
            ["uniprotAccession", "uniprot_accession"],
        ),
    }


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """Build the SH2 mock dataset and return the manifest dataframe."""

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    try:
        payload = fetch_interpro_domains(args.interpro_accession, args.page_size)
    except Exception as exc:
        LOGGER.exception("InterPro request failed for %s", args.interpro_accession)
        row = {
            "uniprot_accession": None,
            "protein_name": None,
            "organism": None,
            "source_database": None,
            "protein_length": None,
            "interpro_accession": args.interpro_accession,
            "domain_index": None,
            "domain_start": None,
            "domain_end": None,
            "domain_length": None,
            "fragment_status": None,
            "in_alphafold": None,
            "afdb_api_url": None,
            "cif_url": None,
            "full_cif_path": None,
            "cropped_cif_path": None,
            "selected_chain_id": None,
            "n_residues_cropped": 0,
            "n_atoms_cropped": 0,
            "download_status": "FAILED",
            "crop_status": "FAILED",
            "error_message": f"InterPro request failed: {exc}",
            "model_created_date": None,
            "latest_version": None,
            "global_metric_value": None,
            "prediction_uniprot_accession": None,
        }
        manifest_rows.append(row)
        df = pd.DataFrame(manifest_rows)
        df.to_csv(args.manifest_path, index=False)
        return df

    raw_results = payload.get("results") or payload.get("data") or []
    LOGGER.info("Number of InterPro records found: %d", len(raw_results))
    parsed_records = parse_interpro_results(payload, args.interpro_accession)
    filtered_records = filter_domain_records(parsed_records)
    LOGGER.info("Number passing filters: %d", len(filtered_records))

    if not raw_results:
        LOGGER.warning("Empty InterPro results for %s", args.interpro_accession)
        df = pd.DataFrame(
            [
                {
                    "uniprot_accession": None,
                    "protein_name": None,
                    "organism": None,
                    "source_database": None,
                    "protein_length": None,
                    "interpro_accession": args.interpro_accession,
                    "domain_index": None,
                    "domain_start": None,
                    "domain_end": None,
                    "domain_length": None,
                    "fragment_status": None,
                    "in_alphafold": None,
                    "afdb_api_url": None,
                    "cif_url": None,
                    "full_cif_path": None,
                    "cropped_cif_path": None,
                    "selected_chain_id": None,
                    "n_residues_cropped": 0,
                    "n_atoms_cropped": 0,
                    "download_status": "FAILED",
                    "crop_status": "FAILED",
                    "error_message": "Empty InterPro results.",
                    "model_created_date": None,
                    "latest_version": None,
                    "global_metric_value": None,
                    "prediction_uniprot_accession": None,
                }
            ]
        )
        df.to_csv(args.manifest_path, index=False)
        return df

    if not filtered_records:
        LOGGER.warning("No InterPro fragments passed filters for %s", args.interpro_accession)
        df = pd.DataFrame(
            columns=[
                "uniprot_accession",
                "protein_name",
                "organism",
                "source_database",
                "protein_length",
                "interpro_accession",
                "domain_index",
                "domain_start",
                "domain_end",
                "domain_length",
                "fragment_status",
                "in_alphafold",
                "afdb_api_url",
                "cif_url",
                "full_cif_path",
                "cropped_cif_path",
                "selected_chain_id",
                "n_residues_cropped",
                "n_atoms_cropped",
                "download_status",
                "crop_status",
                "error_message",
                "model_created_date",
                "latest_version",
                "global_metric_value",
                "prediction_uniprot_accession",
            ]
        )
        df.to_csv(args.manifest_path, index=False)
        return df

    if args.max_records is not None:
        filtered_records = filtered_records[: args.max_records]

    prediction_cache: dict[str, dict[str, Any] | None] = {}
    for record in filtered_records:
        uniprot_accession = record.get("uniprot_accession")
        start = record.get("domain_start")
        end = record.get("domain_end")
        if not isinstance(uniprot_accession, str):
            row = dict(record)
            row.update(
                {
                    "afdb_api_url": None,
                    "cif_url": None,
                    "full_cif_path": None,
                    "cropped_cif_path": None,
                    "selected_chain_id": None,
                    "n_residues_cropped": 0,
                    "n_atoms_cropped": 0,
                    "download_status": "FAILED",
                    "crop_status": "FAILED",
                    "error_message": "Missing UniProt accession in InterPro result.",
                }
            )
            manifest_rows.append(row)
            continue

        prediction: dict[str, Any] | None
        if uniprot_accession not in prediction_cache:
            LOGGER.info("Fetching AlphaFold DB prediction for %s", uniprot_accession)
            try:
                prediction = fetch_afdb_prediction(uniprot_accession)
            except Exception as exc:
                LOGGER.exception("AlphaFold DB lookup failed for %s", uniprot_accession)
                prediction = {"_error": str(exc)}
            prediction_cache[uniprot_accession] = prediction
        else:
            prediction = prediction_cache[uniprot_accession]

        prediction_meta = (
            _extract_prediction_metadata(prediction or {})
            if prediction and "_error" not in prediction
            else {
                "cif_url": None,
                "model_created_date": None,
                "latest_version": None,
                "global_metric_value": None,
                "prediction_uniprot_accession": None,
            }
        )

        afdb_api_url = AFDB_PREDICTION_URL.format(uniprot_accession=uniprot_accession)
        cif_url = prediction_meta["cif_url"]
        full_cif_path = args.raw_dir / f"AF-{uniprot_accession}-F1-model.cif"
        cropped_cif_path = args.processed_dir / f"{uniprot_accession}_SH2_{record.get('domain_index')}_{start}_{end}.cif"

        row = dict(record)
        row.update(
            {
                "afdb_api_url": afdb_api_url,
                "cif_url": cif_url,
                "full_cif_path": str(full_cif_path),
                "cropped_cif_path": str(cropped_cif_path),
                "selected_chain_id": None,
                "n_residues_cropped": 0,
                "n_atoms_cropped": 0,
                "download_status": "SKIPPED",
                "crop_status": "SKIPPED",
                "error_message": None,
                "model_created_date": prediction_meta["model_created_date"],
                "latest_version": prediction_meta["latest_version"],
                "global_metric_value": prediction_meta["global_metric_value"],
                "prediction_uniprot_accession": prediction_meta["prediction_uniprot_accession"],
            }
        )

        if prediction is None:
            row["error_message"] = "Missing AlphaFold DB prediction."
            row["download_status"] = "FAILED"
            row["crop_status"] = "FAILED"
            manifest_rows.append(row)
            continue

        if "_error" in prediction:
            row["error_message"] = f"AlphaFold DB lookup failed: {prediction['_error']}"
            row["download_status"] = "FAILED"
            row["crop_status"] = "FAILED"
            manifest_rows.append(row)
            continue

        if not cif_url:
            row["error_message"] = "Missing cifUrl in AlphaFold DB prediction."
            row["download_status"] = "FAILED"
            row["crop_status"] = "FAILED"
            manifest_rows.append(row)
            continue

        try:
            LOGGER.info("Downloading AlphaFold DB CIF for %s", uniprot_accession)
            download_file(cif_url, full_cif_path, overwrite=args.overwrite)
            row["download_status"] = "SUCCESS"
        except Exception as exc:
            row["download_status"] = "FAILED"
            row["crop_status"] = "FAILED"
            row["error_message"] = f"Failed mmCIF download: {exc}"
            manifest_rows.append(row)
            continue

        try:
            crop_result = crop_mmcif_to_residue_range(full_cif_path, cropped_cif_path, start, end)
        except Exception as exc:
            row["crop_status"] = "FAILED"
            row["error_message"] = f"Failed gemmi parsing or cropping: {exc}"
            manifest_rows.append(row)
            continue

        row.update(crop_result)
        if row["crop_status"] == "SUCCESS":
            LOGGER.info(
                "Cropped domain written: %s -> %s",
                uniprot_accession,
                cropped_cif_path,
            )
        else:
            LOGGER.warning(
                "Skipped or failed accession %s domain %s: %s",
                uniprot_accession,
                record.get("domain_index"),
                row.get("error_message"),
            )

        manifest_rows.append(row)
        _polite_sleep(args.sleep_seconds)

    df = pd.DataFrame(manifest_rows)
    df.to_csv(args.manifest_path, index=False)
    LOGGER.info("Final manifest path: %s", args.manifest_path)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Build a small SH2-domain mock dataset from InterPro and AlphaFold DB.",
    )
    parser.add_argument(
        "--interpro-accession",
        default=DEFAULT_INTERPRO_ACCESSION,
        help="InterPro accession to query (default: IPR000980).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="InterPro page size to request.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/afdb_full"),
        help="Directory for full-length AlphaFold DB mmCIF files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/sh2_cropped"),
        help="Directory for cropped SH2-domain mmCIF files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/tables/sh2_afdb_domain_manifest.csv"),
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously downloaded full-length CIF files.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause between record processing steps.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap the number of filtered records processed.",
    )
    return parser


def main() -> None:
    """Run the command-line interface."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
