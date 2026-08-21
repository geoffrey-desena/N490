#!/usr/bin/env python3
"""
Unpack and inspect the '15 European countries.rar' transmission-network dataset.

Default behaviour:
    1. Find the RAR archive in the current working directory.
    2. Extract it to ./15_european_countries_unpacked/
    3. Print the README, if present.
    4. Inspect every CSV recursively.
    5. Save a compact inventory table as CSV.

The script does not modify the source archive.

RAR extraction requires one of these programs to be installed and available
on PATH: 7z/7zz, unrar, or bsdtar.

Python dependencies:
    pandas
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ARCHIVE = "15 European countries.rar"
DEFAULT_OUTPUT_DIR = "15_european_countries_unpacked"
DEFAULT_HEAD_ROWS = 5

# ---------------------------------------------------------------------
# Output directory for prepared European comparison data
# ---------------------------------------------------------------------

EURO_COMPARISON_DIR = Path.cwd() / "euro-comparison"
EURO_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Archive extraction
# -----------------------------------------------------------------------------

def find_extractor() -> tuple[str, str]:
    """Return (extractor_type, executable_path) for an installed RAR-capable tool."""
    candidates = [
        ("7z", "7z"),
        ("7z", "7zz"),
        ("unrar", "unrar"),
        ("bsdtar", "bsdtar"),
    ]

    for kind, command in candidates:
        executable = shutil.which(command)
        if executable:
            return kind, executable

    raise RuntimeError(
        "No RAR extraction program was found. Install one of: 7-Zip (7z/7zz), "
        "unrar, or bsdtar, then run this script again.\n\n"
        "On macOS with Homebrew, for example:\n"
        "    brew install sevenzip"
    )


def extract_rar(archive_path: Path, output_dir: Path, overwrite: bool = False) -> None:
    """Extract a RAR archive using an available external extraction program."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [p for p in output_dir.rglob("*") if p.is_file()]
    if existing_files and not overwrite:
        print(f"Extraction directory already contains {len(existing_files)} file(s).")
        print("Skipping extraction. Use --overwrite to extract again.\n")
        return

    kind, executable = find_extractor()

    print("=" * 100)
    print("ARCHIVE EXTRACTION")
    print("=" * 100)
    print(f"Archive : {archive_path.resolve()}")
    print(f"Output  : {output_dir.resolve()}")
    print(f"Tool    : {executable}\n")

    if kind == "7z":
        command = [
            executable,
            "x",
            str(archive_path),
            f"-o{output_dir}",
            "-y",
        ]
    elif kind == "unrar":
        command = [
            executable,
            "x",
            "-o+" if overwrite else "-o-",
            str(archive_path),
            str(output_dir) + "/",
        ]
    elif kind == "bsdtar":
        command = [
            executable,
            "-xf",
            str(archive_path),
            "-C",
            str(output_dir),
        ]
    else:
        raise RuntimeError(f"Unsupported extractor type: {kind}")

    result = subprocess.run(command, text=True, capture_output=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"RAR extraction failed with return code {result.returncode}."
        )

    print("Extraction complete.\n")


# -----------------------------------------------------------------------------
# CSV loading helpers
# -----------------------------------------------------------------------------

def detect_encoding(path: Path) -> str:
    """Try a few common encodings and return the first one that decodes cleanly."""
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    raw = path.read_bytes()
    for encoding in encodings:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass

    # latin-1 technically decodes every byte, so this is mostly defensive.
    return "latin-1"


def detect_delimiter(path: Path, encoding: str) -> str | None:
    """Use csv.Sniffer to detect comma/semicolon/tab/pipe delimiters."""
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        sample = f.read(8192)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return None


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Read a CSV while tolerating common delimiter and encoding differences."""
    encoding = detect_encoding(path)
    delimiter = detect_delimiter(path, encoding)

    if delimiter is not None:
        df = pd.read_csv(path, encoding=encoding, sep=delimiter)
        return df, encoding, repr(delimiter)

    # Last-resort automatic delimiter inference.
    df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    return df, encoding, "auto"


# -----------------------------------------------------------------------------
# Inspection
# -----------------------------------------------------------------------------

def print_readmes(root: Path) -> None:
    readmes = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name.lower().startswith("readme")
    )

    if not readmes:
        return

    print("=" * 100)
    print("README FILES")
    print("=" * 100)

    for path in readmes:
        encoding = detect_encoding(path)
        text = path.read_text(encoding=encoding, errors="replace")
        print(f"\n--- {path.relative_to(root)} ---\n")
        print(text.rstrip())
        print()


def inspect_csv(path: Path, root: Path, head_rows: int) -> dict:
    df, encoding, delimiter = read_csv_robust(path)

    relative_path = path.relative_to(root)
    file_size_kb = path.stat().st_size / 1024
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())

    print("\n" + "=" * 100)
    print(relative_path)
    print("=" * 100)
    print(f"Size             : {file_size_kb:,.1f} kB")
    print(f"Encoding         : {encoding}")
    print(f"Delimiter        : {delimiter}")
    print(f"Rows             : {len(df):,}")
    print(f"Columns          : {len(df.columns):,}")
    print(f"Duplicate rows   : {duplicate_rows:,}")
    print(f"Missing cells    : {missing_cells:,}")

    print("\nColumns and dtypes:")
    for column in df.columns:
        n_missing = int(df[column].isna().sum())
        n_unique = int(df[column].nunique(dropna=True))
        print(
            f"  {str(column):<32} "
            f"dtype={str(df[column].dtype):<12} "
            f"unique={n_unique:<7} "
            f"missing={n_missing}"
        )

    if len(df) > 0:
        print(f"\nFirst {min(head_rows, len(df))} row(s):")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.max_colwidth", 40,
        ):
            print(df.head(head_rows).to_string(index=False))

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\nNumeric summary:")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
        ):
            print(numeric.describe().T.to_string())

    return {
        "file": str(relative_path),
        "size_kB": round(file_size_kb, 3),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": " | ".join(map(str, df.columns)),
        "encoding": encoding,
        "delimiter": delimiter,
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
    }


def compare_schemas(csv_paths: list[Path], root: Path) -> None:
    """Print which files share the same set/order of columns."""
    schema_groups: dict[tuple[str, ...], list[str]] = {}

    for path in csv_paths:
        try:
            df, _, _ = read_csv_robust(path)
            schema = tuple(map(str, df.columns))
            schema_groups.setdefault(schema, []).append(str(path.relative_to(root)))
        except Exception as exc:
            schema_groups.setdefault((f"<READ ERROR: {exc}>",), []).append(
                str(path.relative_to(root))
            )

    print("\n" + "=" * 100)
    print("SCHEMA COMPARISON")
    print("=" * 100)
    print(f"Distinct CSV schemas: {len(schema_groups)}\n")

    for i, (schema, files) in enumerate(schema_groups.items(), start=1):
        print(f"Schema {i}: {len(files)} file(s)")
        print("  Columns:")
        for col in schema:
            print(f"    - {col}")
        print("  Files:")
        for filename in files:
            print(f"    - {filename}")
        print()


def inspect_dataset(root: Path, head_rows: int) -> None:
    print_readmes(root)

    csv_paths = sorted(
        p for p in root.rglob("*.csv")
        if p.is_file()
    )

    print("\n" + "=" * 100)
    print("DATASET INVENTORY")
    print("=" * 100)
    print(f"Root directory : {root.resolve()}")
    print(f"CSV files      : {len(csv_paths)}")

    if not csv_paths:
        print("No CSV files found.")
        return

    inventory = []
    failures = []

    for path in csv_paths:
        try:
            inventory.append(inspect_csv(path, root, head_rows))
        except Exception as exc:
            failures.append((path, exc))
            print("\n" + "=" * 100)
            print(path.relative_to(root))
            print("=" * 100)
            print(f"FAILED TO READ: {type(exc).__name__}: {exc}")

    compare_schemas(csv_paths, root)

    if inventory:
        inventory_df = pd.DataFrame(inventory)
        inventory_path = root / "dataset_inventory.csv"
        inventory_df.to_csv(inventory_path, index=False)

        print("=" * 100)
        print("COMPACT INVENTORY")
        print("=" * 100)
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(
                inventory_df[
                    [
                        "file",
                        "rows",
                        "columns",
                        "size_kB",
                        "duplicate_rows",
                        "missing_cells",
                    ]
                ].to_string(index=False)
            )

        print(f"\nSaved inventory: {inventory_path.resolve()}")

    if failures:
        print("\nFiles that could not be read:")
        for path, exc in failures:
            print(f"  - {path.relative_to(root)}: {type(exc).__name__}: {exc}")


# =====================================================================
# LOAD AND SAVE STANDARDIZED COUNTRY DATA
# =====================================================================

def load_country_csv(csv_path):
    """
    Load one country edge-list CSV.

    The source files do not contain column headers.

    Columns:
        node_i      : first endpoint of branch
        node_j      : second endpoint of branch
        voltage_kv  : branch voltage level in kV
    """

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["node_i", "node_j", "voltage_kv"],
        encoding="utf-8-sig",
    )

    # Make sure all source fields are integers.
    df = df.astype(
        {
            "node_i": int,
            "node_j": int,
            "voltage_kv": int,
        }
    )

    return df


def prepare_country_dataframes(extracted_dir, output_dir):
    """
    Load every country edge-list CSV into a standardized pandas
    DataFrame and save the results as pickle files.

    Creates:
        euro-comparison/
            Albania.pkl
            Belgium.pkl
            ...
            european_networks.pkl

    european_networks.pkl contains a dictionary:

        {
            "Albania": dataframe,
            "Belgium": dataframe,
            ...
        }
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    country_data = {}

    # Files generated by this script that should NOT be interpreted
    # as transmission-network edge lists.
    excluded_csvs = {
        "dataset_inventory.csv",
    }

    csv_files = sorted(
        path
        for path in extracted_dir.glob("*.csv")
        if path.name not in excluded_csvs
    )

    print("\n")
    print("=" * 100)
    print("PREPARING COUNTRY DATAFRAMES")
    print("=" * 100)

    for csv_path in csv_files:

        country = csv_path.stem

        df = load_country_csv(csv_path)

        # -------------------------------------------------------------
        # Sanity check
        # -------------------------------------------------------------

        if len(df.columns) != 3:
            print(
                f"Skipping {csv_path.name}: "
                f"expected 3 columns, found {len(df.columns)}"
            )
            continue

        country_data[country] = df

        # -------------------------------------------------------------
        # Save individual country DataFrame
        # -------------------------------------------------------------

        output_path = output_dir / f"{country}.pkl"

        df.to_pickle(output_path)

        nodes = set(df["node_i"]) | set(df["node_j"])
        voltages = sorted(df["voltage_kv"].unique().tolist())

        print(
            f"{country:<25} "
            f"branches={len(df):>5}   "
            f"nodes={len(nodes):>5}   "
            f"voltages={voltages}"
        )

    # -----------------------------------------------------------------
    # Save all countries together
    # -----------------------------------------------------------------

    combined_path = output_dir / "european_networks.pkl"

    pd.to_pickle(country_data, combined_path)

    print("\n")
    print(f"Saved {len(country_data)} country DataFrames.")
    print(f"Individual pickle files:")
    print(f"  {output_dir}")

    print("\n")
    print("Combined country dictionary:")
    print(f"  {combined_path}")

    return country_data

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unpack and inspect the 15-European-countries network dataset."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(DEFAULT_ARCHIVE),
        help=f"RAR archive path (default: {DEFAULT_ARCHIVE!r})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Extraction directory (default: {DEFAULT_OUTPUT_DIR!r})",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=DEFAULT_HEAD_ROWS,
        help=f"Rows to print from each CSV (default: {DEFAULT_HEAD_ROWS})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Extract again even if the output directory already contains files.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction and inspect an already-extracted --output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    archive_path = args.archive.expanduser()
    output_dir = args.output.expanduser()

    if not args.no_extract:
        extract_rar(archive_path, output_dir, overwrite=args.overwrite)
    elif not output_dir.exists():
        raise FileNotFoundError(
            f"--no-extract was specified, but the directory does not exist: "
            f"{output_dir.resolve()}"
        )

    inspect_dataset(output_dir, head_rows=max(args.head, 0))
    
    # =====================================================================
    # PREPARE PICKLED DATA
    # =====================================================================
    
    country_data = prepare_country_dataframes(
        Path.cwd() / "15_european_countries_unpacked",
        EURO_COMPARISON_DIR,
    )


if __name__ == "__main__":
    main()
