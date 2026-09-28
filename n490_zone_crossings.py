# -*- coding: utf-8 -*-
"""Count N490 AC connections between bidding zones by voltage.

Use the same ``nordic490.N490`` model and bus-zone metadata as
``n490_border_crossings.py``. Unlike that script, domestic bidding-zone
boundaries are included, and parallel lines are collapsed independently at
each voltage. Source 380 kV becomes synthetic 400 kV; 275 kV becomes 300 kV.

Run from the nordic-grid project environment::

    python scripts/n490_zone_crossings.py

The default output directory is ``data/raw/n490`` in the user's project.
The resulting pickle is read by line_selection.py; the matching CSV makes
its targets easy to review. Zero-count zone pairs are retained at each
voltage so the absence of an N490 AC connection is explicit.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from nordic490 import N490


N490_OUTPUT_DIR = Path(
    "/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490"
)
OUTPUT_STEM = "N490_zone_crossings_by_voltage"
MODEL_VOLTAGES_KV = (132, 220, 300, 400)
SOURCE_TO_MODEL_VOLTAGE = {
    132: 132,
    220: 220,
    275: 300,
    300: 300,
    380: 400,
    400: 400,
}


def resolve_line_endpoint_columns(lines: pd.DataFrame) -> tuple[str, str]:
    """Identify the N490 AC line terminal columns."""
    for first, second in (
        ("bus0", "bus1"), ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"), ("bus1", "bus2"),
        ("fbus", "tbus"), ("from", "to"),
    ):
        if first in lines.columns and second in lines.columns:
            return first, second
    raise ValueError(
        "Could not identify N490 line endpoint columns; found "
        f"{list(lines.columns)}"
    )


def resolve_voltage_column(lines: pd.DataFrame) -> str:
    """Locate nominal voltage on N490's AC line table."""
    for column in ("Vbase", "vn_kv", "nominal_kv", "voltage_kv"):
        if column in lines.columns:
            return column
    raise ValueError(
        "N490 line table has no voltage column; expected Vbase, vn_kv, "
        "nominal_kv or voltage_kv"
    )


def bus_lookup_from_n490(bus: pd.DataFrame) -> pd.DataFrame:
    """Replicate the border script's index-to-bidding-zone lookup."""
    if not {"bidz", "country"}.issubset(bus.columns):
        raise ValueError("N490 model.bus must contain bidz and country")
    if bus.index.has_duplicates or bus[["bidz", "country"]].isna().any().any():
        raise ValueError("N490 bus IDs must be unique and zone/country nonmissing")
    index = pd.to_numeric(pd.Index(bus.index), errors="raise")
    if not np.isfinite(index).all() or (index % 1 != 0).any():
        raise ValueError("N490 bus IDs must be finite integers")
    lookup = bus[["bidz", "country"]].copy()
    lookup.insert(0, "bus_id", index.astype(int))
    lookup["zone"] = lookup.pop("bidz").astype(str).str.strip().str.upper()
    lookup["country"] = lookup.country.astype(str).str.strip().str.upper()
    if lookup.zone.eq("").any() or lookup.country.eq("").any():
        raise ValueError("N490 contains blank bidding zones or countries")
    if lookup.groupby("zone").country.nunique().gt(1).any():
        raise ValueError("A N490 bidding zone maps to multiple countries")
    return lookup.reset_index(drop=True)


def simple_ac_edges_by_voltage(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse parallel circuits by (model voltage, unordered bus pair)."""
    if lines.empty:
        raise ValueError("N490 line table is empty")
    first, second = resolve_line_endpoint_columns(lines)
    voltage_column = resolve_voltage_column(lines)
    raw = lines[[first, second, voltage_column]].copy()
    if raw.isna().any().any():
        raise ValueError("N490 AC lines contain missing bus endpoints or voltage")
    ids = raw[[first, second]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(ids.to_numpy(dtype=float)).all() or not ids.mod(1).eq(0).all().all():
        raise ValueError("N490 line bus endpoints must be finite integer IDs")
    endpoint_pairs = np.sort(ids.to_numpy(dtype=int), axis=1)
    voltages = pd.to_numeric(raw[voltage_column], errors="raise")
    if not np.isfinite(voltages.to_numpy(dtype=float)).all() or not voltages.mod(1).eq(0).all():
        raise ValueError("N490 AC line voltages must be finite integer kV")
    source_kv = voltages.astype(int)
    unknown = sorted(set(source_kv) - set(SOURCE_TO_MODEL_VOLTAGE))
    if unknown:
        raise ValueError(f"Unmapped N490 AC line voltages: {unknown}")
    edges = pd.DataFrame({
        "vn_kv": source_kv.map(SOURCE_TO_MODEL_VOLTAGE).to_numpy(dtype=int),
        "bus_i": endpoint_pairs[:, 0], "bus_j": endpoint_pairs[:, 1],
    })
    edges = edges.loc[edges.bus_i.ne(edges.bus_j)]
    return edges.drop_duplicates(["vn_kv", "bus_i", "bus_j"]).reset_index(drop=True)


def attach_bus_zones(edges: pd.DataFrame, bus_lookup: pd.DataFrame) -> pd.DataFrame:
    """Join N490 zone/country metadata to both ends of every simple edge."""
    result = edges.copy()
    for terminal in ("i", "j"):
        side = bus_lookup.rename(columns={
            "bus_id": f"bus_{terminal}", "zone": f"zone_{terminal}",
            "country": f"country_{terminal}",
        })
        result = result.merge(side, on=f"bus_{terminal}", how="left", validate="many_to_one")
    if result[["zone_i", "zone_j", "country_i", "country_j"]].isna().any().any():
        raise ValueError("Some N490 AC line endpoints have no matching model.bus row")
    return result


def zone_crossing_summary(model: N490) -> pd.DataFrame:
    """Count voltage-specific simple edges for every distinct zone pair."""
    bus_lookup = bus_lookup_from_n490(model.bus)
    edges = attach_bus_zones(simple_ac_edges_by_voltage(model.line), bus_lookup)
    interzone = edges.loc[edges.zone_i.ne(edges.zone_j)].copy()
    pairs = np.sort(interzone[["zone_i", "zone_j"]].to_numpy(dtype=str), axis=1)
    interzone["zone_a"] = pairs[:, 0]
    interzone["zone_b"] = pairs[:, 1]
    counted = (interzone.groupby(["vn_kv", "zone_a", "zone_b"], as_index=False)
               .size().rename(columns={"size": "n_crossings"}))

    zones = bus_lookup[["zone", "country"]].drop_duplicates().set_index("zone")
    rows = [
        {"vn_kv": voltage, "zone_a": zone_a, "zone_b": zone_b,
         "country_a": str(zones.at[zone_a, "country"]),
         "country_b": str(zones.at[zone_b, "country"])}
        for voltage in MODEL_VOLTAGES_KV
        for zone_a, zone_b in combinations(sorted(zones.index), 2)
    ]
    summary = pd.DataFrame(rows).merge(
        counted, on=["vn_kv", "zone_a", "zone_b"], how="left", validate="one_to_one",
    )
    summary["n_crossings"] = summary.n_crossings.fillna(0).astype(int)
    summary["crossing_scope"] = np.where(
        summary.country_a.eq(summary.country_b), "domestic", "international",
    )
    if int(summary.n_crossings.sum()) != len(interzone):
        raise RuntimeError("Voltage-specific simple AC interzone edge count changed")
    return summary.sort_values(["vn_kv", "zone_a", "zone_b"]).reset_index(drop=True)


def analyze_zone_crossings(
    model: N490,
    output_dir: Path = N490_OUTPUT_DIR,
) -> pd.DataFrame:
    """Save the counts alongside the existing N490 statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = zone_crossing_summary(model)
    summary.to_csv(output_dir / f"{OUTPUT_STEM}.csv", index=False)
    summary.to_pickle(output_dir / f"{OUTPUT_STEM}.pkl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--output-dir", type=Path, default=N490_OUTPUT_DIR)
    args = parser.parse_args()
    print(f"Loading N490 ({args.year})...", flush=True)
    summary = analyze_zone_crossings(N490(year=args.year), args.output_dir)
    print("N490 simple-graph AC interzone crossings by voltage:")
    print(summary.groupby(["vn_kv", "crossing_scope"]).agg(
        zone_pairs=("n_crossings", "size"),
        connected_pairs=("n_crossings", lambda values: int(values.gt(0).sum())),
        ac_routes=("n_crossings", "sum"),
    ).to_string())
    print(f"Saved {OUTPUT_STEM}.csv and {OUTPUT_STEM}.pkl in {args.output_dir}")


if __name__ == "__main__":
    main()
