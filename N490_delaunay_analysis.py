# -*- coding: utf-8 -*-
"""
Analyze overlap between N490 transmission branches and voltage-specific
Delaunay candidate sets.

Purpose
-------
The voltage-specific Delaunay candidate sets are created separately by
``N490_delaunay_creation.py``. This script loads those saved products and
compares them with the actual N490 AC transmission network.

The analysis is performed independently for the 220, 300, and 380 kV
networks.

For each voltage level:

1. Load the saved voltage-specific geographic nodes.
2. Load Delaunay-1 through Delaunay-5 candidate edge sets.
3. Load the saved minimum spanning tree.
4. Select the actual N490 branches at that voltage.
5. Map each N490 branch from its original ``bus0`` and ``bus1`` IDs to the
   corresponding voltage-specific geographic node IDs.
6. Classify each actual branch by Delaunay overlap:
       Delaunay1
       Delaunay2
       Delaunay3
       Delaunay4
       Delaunay5
       None
7. Reclassify Delaunay-1 branches that are also in the MST as ``MST``.
8. Calculate branch counts and proportions by category.

Overlap is based on network topology rather than geometric equality. An N490
branch overlaps a candidate edge when the two connect the same pair of
voltage-specific geographic nodes. The actual N490 branch may therefore have
a curved geographic path while the Delaunay candidate is a straight segment.

Parallel N490 branches are retained as separate observations because the
statistics describe transmission branches rather than unique geographic
corridors.

Outputs
-------
The script writes:

``n490_delaunay_overlap_analysis/n490_220kv_overlap.geojson``
``n490_delaunay_overlap_analysis/n490_300kv_overlap.geojson``
``n490_delaunay_overlap_analysis/n490_380kv_overlap.geojson``
    Actual N490 branches with overlap classifications.

``n490_delaunay_overlap_analysis/n490_delaunay_overlap_counts.csv``
    Counts by voltage and mutually exclusive overlap category.

``n490_delaunay_overlap_analysis/n490_delaunay_overlap_proportions.csv``
    Proportions by voltage and mutually exclusive overlap category.

``n490_delaunay_overlap_analysis/N490_Delaunay_stats_by_voltage.pkl``
    Wide-form per-voltage proportion table for later model construction.

``n490_delaunay_overlap_analysis/N490_Delaunay_edge_classifications.pkl``
    Branch-level classification table without GeoJSON serialization.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import matplotlib.pyplot as plt

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DELAUNAY_DIR = Path("n490_delaunay_analysis")

OUTPUT_DIR = Path("n490_delaunay_overlap_analysis")

SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:3845"

VOLTAGE_LEVELS = [220, 300, 380]
DELAUNAY_K_VALUES = [1, 2, 3, 4, 5]

CATEGORY_ORDER = [
    "MST",
    "Delaunay1",
    "Delaunay2",
    "Delaunay3",
    "Delaunay4",
    "Delaunay5",
    "None",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def canonical_pair(
    u: int,
    v: int,
) -> tuple[int, int]:
    """
    Return a canonical unordered node pair.
    """
    return tuple(sorted((int(u), int(v))))


def parse_bus_ids(
    value,
) -> tuple[int, ...]:
    """
    Parse the ``bus_ids`` field saved in the voltage-node GeoJSON files.

    ``N490_delaunay_creation.py`` saves tuple-valued bus IDs as a
    semicolon-separated string before writing GeoJSON.

    Parameters
    ----------
    value:
        Saved bus-ID value.

    Returns
    -------
    tuple[int, ...]
        Original N490 bus IDs represented by the geographic node.
    """
    if value is None:
        return tuple()

    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(
            int(float(x))
            for x in value
            if pd.notna(x)
        )

    if isinstance(value, (int, np.integer)):
        return (int(value),)

    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return tuple()
        return (int(value),)

    text = str(value).strip()

    if not text:
        return tuple()

    values = []

    for part in text.split(";"):
        part = part.strip()

        if not part:
            continue

        values.append(
            int(float(part))
        )

    return tuple(values)


def edge_set(
    edges: gpd.GeoDataFrame,
) -> set[tuple[int, int]]:
    """
    Convert a candidate edge GeoDataFrame to unordered endpoint pairs.
    """
    required = {"from_node", "to_node"}
    missing = required - set(edges.columns)

    if missing:
        raise ValueError(
            "Candidate edge table is missing columns: "
            f"{sorted(missing)}"
        )

    return {
        canonical_pair(u, v)
        for u, v in zip(
            edges["from_node"],
            edges["to_node"],
        )
    }


# ---------------------------------------------------------------------
# Load saved voltage-specific Delaunay products
# ---------------------------------------------------------------------

def load_voltage_delaunay_sets(
    voltage_kv: int,
    delaunay_dir: Path = DELAUNAY_DIR,
    target_crs: str = TARGET_CRS,
) -> tuple[
    gpd.GeoDataFrame,
    dict[int, gpd.GeoDataFrame],
    gpd.GeoDataFrame,
]:
    """
    Load saved nodes, Delaunay-1 through Delaunay-5, and MST for one voltage.

    Parameters
    ----------
    voltage_kv:
        Voltage network to load.
    delaunay_dir:
        Directory created by ``N490_delaunay_creation.py``.
    target_crs:
        Projected analysis CRS.

    Returns
    -------
    nodes:
        Voltage-specific geographic node table.
    delaunay_sets:
        Dictionary keyed by k = 1...5.
    mst:
        Saved voltage-specific minimum spanning tree.
    """
    voltage_tag = str(int(voltage_kv))

    node_path = (
        delaunay_dir
        / f"n490_{voltage_tag}kv_nodes.geojson"
    )

    mst_path = (
        delaunay_dir
        / f"n490_{voltage_tag}kv_mst.geojson"
    )

    if not node_path.exists():
        raise FileNotFoundError(
            f"Missing voltage-node file: {node_path}"
        )

    if not mst_path.exists():
        raise FileNotFoundError(
            f"Missing MST file: {mst_path}"
        )

    nodes = gpd.read_file(node_path)

    if nodes.crs is None:
        raise ValueError(
            f"Node file has no CRS: {node_path}"
        )

    nodes = nodes.to_crs(target_crs)

    required_node_cols = {
        "node_id",
        "bus_ids",
        "geometry",
    }

    missing = required_node_cols - set(nodes.columns)

    if missing:
        raise ValueError(
            f"{node_path} is missing columns: "
            f"{sorted(missing)}"
        )

    nodes["node_id"] = (
        pd.to_numeric(
            nodes["node_id"],
            errors="raise",
        )
        .astype(int)
    )

    delaunay_sets = {}

    for k in DELAUNAY_K_VALUES:

        path = (
            delaunay_dir
            / f"n490_{voltage_tag}kv_delaunay{k}.geojson"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Delaunay-{k} file: {path}"
            )

        edges = gpd.read_file(path)

        if edges.crs is None:
            raise ValueError(
                f"Delaunay-{k} file has no CRS: {path}"
            )

        edges = edges.to_crs(target_crs)

        edges["from_node"] = (
            pd.to_numeric(
                edges["from_node"],
                errors="raise",
            )
            .astype(int)
        )

        edges["to_node"] = (
            pd.to_numeric(
                edges["to_node"],
                errors="raise",
            )
            .astype(int)
        )

        delaunay_sets[k] = edges

    mst = gpd.read_file(mst_path)

    if mst.crs is None:
        raise ValueError(
            f"MST file has no CRS: {mst_path}"
        )

    mst = mst.to_crs(target_crs)

    mst["from_node"] = (
        pd.to_numeric(
            mst["from_node"],
            errors="raise",
        )
        .astype(int)
    )

    mst["to_node"] = (
        pd.to_numeric(
            mst["to_node"],
            errors="raise",
        )
        .astype(int)
    )

    return nodes, delaunay_sets, mst


# ---------------------------------------------------------------------
# Connect original N490 bus IDs to geographic node IDs
# ---------------------------------------------------------------------

def build_bus_to_node_map(
    nodes: gpd.GeoDataFrame,
) -> dict[int, int]:
    """
    Map every original N490 bus represented in a node file to ``node_id``.

    Multiple original buses may map to the same node when they occupy the
    same physical substation location.
    """
    bus_to_node = {}

    for _, row in nodes.iterrows():

        node_id = int(row["node_id"])

        bus_ids = parse_bus_ids(
            row["bus_ids"]
        )

        if not bus_ids:
            raise ValueError(
                f"Node {node_id} does not contain any bus IDs."
            )

        for bus_id in bus_ids:

            if bus_id in bus_to_node:
                previous = bus_to_node[bus_id]

                if previous != node_id:
                    raise ValueError(
                        f"N490 bus {bus_id} maps to more than one "
                        f"geographic node: {previous} and {node_id}."
                    )

            bus_to_node[bus_id] = node_id

    return bus_to_node


# ---------------------------------------------------------------------
# Build actual N490 branch geometry
# ---------------------------------------------------------------------

def build_n490_voltage_lines(
    lines: pd.DataFrame,
    buses: pd.DataFrame,
    voltage_kv: int,
    source_crs: str = SOURCE_CRS,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Build the actual N490 AC branch GeoDataFrame for one voltage level.

    Parameters
    ----------
    lines:
        ``N490.line`` table.
    buses:
        ``N490.bus`` table.
    voltage_kv:
        Voltage network to extract.
    source_crs:
        CRS of N490 latitude/longitude coordinates.
    target_crs:
        Projected analysis CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Actual N490 branches at the requested voltage.

    Notes
    -----
    ``bus0`` and ``bus1`` are preserved because overlap classification is
    based on exact network topology. Line geometry is retained for diagnostic
    maps and GeoJSON output only.
    """
    required_line_cols = {
        "Vbase",
        "bus0",
        "bus1",
        "lat",
        "lon",
    }

    missing = required_line_cols - set(lines.columns)

    if missing:
        raise ValueError(
            "N490 line table is missing required columns: "
            f"{sorted(missing)}"
        )

    lines_work = lines.copy()

    line_voltage = pd.to_numeric(
        lines_work["Vbase"],
        errors="coerce",
    )

    lines_work = lines_work[
        np.isclose(
            line_voltage,
            float(voltage_kv),
            equal_nan=False,
        )
    ].copy()

    if lines_work.empty:
        raise ValueError(
            f"No N490 branches found at {voltage_kv:g} kV."
        )

    # Preserve the original N490 line index.
    lines_work["line_id"] = lines_work.index

    geometries = []
    valid_indices = []

    for idx, row in lines_work.iterrows():

        lats = row["lat"]
        lons = row["lon"]

        if lats is None or lons is None:
            continue

        points = [
            (float(lon), float(lat))
            for lat, lon in zip(lats, lons)
            if pd.notna(lat) and pd.notna(lon)
        ]

        if len(points) < 2:
            continue

        geometries.append(
            LineString(points)
        )

        valid_indices.append(idx)

    if not valid_indices:
        raise ValueError(
            f"No valid N490 line geometries found at "
            f"{voltage_kv:g} kV."
        )

    lines_work = (
        lines_work
        .loc[valid_indices]
        .copy()
    )

    gdf = gpd.GeoDataFrame(
        lines_work,
        geometry=geometries,
        crs=source_crs,
    )

    # Add useful terminal-bus information for diagnostics.
    if "name" in buses.columns:

        bus_name = buses["name"].to_dict()

        gdf["bus0_name"] = (
            gdf["bus0"]
            .map(bus_name)
        )

        gdf["bus1_name"] = (
            gdf["bus1"]
            .map(bus_name)
        )

    return gdf.to_crs(target_crs)


# ---------------------------------------------------------------------
# Map actual N490 branches onto voltage-specific nodes
# ---------------------------------------------------------------------

def map_n490_lines_to_nodes(
    n490_lines: gpd.GeoDataFrame,
    bus_to_node: dict[int, int],
    voltage_kv: int,
) -> gpd.GeoDataFrame:
    """
    Map N490 ``bus0``/``bus1`` IDs to geographic Delaunay node IDs.

    The operation is exact; no geographic snapping is performed.
    """
    out = n490_lines.copy()

    out["bus0"] = (
        pd.to_numeric(
            out["bus0"],
            errors="raise",
        )
        .astype(int)
    )

    out["bus1"] = (
        pd.to_numeric(
            out["bus1"],
            errors="raise",
        )
        .astype(int)
    )

    out["from_node"] = out["bus0"].map(
        bus_to_node
    )

    out["to_node"] = out["bus1"].map(
        bus_to_node
    )

    unmatched = out[
        out["from_node"].isna()
        | out["to_node"].isna()
    ].copy()

    if not unmatched.empty:

        diagnostic_cols = [
            col
            for col in [
                "line_id",
                "bus0",
                "bus1",
                "bus0_name",
                "bus1_name",
                "Vbase",
            ]
            if col in unmatched.columns
        ]

        print(
            f"\nUnmatched {voltage_kv:g} kV branches"
        )
        print("-" * 40)

        print(
            unmatched[
                diagnostic_cols
            ].to_string(index=False)
        )

        raise ValueError(
            f"{len(unmatched)} {voltage_kv:g} kV N490 branches "
            "could not be mapped to the saved voltage-specific nodes. "
            "The overlap denominator would therefore be incomplete."
        )

    out["from_node"] = (
        out["from_node"].astype(int)
    )

    out["to_node"] = (
        out["to_node"].astype(int)
    )

    self_loops = out[
        out["from_node"]
        == out["to_node"]
    ]

    if not self_loops.empty:
        raise ValueError(
            f"{len(self_loops)} {voltage_kv:g} kV N490 branches "
            "map both endpoints to the same geographic node."
        )

    out["node_pair"] = [
        canonical_pair(u, v)
        for u, v in zip(
            out["from_node"],
            out["to_node"],
        )
    ]

    return out


# ---------------------------------------------------------------------
# Classify overlap
# ---------------------------------------------------------------------

def classify_n490_delaunay_overlap(
    n490_edges: gpd.GeoDataFrame,
    delaunay_sets: dict[int, gpd.GeoDataFrame],
    mst_edges: gpd.GeoDataFrame,
    voltage_kv: int,
) -> gpd.GeoDataFrame:
    """
    Classify every N490 branch by overlap with voltage-specific candidate sets.

    Classification occurs in two stages.

    First, each branch is assigned one of:

    - Delaunay1
    - Delaunay2
    - Delaunay3
    - Delaunay4
    - Delaunay5
    - None

    Second, branches classified as Delaunay1 and also belonging to the MST
    are recategorized as ``MST``.

    The final categories are therefore mutually exclusive.
    """
    out = n490_edges.copy()

    candidate_pairs = {
        k: edge_set(edges)
        for k, edges in delaunay_sets.items()
    }

    mst_pairs = edge_set(
        mst_edges
    )

    # -------------------------------------------------------------
    # Record raw membership explicitly for diagnostics.
    # -------------------------------------------------------------
    for k in DELAUNAY_K_VALUES:

        out[f"overlap_delaunay{k}"] = (
            out["node_pair"].isin(
                candidate_pairs[k]
            )
        )

    out["overlap_mst"] = (
        out["node_pair"].isin(
            mst_pairs
        )
    )

    # -------------------------------------------------------------
    # The Delaunay-k sets should be mutually exclusive because k is
    # defined as exact shortest-path distance in Delaunay-1.
    # -------------------------------------------------------------
    overlap_columns = [
        f"overlap_delaunay{k}"
        for k in DELAUNAY_K_VALUES
    ]

    membership_count = (
        out[overlap_columns]
        .sum(axis=1)
    )

    ambiguous = out[
        membership_count > 1
    ]

    if not ambiguous.empty:
        raise RuntimeError(
            f"{len(ambiguous)} {voltage_kv:g} kV N490 branches "
            "appear in more than one Delaunay-k set. "
            "The saved candidate sets are not mutually exclusive."
        )

    # -------------------------------------------------------------
    # Initial Delaunay classification.
    # -------------------------------------------------------------
    out["delaunay_category"] = "None"

    for k in DELAUNAY_K_VALUES:

        mask = out[
            f"overlap_delaunay{k}"
        ]

        out.loc[
            mask,
            "delaunay_category",
        ] = f"Delaunay{k}"

    # -------------------------------------------------------------
    # MST must be a subset of Delaunay-1.
    # -------------------------------------------------------------
    invalid_mst = out[
        out["overlap_mst"]
        & ~out["overlap_delaunay1"]
    ]

    if not invalid_mst.empty:
        raise RuntimeError(
            f"{len(invalid_mst)} {voltage_kv:g} kV N490 branches "
            "overlap the saved MST but not the saved Delaunay-1 set."
        )

    # -------------------------------------------------------------
    # Final mutually exclusive category.
    # -------------------------------------------------------------
    out["category"] = (
        out["delaunay_category"]
    )

    out.loc[
        out["overlap_mst"],
        "category",
    ] = "MST"

    return out


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_voltage_overlap(
    classified: gpd.GeoDataFrame,
    voltage_kv: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate mutually exclusive category counts and proportions.
    """
    counts = (
        classified["category"]
        .value_counts()
        .reindex(
            CATEGORY_ORDER,
            fill_value=0,
        )
    )

    total = int(
        counts.sum()
    )

    count_table = (
        counts
        .rename("count")
        .rename_axis("category")
        .reset_index()
    )

    count_table.insert(
        0,
        "Vbase",
        int(voltage_kv),
    )

    count_table["total_branches"] = total

    proportion_table = (
        count_table[
            ["Vbase", "category"]
        ]
        .copy()
    )

    proportion_table["proportion"] = (
        count_table["count"] / total
        if total
        else 0.0
    )

    return count_table, proportion_table


def build_wide_statistics(
    classified_by_voltage: dict[int, gpd.GeoDataFrame],
) -> pd.DataFrame:
    """
    Build the wide per-voltage quota table.

    Final overlap categories are mutually exclusive. Consequently,
    ``Delaunay1`` does NOT include MST branches.

    A separate ``Delaunay1_including_MST`` column is included for comparison
    with the older analysis convention.
    """
    rows = []

    for voltage_kv, classified in (
        classified_by_voltage.items()
    ):

        total = len(classified)

        if total == 0:
            continue

        final_props = (
            classified["category"]
            .value_counts(normalize=True)
        )

        raw_d1_share = float(
            classified[
                "overlap_delaunay1"
            ].mean()
        )

        row = {
            "Vbase": int(voltage_kv),
            "Total_Branches": int(total),
            "MST": float(
                final_props.get(
                    "MST",
                    0.0,
                )
            ),
            "Delaunay1": float(
                final_props.get(
                    "Delaunay1",
                    0.0,
                )
            ),
            "Delaunay2": float(
                final_props.get(
                    "Delaunay2",
                    0.0,
                )
            ),
            "Delaunay3": float(
                final_props.get(
                    "Delaunay3",
                    0.0,
                )
            ),
            "Delaunay4": float(
                final_props.get(
                    "Delaunay4",
                    0.0,
                )
            ),
            "Delaunay5": float(
                final_props.get(
                    "Delaunay5",
                    0.0,
                )
            ),
            "None": float(
                final_props.get(
                    "None",
                    0.0,
                )
            ),

            # Useful for comparison with the older workflow, in which
            # "Delaunay" included the MST before MST was subtracted.
            "Delaunay1_including_MST": raw_d1_share,
        }

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("Vbase")
        .reset_index(drop=True)
    )

def build_average_overlap_statistics(
    stats_by_voltage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the mean overlap proportion across voltage levels for each
    mutually exclusive Delaunay/MST category.

    Parameters
    ----------
    stats_by_voltage:
        Wide-form voltage-specific overlap statistics produced by
        ``build_wide_statistics()``.

    Returns
    -------
    pandas.DataFrame
        One row per edge-set category containing the mean overlap
        proportion and percentage across all analyzed voltage levels.
    """
    categories = [
        "MST",
        "Delaunay1",
        "Delaunay2",
        "Delaunay3",
        "Delaunay4",
        "Delaunay5",
    ]

    labels = {
        "MST": "MST",
        "Delaunay1": "Delaunay",
        "Delaunay2": "Delaunay 2",
        "Delaunay3": "Delaunay 3",
        "Delaunay4": "Delaunay 4",
        "Delaunay5": "Delaunay 5",
    }

    rows = []

    for category in categories:
        mean_proportion = float(
            stats_by_voltage[category].mean()
        )

        rows.append(
            {
                "edge_set": labels[category],
                "average_proportion": mean_proportion,
                "average_percentage": 100.0 * mean_proportion,
            }
        )

    averages = pd.DataFrame(rows)

    return averages


def print_voltage_summary(
    classified: gpd.GeoDataFrame,
    voltage_kv: int,
) -> None:
    """
    Print overlap diagnostics for one voltage network.
    """
    print("\n" + "=" * 72)
    print(
        f"N490 {voltage_kv:g} kV overlap analysis"
    )
    print("=" * 72)

    total = len(classified)

    print(
        f"N490 branches:                   {total}"
    )

    print("\nFinal mutually exclusive categories")
    print("-----------------------------------")

    summary = (
        classified["category"]
        .value_counts()
        .reindex(
            CATEGORY_ORDER,
            fill_value=0,
        )
        .rename("count")
        .to_frame()
    )

    summary["proportion"] = (
        summary["count"] / total
    )

    print(
        summary
        .assign(
            proportion=lambda x:
            x["proportion"].round(4)
        )
        .to_string()
    )

    print("\nRaw Delaunay membership")
    print("-----------------------")

    for k in DELAUNAY_K_VALUES:

        count = int(
            classified[
                f"overlap_delaunay{k}"
            ].sum()
        )

        share = (
            count / total
            if total
            else 0.0
        )

        print(
            f"Delaunay-{k}:                    "
            f"{count:4d}  ({share:.4f})"
        )

    mst_count = int(
        classified["overlap_mst"].sum()
    )

    mst_share = (
        mst_count / total
        if total
        else 0.0
    )

    print(
        f"MST:                             "
        f"{mst_count:4d}  ({mst_share:.4f})"
    )


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run voltage-specific N490 Delaunay overlap analysis.
    """
    model = N490(year=2018)

    buses = model.bus.copy()
    lines = model.line.copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classified_by_voltage = {}

    all_count_tables = []
    all_proportion_tables = []

    # -------------------------------------------------------------
    # Analyze each voltage network independently.
    # -------------------------------------------------------------
    for voltage_kv in VOLTAGE_LEVELS:

        # ---------------------------------------------------------
        # 1. Load saved candidate products.
        # ---------------------------------------------------------
        (
            nodes,
            delaunay_sets,
            mst_edges,
        ) = load_voltage_delaunay_sets(
            voltage_kv=voltage_kv,
        )

        # ---------------------------------------------------------
        # 2. Recover exact original-bus -> geographic-node map.
        # ---------------------------------------------------------
        bus_to_node = build_bus_to_node_map(
            nodes
        )

        # ---------------------------------------------------------
        # 3. Extract actual N490 branches at this voltage.
        # ---------------------------------------------------------
        n490_lines = build_n490_voltage_lines(
            lines=lines,
            buses=buses,
            voltage_kv=voltage_kv,
        )

        # ---------------------------------------------------------
        # 4. Map N490 branch endpoints to the saved geographic
        #    Delaunay node IDs.
        # ---------------------------------------------------------
        n490_edges = map_n490_lines_to_nodes(
            n490_lines=n490_lines,
            bus_to_node=bus_to_node,
            voltage_kv=voltage_kv,
        )

        # ---------------------------------------------------------
        # 5. Classify Delaunay overlap, then recategorize MST.
        # ---------------------------------------------------------
        classified = (
            classify_n490_delaunay_overlap(
                n490_edges=n490_edges,
                delaunay_sets=delaunay_sets,
                mst_edges=mst_edges,
                voltage_kv=voltage_kv,
            )
        )

        classified_by_voltage[
            voltage_kv
        ] = classified

        # ---------------------------------------------------------
        # 6. Diagnostics.
        # ---------------------------------------------------------
        print_voltage_summary(
            classified=classified,
            voltage_kv=voltage_kv,
        )

        # ---------------------------------------------------------
        # 7. Summaries.
        # ---------------------------------------------------------
        (
            counts,
            proportions,
        ) = summarize_voltage_overlap(
            classified=classified,
            voltage_kv=voltage_kv,
        )

        all_count_tables.append(
            counts
        )

        all_proportion_tables.append(
            proportions
        )

        # ---------------------------------------------------------
        # 8. Save branch-level geographic classification.
        # ---------------------------------------------------------
        voltage_tag = str(
            int(voltage_kv)
        )

        classified_out = (
            classified
            .drop(columns=["node_pair"])
            .reset_index(drop=True)
            .to_crs(SOURCE_CRS)
        )
        
        classified_out.to_file(
            OUTPUT_DIR
            / (
                f"n490_{voltage_tag}kv_"
                "overlap.geojson"
            ),
            driver="GeoJSON",
            index=False,
        )

    # -------------------------------------------------------------
    # Combined long-form tables.
    # -------------------------------------------------------------
    counts_all = pd.concat(
        all_count_tables,
        ignore_index=True,
    )

    proportions_all = pd.concat(
        all_proportion_tables,
        ignore_index=True,
    )

    counts_all.to_csv(
        OUTPUT_DIR
        / "n490_delaunay_overlap_counts.csv",
        index=False,
    )

    proportions_all.to_csv(
        OUTPUT_DIR
        / "n490_delaunay_overlap_proportions.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Wide quota-style table.
    # -------------------------------------------------------------
    stats_by_voltage = build_wide_statistics(
        classified_by_voltage
    )

    print("\n")
    print("=" * 72)
    print("N490 Delaunay overlap statistics by voltage")
    print("=" * 72)

    print(
        stats_by_voltage
        .round(4)
        .to_string(index=False)
    )

    stats_by_voltage.to_csv(
        OUTPUT_DIR
        / "N490_Delaunay_stats_by_voltage.csv",
        index=False,
    )

    stats_by_voltage.to_pickle(
        OUTPUT_DIR
        / "N490_Delaunay_stats_by_voltage.pkl"
    )
    
    # ---------------------------------------------------------------------
    # Plot overlap proportions by voltage
    # ---------------------------------------------------------------------
    
    def plot_overlap_statistics(
        stats_by_voltage: pd.DataFrame,
    ) -> None:
        """
        Plot N490 overlap-category proportions for each voltage level.
    
        The plotted categories are mutually exclusive: MST, Delaunay-1
        excluding MST, and Delaunay-2 through Delaunay-5.
    
        Parameters
        ----------
        stats_by_voltage:
            Wide-form voltage-specific overlap statistics produced by
            ``build_wide_statistics()``.
    
        Returns
        -------
        None
        """
        categories = [
            "MST",
            "Delaunay1",
            "Delaunay2",
            "Delaunay3",
            "Delaunay4",
            "Delaunay5",
        ]
    
        labels = [
            "MST",
            "Delaunay",
            "Delaunay 2",
            "Delaunay 3",
            "Delaunay 4",
            "Delaunay 5",
        ]
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        x = np.arange(len(categories))
    
        markers = {
            220: "o",
            300: "s",
            380: "^",
        }
    
        for _, row in stats_by_voltage.iterrows():
    
            voltage_kv = int(row["Vbase"])
    
            percentages = [
                100.0 * float(row[category])
                for category in categories
            ]
    
            ax.plot(
                x,
                percentages,
                marker=markers.get(voltage_kv, "o"),
                markersize=9,
                linewidth=2.0,
                label=f"{voltage_kv} kV",
            )
    
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    
        ax.set_xlabel("Edge set")
        ax.set_ylabel("Percentage overlap (%)")
    
        ax.set_title(
            "N490 overlap with voltage-specific Delaunay edge sets"
        )
    
        ax.set_ylim(bottom=0)
    
        ax.grid(
            axis="y",
            alpha=0.3,
        )
    
        ax.legend(
            title="Voltage level"
        )
    
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # Combined branch-level non-geographic table.
    # -------------------------------------------------------------
    branch_tables = []

    for voltage_kv, classified in (
        classified_by_voltage.items()
    ):

        table = (
            pd.DataFrame(
                classified.drop(
                    columns="geometry"
                )
            )
            .copy()
        )

        table["Vbase_analysis"] = (
            int(voltage_kv)
        )

        branch_tables.append(
            table
        )

    branch_classifications = pd.concat(
        branch_tables,
        ignore_index=True,
    )

    branch_classifications.to_pickle(
        OUTPUT_DIR
        / "N490_Delaunay_edge_classifications.pkl"
    )
    

    print(
        "\nSaved overlap analysis to:"
    )
    print(
        f"  {OUTPUT_DIR.resolve()}"
    )
    
    print(
        stats_by_voltage
        .round(4)
        .to_string(index=False)
    )
    
    plot_overlap_statistics(
        stats_by_voltage
    )
    
    # -------------------------------------------------------------
    # Average overlap statistics across voltage levels
    # -------------------------------------------------------------
    average_stats = build_average_overlap_statistics(
        stats_by_voltage
    )
    
    print("\n")
    print("=" * 72)
    print("Average N490 Delaunay overlap across voltage levels")
    print("=" * 72)
    
    print(
        average_stats
        .round(
            {
                "average_proportion": 4,
                "average_percentage": 2,
            }
        )
        .to_string(index=False)
    )
    
    average_stats.to_pickle(
        OUTPUT_DIR
        / "N490_Delaunay_averages.pkl"
    )


if __name__ == "__main__":
    main()