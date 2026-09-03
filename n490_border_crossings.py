# -*- coding: utf-8 -*-
"""
N490 international bidding-zone crossing statistics.

This script counts international AC-line connections between N490 bidding zones
using the SIMPLE-GRAPH representation of the complete N490 line network.

The analysis is voltage agnostic:
all rows in ``model.line`` are considered together, regardless of ``Vbase``.

Parallel circuits are removed by collapsing multiple line rows connecting the
same unordered pair of buses into a single simple-graph edge.

Only international crossings are counted:
a line is retained only when the ``country`` values of its terminal buses differ.

The final summary contains every possible pair of N490 bidding zones belonging
to different countries. International zone pairs with no N490 connection are
therefore retained with:

    n_connections = 0

Outputs
-------
N490_international_zone_crossings.csv
N490_international_zone_crossings.pkl
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from nordic490 import N490


# =====================================================================
# CONFIGURATION
# =====================================================================

N490_OUTPUT_DIR = Path(
    "/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490"
)


# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

def ensure_output_dir(
    output_dir: Path = N490_OUTPUT_DIR,
) -> Path:
    """Create the output directory if needed and return it."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =====================================================================
# N490 LOADING
# =====================================================================

def load_n490_model(
    year: int = 2018,
) -> N490:
    """
    Load the Nordic490 model.
    """

    return N490(
        year=year
    )


# =====================================================================
# LINE ENDPOINT DETECTION
# =====================================================================

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two bus-endpoint columns in ``model.line``.
    """

    candidate_pairs = [
        ("bus0", "bus1"),
        ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"),
        ("bus1", "bus2"),
        ("fbus", "tbus"),
        ("from", "to"),
    ]

    for bus0_col, bus1_col in candidate_pairs:

        if (
            bus0_col in lines.columns
            and bus1_col in lines.columns
        ):

            return (
                bus0_col,
                bus1_col,
            )

    raise ValueError(
        "Could not identify line endpoint columns.\n"
        f"Available columns:\n"
        f"{lines.columns.tolist()}"
    )


# =====================================================================
# SIMPLE GRAPH
# =====================================================================

def make_n490_simple_graph(
    lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse N490 AC lines to a simple graph.

    Multiple line rows connecting the same unordered pair of buses are
    represented by one edge.

    Voltage is deliberately ignored.
    """

    bus0_col, bus1_col = resolve_line_endpoint_columns(lines)

    edges = (
        lines[
            [
                bus0_col,
                bus1_col,
            ]
        ]
        .dropna()
        .copy()
    )

    # N490 line endpoint IDs may be stored as floats even though
    # they represent integer bus IDs.
    edges[bus0_col] = (
        pd.to_numeric(edges[bus0_col], errors="raise")
        .astype(int)
    )

    edges[bus1_col] = (
        pd.to_numeric(edges[bus1_col], errors="raise")
        .astype(int)
    )

    # Canonical unordered bus pair.
    endpoint_array = np.sort(
        edges[
            [
                bus0_col,
                bus1_col,
            ]
        ].to_numpy(),
        axis=1,
    )

    edges["bus_i"] = endpoint_array[:, 0]
    edges["bus_j"] = endpoint_array[:, 1]

    # Remove self-loops.
    edges = edges.loc[
        edges["bus_i"] != edges["bus_j"]
    ].copy()

    # Remove parallel circuits.
    simple_edges = (
        edges[
            [
                "bus_i",
                "bus_j",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return simple_edges


# =====================================================================
# BUS METADATA
# =====================================================================

def build_bus_lookup(
    bus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a lookup table from N490 bus ID to bidding zone and country.

    N490 conventions:
        bidz    = bidding zone
        country = country code

    The dataframe index is the N490 bus identifier.
    """

    required_columns = [
        "bidz",
        "country",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in bus.columns
    ]

    if missing_columns:
        raise ValueError(
            "model.bus is missing required columns: "
            f"{missing_columns}\n"
            f"Available columns:\n"
            f"{bus.columns.tolist()}"
        )

    lookup = (
        bus[
            [
                "bidz",
                "country",
            ]
        ]
        .copy()
    )

    # Preserve N490 bus IDs as integers.
    lookup["bus_id"] = (
        pd.to_numeric(
            lookup.index,
            errors="raise",
        )
        .astype(int)
    )

    lookup["bidz"] = (
        lookup["bidz"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    lookup["country"] = (
        lookup["country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return (
        lookup[
            [
                "bus_id",
                "bidz",
                "country",
            ]
        ]
        .reset_index(drop=True)
    )


# =====================================================================
# ATTACH BUS INFORMATION TO EDGES
# =====================================================================

def attach_bus_metadata(
    edges: pd.DataFrame,
    bus_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach bidding-zone and country information to both edge endpoints.
    """

    edge_table = edges.copy()

    # -------------------------------------------------------------
    # Endpoint i
    # -------------------------------------------------------------

    endpoint_i = (
        bus_lookup.rename(
            columns={
                "bus_id": "bus_i",
                "bidz": "bidz_i",
                "country": "country_i",
            }
        )
    )

    edge_table = edge_table.merge(
        endpoint_i,
        on="bus_i",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------
    # Endpoint j
    # -------------------------------------------------------------

    endpoint_j = (
        bus_lookup.rename(
            columns={
                "bus_id": "bus_j",
                "bidz": "bidz_j",
                "country": "country_j",
            }
        )
    )

    edge_table = edge_table.merge(
        endpoint_j,
        on="bus_j",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------
    # Verify that all edge endpoints matched model.bus.
    # -------------------------------------------------------------

    metadata_columns = [
        "bidz_i",
        "country_i",
        "bidz_j",
        "country_j",
    ]

    missing_metadata = edge_table.loc[
        edge_table[
            metadata_columns
        ]
        .isna()
        .any(
            axis=1
        )
    ]

    if not missing_metadata.empty:

        raise ValueError(
            "Some line endpoints could not be matched to model.bus:\n"
            f"{missing_metadata.to_string(index=False)}"
        )

    return edge_table


# =====================================================================
# OBSERVED INTERNATIONAL CROSSINGS
# =====================================================================

def count_international_crossings(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count simple-graph connections between international bidding-zone pairs.

    A simple edge is international when:

        country_i != country_j

    Zone-pair direction is ignored.
    """

    international = edges.loc[
        edges["country_i"]
        != edges["country_j"]
    ].copy()

    # -------------------------------------------------------------
    # Convert each bidding-zone pair to a stable unordered pair.
    # -------------------------------------------------------------

    zone_pairs = np.sort(
        international[
            [
                "bidz_i",
                "bidz_j",
            ]
        ]
        .astype(str)
        .to_numpy(),
        axis=1,
    )

    international["bidz_1"] = (
        zone_pairs[:, 0]
    )

    international["bidz_2"] = (
        zone_pairs[:, 1]
    )

    crossing_counts = (
        international
        .groupby(
            [
                "bidz_1",
                "bidz_2",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "n_connections",
            }
        )
    )

    return crossing_counts


# =====================================================================
# ALL POSSIBLE INTERNATIONAL ZONE PAIRS
# =====================================================================

def build_all_international_zone_pairs(
    bus_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct all possible pairs of bidding zones belonging to different
    countries.

    These pairs form the complete reference set used in the final output,
    allowing unconnected international pairs to be represented explicitly
    with zero connections.
    """

    zone_table = (
        bus_lookup[
            [
                "bidz",
                "country",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "country",
                "bidz",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Check that every bidding zone maps to one country only.
    # -------------------------------------------------------------

    countries_per_zone = (
        zone_table
        .groupby(
            "bidz"
        )["country"]
        .nunique()
    )

    ambiguous_zones = (
        countries_per_zone.loc[
            countries_per_zone > 1
        ]
    )

    if not ambiguous_zones.empty:

        raise ValueError(
            "Some bidding zones map to more than one country:\n"
            f"{ambiguous_zones}"
        )

    zone_to_country = dict(
        zip(
            zone_table["bidz"],
            zone_table["country"],
        )
    )

    rows = []

    for bidz_a, bidz_b in combinations(
        sorted(
            zone_to_country
        ),
        2,
    ):

        country_a = (
            zone_to_country[
                bidz_a
            ]
        )

        country_b = (
            zone_to_country[
                bidz_b
            ]
        )

        # ---------------------------------------------------------
        # Only international combinations.
        # ---------------------------------------------------------

        if country_a == country_b:
            continue

        rows.append(
            {
                "bidz_1": bidz_a,
                "bidz_2": bidz_b,
                "country_1": country_a,
                "country_2": country_b,
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# FINAL SUMMARY
# =====================================================================

def build_international_crossing_summary(
    bus_lookup: pd.DataFrame,
    crossing_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine observed crossing counts with the complete set of possible
    international bidding-zone pairs.
    """

    all_pairs = (
        build_all_international_zone_pairs(
            bus_lookup
        )
    )

    summary = all_pairs.merge(
        crossing_counts,
        on=[
            "bidz_1",
            "bidz_2",
        ],
        how="left",
        validate="one_to_one",
    )

    summary["n_connections"] = (
        summary["n_connections"]
        .fillna(0)
        .astype(int)
    )

    summary["bidz_pair"] = (
        summary["bidz_1"]
        + "-"
        + summary["bidz_2"]
    )

    # Final output guard: domestic bidding-zone pairs must never be
    # included in the saved border-crossing quota table.  This is
    # intentionally applied here as well as when ``all_pairs`` is built,
    # so the output contract remains explicit if the upstream pair-building
    # logic is changed later.
    summary = summary.loc[
        summary["country_1"] != summary["country_2"]
    ].copy()

    if (
        summary["country_1"] == summary["country_2"]
    ).any():
        raise AssertionError(
            "Domestic bidding-zone pairs reached the international "
            "crossing summary."
        )

    summary = (
        summary[
            [
                "bidz_pair",
                "bidz_1",
                "bidz_2",
                "country_1",
                "country_2",
                "n_connections",
            ]
        ]
        .sort_values(
            [
                "country_1",
                "country_2",
                "bidz_1",
                "bidz_2",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# =====================================================================
# ANALYSIS
# =====================================================================

def analyze_international_zone_crossings(
    model: N490,
    output_dir: Path = N490_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Calculate and save N490 international bidding-zone crossing counts.
    """

    output_dir = ensure_output_dir(
        output_dir
    )

    bus = (
        model.bus.copy()
    )

    lines = (
        model.line.copy()
    )

    # -------------------------------------------------------------
    # Build simple graph.
    # -------------------------------------------------------------

    simple_edges = (
        make_n490_simple_graph(
            lines
        )
    )

    # -------------------------------------------------------------
    # Attach bidding-zone and country metadata.
    # -------------------------------------------------------------

    bus_lookup = (
        build_bus_lookup(
            bus
        )
    )

    edges_with_metadata = (
        attach_bus_metadata(
            edges=simple_edges,
            bus_lookup=bus_lookup,
        )
    )

    # -------------------------------------------------------------
    # Count observed international crossings.
    # -------------------------------------------------------------

    crossing_counts = (
        count_international_crossings(
            edges_with_metadata
        )
    )

    # -------------------------------------------------------------
    # Add zero-count international zone pairs.
    # -------------------------------------------------------------

    summary = (
        build_international_crossing_summary(
            bus_lookup=bus_lookup,
            crossing_counts=crossing_counts,
        )
    )

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    summary.to_csv(
        output_dir
        / "N490_international_zone_crossings.csv",
        index=False,
    )

    summary.to_pickle(
        output_dir
        / "N490_international_zone_crossings.pkl"
    )

    return summary


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """
    Run the international bidding-zone crossing analysis.
    """

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    output_dir = ensure_output_dir(
        N490_OUTPUT_DIR
    )

    print(
        "Loading N490 model..."
    )

    model = load_n490_model(
        year=2018
    )

    print(
        "\nCalculating international "
        "bidding-zone crossings..."
    )

    summary = (
        analyze_international_zone_crossings(
            model=model,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    n_connected_pairs = int(
        (
            summary["n_connections"]
            > 0
        )
        .sum()
    )

    n_zero_pairs = int(
        (
            summary["n_connections"]
            == 0
        )
        .sum()
    )

    n_international_edges = int(
        summary["n_connections"]
        .sum()
    )

    print("\n")
    print("=" * 90)
    print(
        "N490 INTERNATIONAL BIDDING-ZONE CROSSINGS"
    )
    print("=" * 90)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 90)
    print(
        "SUMMARY"
    )
    print("=" * 90)

    print(
        f"International simple-graph edges:    "
        f"{n_international_edges}"
    )

    print(
        f"Connected international zone pairs: "
        f"{n_connected_pairs}"
    )

    print(
        f"Unconnected international pairs:    "
        f"{n_zero_pairs}"
    )

    # -------------------------------------------------------------
    # Print zero-count pairs separately.
    # -------------------------------------------------------------

    zero_pairs = summary.loc[
        summary["n_connections"]
        == 0
    ].copy()

    print("\n")
    print("=" * 90)
    print(
        "INTERNATIONAL ZONE PAIRS WITH ZERO CONNECTIONS"
    )
    print("=" * 90)

    print(
        zero_pairs.to_string(
            index=False
        )
    )

    print("\nSaved outputs to:")
    print(
        output_dir
    )


if __name__ == "__main__":
    main()
