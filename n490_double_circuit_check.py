#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare parallel / duplicate line prevalence in N490 and European grids.

For N490 and each of the 15 European comparison networks:

1. Group lines by voltage level.
2. Treat connections as undirected:
       i -> j  ==  j -> i
3. Count:
       - total line rows
       - unique bus pairs
       - bus pairs represented more than once
       - redundant line rows beyond one edge per bus pair
       - line rows belonging to parallel-pair groups
4. Report duplicate-line percentages at each voltage level.

Definitions
-----------
For a voltage-specific edge set:

    duplicate_lines = total_lines - unique_bus_pairs

and

    duplicate_lines_pct =
        100 * duplicate_lines / total_lines

Thus:
    - double circuit -> 1 duplicate line
    - triple circuit -> 2 duplicate lines

A second statistic is also reported:

    lines_in_parallel_pairs_pct

This is the fraction of all line rows that belong to a bus pair
having multiplicity > 1.

Inputs
------
N490:
    loaded directly using nordic490.N490

European networks:
    euro-comparison/european_networks.pkl

Outputs
-------
Console summary

CSV:
    euro-comparison/
        duplicate_line_summary.csv

Pickle:
    euro-comparison/
        duplicate_line_summary.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nordic490 import N490


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

EURO_DIR = (
    WORKING_DIR
    / "euro-comparison"
)

EURO_FILE = (
    EURO_DIR
    / "european_networks.pkl"
)

OUTPUT_CSV = (
    EURO_DIR
    / "duplicate_line_summary.csv"
)

OUTPUT_PKL = (
    EURO_DIR
    / "duplicate_line_summary.pkl"
)


# =====================================================================
# N490 ENDPOINT DETECTION
# =====================================================================

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify bus endpoint columns in N490 model.line.
    """

    candidate_pairs = [
        ("bus0", "bus1"),
        ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"),
        ("fbus", "tbus"),
        ("from", "to"),
    ]

    for col0, col1 in candidate_pairs:

        if (
            col0 in lines.columns
            and col1 in lines.columns
        ):
            return col0, col1

    raise ValueError(
        "Could not identify N490 line endpoint columns.\n"
        f"Available columns:\n"
        f"{lines.columns.tolist()}"
    )


# =====================================================================
# DUPLICATE STATISTICS
# =====================================================================

def calculate_duplicate_statistics(
    edges: pd.DataFrame,
    node_i_col: str,
    node_j_col: str,
) -> dict:
    """
    Calculate duplicate / parallel-edge statistics for one edge set.

    Connections are treated as undirected.
    """

    if edges.empty:

        return {
            "total_lines": 0,
            "unique_pairs": 0,
            "duplicate_pairs": 0,
            "duplicate_lines": 0,
            "duplicate_lines_pct": np.nan,
            "lines_in_parallel_pairs": 0,
            "lines_in_parallel_pairs_pct": np.nan,
            "max_multiplicity": 0,
        }

    working = edges[
        [
            node_i_col,
            node_j_col,
        ]
    ].dropna().copy()

    # -------------------------------------------------------------
    # Canonical undirected bus pair
    # -------------------------------------------------------------

    working["pair_0"] = working[
        [
            node_i_col,
            node_j_col,
        ]
    ].min(
        axis=1
    )

    working["pair_1"] = working[
        [
            node_i_col,
            node_j_col,
        ]
    ].max(
        axis=1
    )

    # -------------------------------------------------------------
    # Multiplicity of each bus pair
    # -------------------------------------------------------------

    pair_counts = (
        working.groupby(
            [
                "pair_0",
                "pair_1",
            ]
        )
        .size()
        .rename("multiplicity")
        .reset_index()
    )

    total_lines = len(
        working
    )

    unique_pairs = len(
        pair_counts
    )

    parallel_pairs = pair_counts.loc[
        pair_counts[
            "multiplicity"
        ] > 1
    ]

    duplicate_pairs = len(
        parallel_pairs
    )

    # Number of redundant rows removed when converting
    # to a simple graph.
    duplicate_lines = int(
        total_lines
        - unique_pairs
    )

    duplicate_lines_pct = (
        100.0
        * duplicate_lines
        / total_lines
    )

    # Number of line rows belonging to any duplicated pair.
    lines_in_parallel_pairs = int(
        parallel_pairs[
            "multiplicity"
        ].sum()
    )

    lines_in_parallel_pairs_pct = (
        100.0
        * lines_in_parallel_pairs
        / total_lines
    )

    max_multiplicity = int(
        pair_counts[
            "multiplicity"
        ].max()
    )

    return {
        "total_lines":
            total_lines,

        "unique_pairs":
            unique_pairs,

        "duplicate_pairs":
            duplicate_pairs,

        "duplicate_lines":
            duplicate_lines,

        "duplicate_lines_pct":
            duplicate_lines_pct,

        "lines_in_parallel_pairs":
            lines_in_parallel_pairs,

        "lines_in_parallel_pairs_pct":
            lines_in_parallel_pairs_pct,

        "max_multiplicity":
            max_multiplicity,
    }


# =====================================================================
# N490 ANALYSIS
# =====================================================================

def analyze_n490():
    """
    Calculate duplicate-line statistics for every N490 voltage level.
    """

    model = N490(
        year=2018
    )

    lines = model.line.copy()

    node_i_col, node_j_col = (
        resolve_line_endpoint_columns(
            lines
        )
    )

    if "Vbase" not in lines.columns:

        raise ValueError(
            "N490 model.line does not contain Vbase."
        )

    voltage_series = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    voltage_levels = sorted(
        voltage_series
        .dropna()
        .unique()
    )

    rows = []

    for voltage in voltage_levels:

        voltage_edges = lines.loc[
            np.isclose(
                voltage_series,
                voltage,
                equal_nan=False,
            )
        ].copy()

        stats = (
            calculate_duplicate_statistics(
                edges=voltage_edges,
                node_i_col=node_i_col,
                node_j_col=node_j_col,
            )
        )

        rows.append(
            {
                "dataset": "N490",
                "country": "N490",
                "voltage_kv": float(
                    voltage
                ),
                **stats,
            }
        )

    return rows


# =====================================================================
# EUROPEAN ANALYSIS
# =====================================================================

def analyze_european_networks():
    """
    Calculate duplicate-line statistics for every country and
    every voltage level.
    """

    if not EURO_FILE.exists():

        raise FileNotFoundError(
            f"European network pickle not found:\n"
            f"{EURO_FILE}"
        )

    networks = pd.read_pickle(
        EURO_FILE
    )

    rows = []

    for country in sorted(
        networks
    ):

        edges = networks[
            country
        ].copy()

        required = {
            "node_i",
            "node_j",
            "voltage_kv",
        }

        missing = (
            required
            - set(
                edges.columns
            )
        )

        if missing:

            raise ValueError(
                f"{country} is missing columns: "
                f"{sorted(missing)}"
            )

        voltage_levels = sorted(
            pd.to_numeric(
                edges[
                    "voltage_kv"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        for voltage in voltage_levels:

            voltage_edges = edges.loc[
                pd.to_numeric(
                    edges[
                        "voltage_kv"
                    ],
                    errors="coerce",
                )
                == voltage
            ].copy()

            stats = (
                calculate_duplicate_statistics(
                    edges=voltage_edges,
                    node_i_col="node_i",
                    node_j_col="node_j",
                )
            )

            rows.append(
                {
                    "dataset":
                        "European comparison",

                    "country":
                        country,

                    "voltage_kv":
                        float(voltage),

                    **stats,
                }
            )

    return rows


# =====================================================================
# PRINT SUMMARY
# =====================================================================

def print_summary(
    summary: pd.DataFrame,
):
    """
    Print duplicate statistics grouped by network.
    """

    print("\n")
    print("=" * 125)
    print(
        "DUPLICATE / PARALLEL LINE SUMMARY BY VOLTAGE LEVEL"
    )
    print("=" * 125)

    display_columns = [
        "country",
        "voltage_kv",
        "total_lines",
        "unique_pairs",
        "duplicate_pairs",
        "duplicate_lines",
        "duplicate_lines_pct",
        "lines_in_parallel_pairs_pct",
        "max_multiplicity",
    ]

    display = summary[
        display_columns
    ].copy()

    display = display.rename(
        columns={
            "voltage_kv":
                "kV",

            "total_lines":
                "lines",

            "unique_pairs":
                "unique",

            "duplicate_pairs":
                "dup_pairs",

            "duplicate_lines":
                "dup_lines",

            "duplicate_lines_pct":
                "dup_%",

            "lines_in_parallel_pairs_pct":
                "parallel_lines_%",

            "max_multiplicity":
                "max_mult",
        }
    )

    print(
        display
        .round(
            {
                "kV": 0,
                "dup_%": 2,
                "parallel_lines_%": 2,
            }
        )
        .to_string(
            index=False
        )
    )


# =====================================================================
# COUNTRY-LEVEL COMPACT SUMMARY
# =====================================================================

def print_country_compact_summary(
    summary: pd.DataFrame,
):
    """
    Print just the requested duplicate percentage by country/voltage.
    """

    print("\n")
    print("=" * 100)
    print(
        "DUPLICATE LINES AS PERCENT OF TOTAL LINES"
    )
    print("=" * 100)

    compact = summary[
        [
            "country",
            "voltage_kv",
            "total_lines",
            "duplicate_lines",
            "duplicate_lines_pct",
        ]
    ].copy()

    compact = compact.rename(
        columns={
            "voltage_kv":
                "voltage_kV",

            "total_lines":
                "lines",

            "duplicate_lines":
                "duplicates",

            "duplicate_lines_pct":
                "duplicate_pct",
        }
    )

    print(
        compact
        .round(
            {
                "voltage_kV": 0,
                "duplicate_pct": 2,
            }
        )
        .to_string(
            index=False
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Analyze
    # -----------------------------------------------------------------

    rows = []

    rows.extend(
        analyze_n490()
    )

    rows.extend(
        analyze_european_networks()
    )

    summary = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------------------
    # Sort
    # -----------------------------------------------------------------

    summary = summary.sort_values(
        [
            "country",
            "voltage_kv",
        ]
    ).reset_index(
        drop=True
    )

    # Keep N490 at the top for easier comparison.
    n490_rows = summary.loc[
        summary["country"]
        == "N490"
    ]

    euro_rows = summary.loc[
        summary["country"]
        != "N490"
    ]

    summary = pd.concat(
        [
            n490_rows,
            euro_rows,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Print
    # -----------------------------------------------------------------

    print_summary(
        summary
    )

    print_country_compact_summary(
        summary
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    summary.to_pickle(
        OUTPUT_PKL
    )

    print("\n")
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        f"CSV:\n"
        f"  {OUTPUT_CSV}"
    )

    print(
        f"\nPickle:\n"
        f"  {OUTPUT_PKL}"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()