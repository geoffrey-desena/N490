#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mean node degree of complete European networks by voltage class
================================================================

Purpose
-------
Use the results produced by ``n490_europe_node_degree_parameters.py`` to
summarize the mean node degree of the COMPLETE European comparison networks.

This analysis is intentionally independent of N490. It asks:

    Across the European comparison dataset, what number of branch incidences
    per node is typical when parallel circuits are retained?

For each country/voltage network, mean node degree is calculated from the
complete edge list statistics already stored in the parameter table:

    <k> = 2 E / N

where:

    E = number of branches/circuits in the complete network
    N = number of nodes represented by those branches

The country/voltage observations are then grouped into the same four voltage
classes used in the A-gamma analysis:

    <200 kV
    200-299 kV
    300-349 kV
    >=350 kV

For each voltage class the script reports:

    - number of network observations
    - mean of the network mean degrees
    - median
    - sample standard deviation (ddof=1)
    - minimum
    - first quartile
    - third quartile
    - maximum

A box-and-whisker plot is produced with every individual country/voltage
observation shown as a point.

Outputs
-------
Statistics are saved to:

    euro-comparison/node-degree-parameter-comparison/
        complete_network_mean_degree_statistics.pkl
        complete_network_mean_degree_statistics.csv

The per-network mean-degree values used to calculate those statistics are also
saved for traceability:

        complete_network_mean_degree_values.pkl
        complete_network_mean_degree_values.csv

The figure is displayed but NOT saved.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

INPUT_FILE = (
    WORKING_DIR
    / "euro-comparison"
    / "node-degree-parameter-comparison"
    / "euro_n490_node_degree_parameters.pkl"
)

OUTPUT_DIR = (
    WORKING_DIR
    / "euro-comparison"
    / "node-degree-parameter-comparison"
)

STATISTICS_PICKLE = (
    OUTPUT_DIR
    / "complete_network_mean_degree_statistics.pkl"
)

STATISTICS_CSV = (
    OUTPUT_DIR
    / "complete_network_mean_degree_statistics.csv"
)

VALUES_PICKLE = (
    OUTPUT_DIR
    / "complete_network_mean_degree_values.pkl"
)

VALUES_CSV = (
    OUTPUT_DIR
    / "complete_network_mean_degree_values.csv"
)


# =====================================================================
# SETTINGS
# =====================================================================

VOLTAGE_CLASS_ORDER = [
    "<200 kV",
    "200-299 kV",
    "300-349 kV",
    ">=350 kV",
]

FIGSIZE = (9.0, 6.5)

# Horizontal jitter only separates points visually. A fixed seed keeps the
# figure reproducible from run to run.
JITTER_WIDTH = 0.10
RANDOM_SEED = 490

BOX_WIDTH = 0.55
POINT_SIZE = 42
POINT_ALPHA = 0.72


# =====================================================================
# LOAD AND FILTER DATA
# =====================================================================

def load_complete_european_networks(input_file):
    """
    Load the saved A-gamma parameter table and retain only European complete
    networks.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            "\nCould not find the saved parameter table:\n"
            f"  {input_file}\n\n"
            "Run n490_europe_node_degree_parameters.py first."
        )

    df = pd.read_pickle(input_file)

    required_columns = {
        "source",
        "system",
        "graph_type",
        "voltage",
        "voltage_class",
        "n_nodes",
        "n_edges",
    }

    missing = sorted(required_columns.difference(df.columns))

    if missing:
        raise KeyError(
            "\nThe saved parameter table is missing required columns:\n  "
            + ", ".join(missing)
        )

    european = df.loc[
        (df["source"] == "Europe")
        & (df["graph_type"] == "complete")
    ].copy()

    if european.empty:
        raise ValueError(
            "No European complete-network rows were found in the input table."
        )

    return european


# =====================================================================
# MEAN NODE DEGREE
# =====================================================================

def calculate_network_mean_degrees(european):
    """
    Calculate the average node degree of each complete network:

        <k> = 2 E / N

    Parallel circuits are retained because n_edges comes from the complete
    graph representation.
    """

    values = european.copy()

    invalid = (
        values["n_nodes"].isna()
        | values["n_edges"].isna()
        | (values["n_nodes"] <= 0)
        | (values["n_edges"] < 0)
    )

    if invalid.any():
        print(
            "\nWARNING: Dropping rows with invalid node/edge counts:"
        )
        print(
            values.loc[
                invalid,
                [
                    "system",
                    "voltage",
                    "voltage_class",
                    "n_nodes",
                    "n_edges",
                ],
            ].to_string(index=False)
        )

        values = values.loc[~invalid].copy()

    values["mean_degree"] = (
        2.0 * values["n_edges"] / values["n_nodes"]
    )

    # Keep only columns useful for this analysis.
    keep_columns = [
        "system",
        "voltage",
        "voltage_class",
        "n_nodes",
        "n_edges",
        "mean_degree",
    ]

    # Preserve constituent_voltages when present, since it is useful for the
    # aggregated subtransmission observation.
    if "constituent_voltages" in values.columns:
        keep_columns.insert(2, "constituent_voltages")

    values = values[keep_columns].copy()

    values["voltage_class"] = pd.Categorical(
        values["voltage_class"],
        categories=VOLTAGE_CLASS_ORDER,
        ordered=True,
    )

    return values.sort_values(
        ["voltage_class", "system", "voltage"]
    ).reset_index(drop=True)


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

def build_summary_statistics(values):
    """Summarize the distribution of network-average degree by voltage class."""

    rows = []

    for voltage_class in VOLTAGE_CLASS_ORDER:
        group = values.loc[
            values["voltage_class"] == voltage_class,
            "mean_degree",
        ].dropna()

        if group.empty:
            rows.append(
                {
                    "voltage_class": voltage_class,
                    "n_networks": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "max": np.nan,
                }
            )
            continue

        rows.append(
            {
                "voltage_class": voltage_class,
                "n_networks": int(group.size),
                "mean": float(group.mean()),
                "median": float(group.median()),
                # Sample standard deviation across observed networks.
                "std": float(group.std(ddof=1)) if group.size > 1 else np.nan,
                "min": float(group.min()),
                "q1": float(group.quantile(0.25)),
                "q3": float(group.quantile(0.75)),
                "max": float(group.max()),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# CONSOLE OUTPUT
# =====================================================================

def print_network_values(values):
    """Print all individual complete-network mean-degree observations."""

    print("\n")
    print("=" * 120)
    print("COMPLETE EUROPEAN NETWORK MEAN DEGREE VALUES")
    print("=" * 120)

    columns = [
        "system",
        "voltage",
        "voltage_class",
        "n_nodes",
        "n_edges",
        "mean_degree",
    ]

    print(
        values[columns].to_string(
            index=False,
            formatters={
                "voltage": "{:.0f}".format,
                "mean_degree": "{:.4f}".format,
            },
        )
    )


def print_summary_statistics(summary):
    """Print class-level descriptive statistics."""

    print("\n")
    print("=" * 120)
    print("MEAN DEGREE STATISTICS BY VOLTAGE CLASS - COMPLETE NETWORKS")
    print("=" * 120)

    print(
        summary.to_string(
            index=False,
            formatters={
                "mean": "{:.4f}".format,
                "median": "{:.4f}".format,
                "std": "{:.4f}".format,
                "min": "{:.4f}".format,
                "q1": "{:.4f}".format,
                "q3": "{:.4f}".format,
                "max": "{:.4f}".format,
            },
        )
    )

    print("\nNotes:")
    print("  - <k> for each network is calculated as 2E/N.")
    print("  - E retains parallel circuits because only complete networks are used.")
    print("  - 'std' is the sample standard deviation across networks (ddof=1).")
    print("  - Each country/voltage network is one observation.")


# =====================================================================
# BOX-AND-WHISKER PLOT
# =====================================================================

def plot_mean_degree_distributions(values):
    """
    Box-and-whisker plot by voltage class with every network observation
    overlaid as a jittered point.
    """

    data_by_class = []

    for voltage_class in VOLTAGE_CLASS_ORDER:
        data = (
            values.loc[
                values["voltage_class"] == voltage_class,
                "mean_degree",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        data_by_class.append(data)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    positions = np.arange(1, len(VOLTAGE_CLASS_ORDER) + 1)

    ax.boxplot(
        data_by_class,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=False,
        medianprops={"linewidth": 1.8},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    # Show every actual observation. Suppressing boxplot fliers above avoids
    # plotting extreme observations twice.
    rng = np.random.default_rng(RANDOM_SEED)

    for x_position, data in zip(positions, data_by_class):
        if len(data) == 0:
            continue

        jitter = rng.uniform(
            -JITTER_WIDTH,
            JITTER_WIDTH,
            size=len(data),
        )

        ax.scatter(
            np.full(len(data), x_position, dtype=float) + jitter,
            data,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(VOLTAGE_CLASS_ORDER)

    ax.set_xlabel("Voltage class")
    ax.set_ylabel(r"Network average node degree, $\langle k \rangle$")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(values, summary):
    """Save both the descriptive statistics and their underlying values."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert categorical column back to plain strings before saving so the
    # files remain straightforward to consume in later scripts.
    values_to_save = values.copy()
    values_to_save["voltage_class"] = (
        values_to_save["voltage_class"].astype("object")
    )

    summary.to_pickle(STATISTICS_PICKLE)
    summary.to_csv(STATISTICS_CSV, index=False)

    values_to_save.to_pickle(VALUES_PICKLE)
    values_to_save.to_csv(VALUES_CSV, index=False)

    print("\n")
    print("=" * 120)
    print("SAVED RESULTS")
    print("=" * 120)
    print(f"Statistics pickle:\n  {STATISTICS_PICKLE}")
    print(f"Statistics CSV:\n  {STATISTICS_CSV}")
    print(f"Per-network values pickle:\n  {VALUES_PICKLE}")
    print(f"Per-network values CSV:\n  {VALUES_CSV}")
    print("\nFigure is displayed only and is not saved.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 120)
    print("COMPLETE-NETWORK MEAN NODE DEGREE ANALYSIS")
    print("=" * 120)
    print(f"Input:\n  {INPUT_FILE}")

    european = load_complete_european_networks(INPUT_FILE)

    print(f"\nEuropean complete-network rows loaded: {len(european)}")

    values = calculate_network_mean_degrees(european)
    summary = build_summary_statistics(values)

    print_network_values(values)
    print_summary_statistics(summary)

    save_results(values, summary)

    plot_mean_degree_distributions(values)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
