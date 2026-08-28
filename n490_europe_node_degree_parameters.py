#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
European + N490 node-degree exponential parameter comparison
============================================================

Purpose
-------
Fit the anchored exponential CCDF model used in
``euro_nd_parameters_plot.py`` to:

    1. the 15 European comparison networks, and
    2. the N490 network,

for BOTH:

    - complete networks (parallel circuits retained), and
    - simple graphs (parallel circuits collapsed).

For every system, all voltage levels below 200 kV are aggregated into one
subtransmission network. Every voltage level >= 200 kV is analyzed separately.
For N490 this therefore gives the four expected groups:

    <200 kV (132 kV), 220 kV, 300 kV, 380 kV.

Model
-----
For k >= 2:

    P(K >= k) = A * exp(-(k - 2) / gamma)

Outputs
-------
A tidy parameter table is saved as both pickle and CSV. Two A-gamma scatter
plots are produced: one for the complete networks and one for the simple graphs.
European points are semi-transparent; N490 points are opaque and emphasized.

Expected European input
-----------------------
    euro-comparison/european_networks.pkl

The European pickle should be a dict-like object mapping country name to a
DataFrame with columns:

    node_i, node_j, voltage_kv

N490 input
----------
The Nordic490 data are loaded directly from ``nordic490.N490(year=2018)``.
The analysis uses only ``model.line`` branches. Parallel AC lines are retained
in the complete-network analysis and collapsed in the simple-graph analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nordic490 import N490


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

EURO_INPUT_FILE = (
    WORKING_DIR
    / "euro-comparison"
    / "european_networks.pkl"
)

OUTPUT_DIR = (
    WORKING_DIR
    / "euro-comparison"
    / "node-degree-parameter-comparison"
)

PARAMETER_PICKLE = (
    OUTPUT_DIR
    / "euro_n490_node_degree_parameters.pkl"
)

PARAMETER_CSV = (
    OUTPUT_DIR
    / "euro_n490_node_degree_parameters.csv"
)

COMPLETE_FIGURE = (
    OUTPUT_DIR
    / "A_gamma_complete_networks.png"
)

SIMPLE_FIGURE = (
    OUTPUT_DIR
    / "A_gamma_simple_graphs.png"
)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200.0
MIN_FIT_DEGREE = 2

FIGSIZE = (9.0, 7.0)
EURO_ALPHA = 0.38
N490_ALPHA = 1.00
EURO_MARKER_SIZE = 70
N490_MARKER_SIZE = 190

# Numerical tolerance used when matching nominal voltage levels.
VOLTAGE_TOLERANCE_KV = 1e-6


# =====================================================================
# PLOT DEFINITIONS
# =====================================================================

VOLTAGE_CLASS_ORDER = [
    "<200 kV",
    "200-299 kV",
    "300-349 kV",
    ">=350 kV",
]

VOLTAGE_COLOR_MAP = {
    "<200 kV": "black",
    "200-299 kV": "green",
    "300-349 kV": "gold",
    ">=350 kV": "red",
}


# =====================================================================
# ANCHORED EXPONENTIAL MODEL
# =====================================================================

def anchored_exponential_degree_distribution(k, A, gamma):
    """Anchored exponential CCDF used in the previous analysis."""

    return A * np.exp(-(k - 2.0) / gamma)


# =====================================================================
# GRAPH UTILITIES
# =====================================================================

def make_simple_graph(edges):
    """Collapse parallel edges connecting the same unordered node pair."""

    if edges.empty:
        return edges.copy()

    simple_edges = edges.copy()

    # Convert endpoint labels to strings only for constructing the unordered
    # pair key. This avoids assumptions about whether bus IDs are int or str.
    node_i_key = simple_edges["node_i"].astype(str)
    node_j_key = simple_edges["node_j"].astype(str)

    simple_edges["_pair_i"] = np.where(
        node_i_key <= node_j_key,
        node_i_key,
        node_j_key,
    )
    simple_edges["_pair_j"] = np.where(
        node_i_key <= node_j_key,
        node_j_key,
        node_i_key,
    )

    simple_edges = (
        simple_edges
        .drop_duplicates(
            subset=["_pair_i", "_pair_j"],
            keep="first",
        )
        .drop(columns=["_pair_i", "_pair_j"])
        .reset_index(drop=True)
    )

    return simple_edges


def calculate_node_degrees(edges):
    """Calculate node degree from an edge list."""

    if edges.empty:
        return pd.Series(dtype=int)

    endpoints = pd.concat(
        [
            edges["node_i"],
            edges["node_j"],
        ],
        ignore_index=True,
    )

    return endpoints.value_counts().sort_index()


# =====================================================================
# CCDF + FIT
# =====================================================================

def calculate_ccdf(degrees):
    """Calculate P(K >= k) for k = 1 ... max degree."""

    if len(degrees) == 0:
        return np.array([]), np.array([])

    max_degree = int(degrees.max())
    k = np.arange(1, max_degree + 1, dtype=float)

    probability = np.array(
        [np.mean(degrees >= degree) for degree in k],
        dtype=float,
    )

    return k, probability


def fit_anchored_exponential(k, probability):
    """
    Fit

        P(K >= k) = A exp(-(k - 2) / gamma)

    using only k >= 2. Returns A, gamma, R2, RMSE, and number of fit points.
    """

    fit_mask = k >= MIN_FIT_DEGREE
    k_fit = k[fit_mask]
    probability_fit = probability[fit_mask]

    if len(k_fit) < 2:
        return np.nan, np.nan, np.nan, np.nan, len(k_fit)

    A_initial = float(probability_fit[0])
    gamma_initial = 2.0

    try:
        popt, _ = curve_fit(
            anchored_exponential_degree_distribution,
            k_fit,
            probability_fit,
            p0=[A_initial, gamma_initial],
            bounds=(
                [0.0, 1e-8],
                [1.0, np.inf],
            ),
            maxfev=100000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return np.nan, np.nan, np.nan, np.nan, len(k_fit)

    A = float(popt[0])
    gamma = float(popt[1])

    fitted = anchored_exponential_degree_distribution(
        k_fit,
        A,
        gamma,
    )

    residuals = probability_fit - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(
        np.sum(
            (probability_fit - probability_fit.mean()) ** 2
        )
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return A, gamma, r2, rmse, len(k_fit)


# =====================================================================
# VOLTAGE GROUPING
# =====================================================================

def classify_voltage(voltage):
    """Return the plotting class for a nominal voltage."""

    if voltage < 200.0:
        return "<200 kV"
    if voltage < 300.0:
        return "200-299 kV"
    if voltage < 350.0:
        return "300-349 kV"
    return ">=350 kV"


def build_voltage_groups(edges):
    """
    Build one aggregated <200 kV network and one network for every distinct
    voltage level >=200 kV.
    """

    groups = []

    valid_edges = edges.dropna(
        subset=["node_i", "node_j", "voltage_kv"]
    ).copy()

    # Remove self loops. A self loop would count twice in the endpoint-count
    # degree calculation and is not part of the intended network topology.
    valid_edges = valid_edges.loc[
        valid_edges["node_i"].astype(str)
        != valid_edges["node_j"].astype(str)
    ].copy()

    # -------------------------------------------------------------
    # Aggregated subtransmission network
    # -------------------------------------------------------------

    sub_edges = valid_edges.loc[
        valid_edges["voltage_kv"] < SUBTRANSMISSION_LIMIT_KV
    ].copy()

    if not sub_edges.empty:
        unique_sub_voltages = sorted(
            float(v)
            for v in sub_edges["voltage_kv"].dropna().unique()
        )

        groups.append(
            {
                "voltage": float(max(unique_sub_voltages)),
                "voltage_label": "<200 kV",
                "voltage_class": "<200 kV",
                "constituent_voltages": tuple(unique_sub_voltages),
                "edges": sub_edges,
            }
        )

    # -------------------------------------------------------------
    # Individual >=200 kV networks
    # -------------------------------------------------------------

    voltages = sorted(
        float(v)
        for v in valid_edges.loc[
            valid_edges["voltage_kv"] >= SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ].dropna().unique()
    )

    for voltage in voltages:
        voltage_edges = valid_edges.loc[
            np.isclose(
                valid_edges["voltage_kv"].astype(float),
                voltage,
                atol=VOLTAGE_TOLERANCE_KV,
                rtol=0.0,
            )
        ].copy()

        groups.append(
            {
                "voltage": voltage,
                "voltage_label": f"{voltage:g} kV",
                "voltage_class": classify_voltage(voltage),
                "constituent_voltages": (voltage,),
                "edges": voltage_edges,
            }
        )

    return groups


# =====================================================================
# INPUT NORMALIZATION
# =====================================================================

def resolve_n490_endpoint_columns(lines):
    """Identify the two bus-endpoint columns in ``model.line``."""

    candidate_pairs = [
        ("bus0", "bus1"),
        ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"),
        ("fbus", "tbus"),
        ("from", "to"),
    ]

    for node_i_column, node_j_column in candidate_pairs:
        if (
            node_i_column in lines.columns
            and node_j_column in lines.columns
        ):
            return node_i_column, node_j_column

    raise ValueError(
        "Could not identify N490 line endpoint columns.\n"
        f"Available columns:\n{lines.columns.tolist()}"
    )


def normalize_n490_edges(lines):
    """Convert ``N490.model.line`` to node_i/node_j/voltage_kv."""

    if "Vbase" not in lines.columns:
        raise ValueError(
            "N490 model.line does not contain the expected 'Vbase' column."
        )

    node_i_column, node_j_column = resolve_n490_endpoint_columns(lines)

    normalized = lines[
        [node_i_column, node_j_column, "Vbase"]
    ].copy()

    normalized.columns = [
        "node_i",
        "node_j",
        "voltage_kv",
    ]

    normalized["voltage_kv"] = pd.to_numeric(
        normalized["voltage_kv"],
        errors="coerce",
    )

    normalized = normalized.dropna(
        subset=["node_i", "node_j", "voltage_kv"]
    ).reset_index(drop=True)

    return normalized


def normalize_european_edges(edges, country):
    """Validate and normalize one European edge table."""

    required = {"node_i", "node_j", "voltage_kv"}
    missing = required - set(edges.columns)

    if missing:
        raise KeyError(
            f"European data for {country} are missing columns: "
            f"{sorted(missing)}"
        )

    normalized = edges[
        ["node_i", "node_j", "voltage_kv"]
    ].copy()

    normalized["voltage_kv"] = pd.to_numeric(
        normalized["voltage_kv"],
        errors="coerce",
    )

    normalized = normalized.dropna(
        subset=["node_i", "node_j", "voltage_kv"]
    ).reset_index(drop=True)

    return normalized


# =====================================================================
# PARAMETER TABLE
# =====================================================================

def analyze_one_system(system_name, source, edges):
    """Fit all voltage groups for complete and simple graph forms."""

    rows = []

    groups = build_voltage_groups(edges)

    for group in groups:
        complete_edges = group["edges"].copy()

        graph_versions = {
            "complete": complete_edges,
            "simple": make_simple_graph(complete_edges),
        }

        for graph_type, graph_edges in graph_versions.items():
            degrees = calculate_node_degrees(graph_edges)
            k, probability = calculate_ccdf(degrees)

            A, gamma, r2, rmse, n_fit_points = fit_anchored_exponential(
                k,
                probability,
            )

            rows.append(
                {
                    "source": source,
                    "system": system_name,
                    "graph_type": graph_type,
                    "voltage": group["voltage"],
                    "voltage_label": group["voltage_label"],
                    "voltage_class": group["voltage_class"],
                    "constituent_voltages": group["constituent_voltages"],
                    "n_nodes": int(len(degrees)),
                    "n_edges": int(len(graph_edges)),
                    "max_degree": (
                        int(degrees.max())
                        if len(degrees) > 0
                        else 0
                    ),
                    "n_fit_points": int(n_fit_points),
                    "A": A,
                    "gamma": gamma,
                    "r2": r2,
                    "rmse": rmse,
                }
            )

    return rows


def build_parameter_table(euro_networks, n490_edges):
    """Analyze Europe and N490 using one identical fitting pipeline."""

    rows = []

    for country in sorted(euro_networks):
        country_edges = normalize_european_edges(
            euro_networks[country],
            country,
        )

        rows.extend(
            analyze_one_system(
                system_name=country,
                source="Europe",
                edges=country_edges,
            )
        )

    rows.extend(
        analyze_one_system(
            system_name="N490",
            source="N490",
            edges=n490_edges,
        )
    )

    parameter_df = pd.DataFrame(rows)

    parameter_df = parameter_df.sort_values(
        by=[
            "graph_type",
            "source",
            "system",
            "voltage",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return parameter_df


# =====================================================================
# REPORTING
# =====================================================================

def print_n490_summary(parameter_df):
    """Print the N490 fit values prominently for inspection."""

    n490 = parameter_df.loc[
        parameter_df["source"] == "N490"
    ].copy()

    print("\n")
    print("=" * 112)
    print("N490 NODE-DEGREE EXPONENTIAL PARAMETERS")
    print("=" * 112)

    if n490.empty:
        print("No N490 results were produced.")
        return

    print(
        n490[
            [
                "graph_type",
                "voltage_label",
                "n_nodes",
                "n_edges",
                "A",
                "gamma",
                "r2",
                "rmse",
            ]
        ].to_string(
            index=False,
            formatters={
                "A": "{:.4f}".format,
                "gamma": "{:.4f}".format,
                "r2": "{:.4f}".format,
                "rmse": "{:.4f}".format,
            },
        )
    )


def print_dataset_summary(parameter_df):
    """Print counts of fitted networks by source/graph type."""

    summary = (
        parameter_df
        .groupby(["graph_type", "source"], dropna=False)
        .agg(
            networks=("system", "size"),
            valid_fits=("A", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )

    print("\n")
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))


# =====================================================================
# PLOTTING
# =====================================================================

def plot_parameter_space(parameter_df, graph_type, output_file):
    """Plot European and N490 A-gamma values for one graph representation."""

    valid = parameter_df.loc[
        (parameter_df["graph_type"] == graph_type)
        & parameter_df["A"].notna()
        & parameter_df["gamma"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # -------------------------------------------------------------
    # European comparison networks
    # -------------------------------------------------------------

    europe = valid.loc[valid["source"] == "Europe"]

    for voltage_class in VOLTAGE_CLASS_ORDER:
        group = europe.loc[
            europe["voltage_class"] == voltage_class
        ]

        if group.empty:
            continue

        ax.scatter(
            group["A"],
            group["gamma"],
            s=EURO_MARKER_SIZE,
            marker="o",
            color=VOLTAGE_COLOR_MAP[voltage_class],
            edgecolor="black",
            linewidth=0.45,
            alpha=EURO_ALPHA,
            label=voltage_class,
            zorder=2,
        )

    # -------------------------------------------------------------
    # N490: opaque, larger, same voltage-class colors
    # -------------------------------------------------------------

    n490 = valid.loc[valid["source"] == "N490"]

    for voltage_class in VOLTAGE_CLASS_ORDER:
        group = n490.loc[
            n490["voltage_class"] == voltage_class
        ]

        if group.empty:
            continue

        ax.scatter(
            group["A"],
            group["gamma"],
            s=N490_MARKER_SIZE,
            marker="*",
            color=VOLTAGE_COLOR_MAP[voltage_class],
            edgecolor="black",
            linewidth=1.0,
            alpha=N490_ALPHA,
            zorder=5,
        )

    # N490 text labels make the four Nordic reference points explicit while
    # leaving the European country markers unlabeled.
    for _, row in n490.iterrows():
        if row["voltage_class"] == "<200 kV":
            annotation_label = f"{row['voltage']:g} kV"
        else:
            annotation_label = row["voltage_label"]

        ax.annotate(
            annotation_label,
            xy=(row["A"], row["gamma"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            zorder=6,
        )

    # -------------------------------------------------------------
    # Legends: voltage color + N490 marker semantics
    # -------------------------------------------------------------

    voltage_handles = []

    for voltage_class in VOLTAGE_CLASS_ORDER:
        if not valid.loc[
            valid["voltage_class"] == voltage_class
        ].empty:
            voltage_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=VOLTAGE_COLOR_MAP[voltage_class],
                    markeredgecolor="black",
                    markersize=8,
                    label=voltage_class,
                )
            )

    source_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            alpha=EURO_ALPHA,
            markersize=8,
            label="European networks",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=13,
            label="N490",
        ),
    ]

    legend_voltage = ax.legend(
        handles=voltage_handles,
        title="Voltage group",
        frameon=False,
        loc="best",
    )
    ax.add_artist(legend_voltage)

    ax.legend(
        handles=source_handles,
        title="Dataset",
        frameon=False,
        loc="upper right",
    )

    ax.set_xlabel(r"$A$", fontsize=13)
    ax.set_ylabel(r"$\gamma$", fontsize=13)

    graph_title = (
        "Complete networks"
        if graph_type == "complete"
        else "Simple graphs"
    )

    ax.set_title(
        f"Anchored exponential node-degree parameters — {graph_title}"
    )

    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("=" * 100)
    print("LOADING EUROPEAN NETWORK DATA")
    print("=" * 100)
    print(f"Input:\n  {EURO_INPUT_FILE}")

    if not EURO_INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nCould not find European network pickle:\n"
            f"  {EURO_INPUT_FILE}"
        )

    euro_networks = pd.read_pickle(EURO_INPUT_FILE)

    if not isinstance(euro_networks, dict):
        raise TypeError(
            "European input must be a dict mapping country name to DataFrame."
        )

    print(f"Countries loaded: {len(euro_networks)}")

    print("\n")
    print("=" * 100)
    print("LOADING N490 LINE DATA")
    print("=" * 100)

    n490_model = N490(year=2018)
    n490_edges = normalize_n490_edges(
        n490_model.line.copy()
    )

    print(f"N490 line rows loaded: {len(n490_edges)}")
    print(
        "N490 voltages found: "
        + ", ".join(
            f"{v:g} kV"
            for v in sorted(n490_edges["voltage_kv"].unique())
        )
    )

    n490_expected_groups = [132.0, 220.0, 300.0, 380.0]
    n490_found = set(
        float(v)
        for v in n490_edges["voltage_kv"].unique()
    )

    missing_expected = [
        voltage
        for voltage in n490_expected_groups
        if not any(
            np.isclose(voltage, found)
            for found in n490_found
        )
    ]

    if missing_expected:
        print(
            "WARNING: expected N490 voltage levels not present in "
            "model.line: "
            + ", ".join(f"{v:g} kV" for v in missing_expected)
        )

    parameter_df = build_parameter_table(
        euro_networks,
        n490_edges,
    )

    print_dataset_summary(parameter_df)
    print_n490_summary(parameter_df)

    # -------------------------------------------------------------
    # Save reusable results
    # -------------------------------------------------------------

    parameter_df.to_pickle(PARAMETER_PICKLE)
    parameter_df.to_csv(PARAMETER_CSV, index=False)

    print("\n")
    print("=" * 100)
    print("SAVED PARAMETER TABLE")
    print("=" * 100)
    print(f"Pickle:\n  {PARAMETER_PICKLE}")
    print(f"CSV:\n  {PARAMETER_CSV}")

    # -------------------------------------------------------------
    # A-gamma plots
    # -------------------------------------------------------------

    plot_parameter_space(
        parameter_df,
        graph_type="complete",
        output_file=COMPLETE_FIGURE,
    )

    plot_parameter_space(
        parameter_df,
        graph_type="simple",
        output_file=SIMPLE_FIGURE,
    )

    print("\n")
    print("=" * 100)
    print("SAVED FIGURES")
    print("=" * 100)
    print(f"Complete network:\n  {COMPLETE_FIGURE}")
    print(f"Simple graph:\n  {SIMPLE_FIGURE}")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
