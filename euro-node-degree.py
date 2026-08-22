#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
European network node-degree analysis
=====================================

Loads the pickled European transmission/sub-transmission network data
and calculates complementary cumulative node-degree distributions for:

    1. Each complete country network
    2. One combined sub-transmission network containing all
       voltage levels below 200 kV
    3. Each voltage level >= 200 kV individually

Each network is analyzed in TWO representations:

    complete
        All line rows are retained. Parallel circuits contribute
        separately to node degree.

    simple
        Multiple edges connecting the same unordered pair of nodes
        are collapsed to a single edge before node degrees are
        calculated.

For every representation, fit:

    P(K >= k) = C * exp(-k / gamma)

with both C and gamma as free parameters.

Outputs
-------
Console:
    Country-by-country summary containing:
        graph representation
        voltage group
        number of nodes
        number of branches
        branches removed by simplification
        mean degree
        C
        gamma
        R^2

Plots:
    euro-comparison/
        node-degree-analysis/
            complete/
                Albania/
                ...
            simple/
                Albania/
                ...

Tables:
    node_degree_fit_summary.csv
    node_degree_fit_summary.pkl

    fitted_parameter_table.csv
    fitted_parameter_table.pkl

The fitted parameter table contains separate C and gamma values for
both complete and simple graph representations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

DATA_DIR = (
    WORKING_DIR
    / "euro-comparison"
)

INPUT_FILE = (
    DATA_DIR
    / "european_networks.pkl"
)

OUTPUT_DIR = (
    DATA_DIR
    / "node-degree-analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200

GRAPH_TYPES = [
    "complete",
    "simple",
]

FIGSIZE = (7.5, 5.5)
DPI = 300


# =====================================================================
# EXPONENTIAL MODEL
# =====================================================================

def exponential_degree_distribution(
    k,
    C,
    gamma,
):
    """
    Complementary cumulative exponential model:

        P(K >= k) = C * exp(-k / gamma)

    Both C and gamma are fitted freely.
    """

    return (
        C
        * np.exp(
            -k / gamma
        )
    )


# =====================================================================
# SIMPLE-GRAPH CONVERSION
# =====================================================================

def make_simple_graph(
    edges,
):
    """
    Collapse parallel edges connecting the same unordered node pair.

    Parameters
    ----------
    edges : pandas.DataFrame
        Must contain:
            node_i
            node_j

    Returns
    -------
    pandas.DataFrame
        One row per unique unordered node pair.

    Notes
    -----
    Direction is ignored.

    Thus:

        node_i = 4, node_j = 12

    is considered identical to:

        node_i = 12, node_j = 4

    Simplification is performed on the edge set supplied to this
    function. Therefore, if a combined voltage group is supplied,
    duplicate node pairs are collapsed across the complete group.
    """

    if edges.empty:
        return edges.copy()

    simple_edges = edges.copy()

    # -------------------------------------------------------------
    # Canonical unordered node pair
    # -------------------------------------------------------------

    simple_edges["_pair_i"] = (
        simple_edges[
            [
                "node_i",
                "node_j",
            ]
        ]
        .min(
            axis=1
        )
    )

    simple_edges["_pair_j"] = (
        simple_edges[
            [
                "node_i",
                "node_j",
            ]
        ]
        .max(
            axis=1
        )
    )

    # -------------------------------------------------------------
    # Keep one edge per unordered pair
    # -------------------------------------------------------------

    simple_edges = (
        simple_edges
        .drop_duplicates(
            subset=[
                "_pair_i",
                "_pair_j",
            ],
            keep="first",
        )
        .copy()
    )

    # Use canonical endpoints.
    simple_edges["node_i"] = (
        simple_edges["_pair_i"]
    )

    simple_edges["node_j"] = (
        simple_edges["_pair_j"]
    )

    simple_edges = (
        simple_edges
        .drop(
            columns=[
                "_pair_i",
                "_pair_j",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return simple_edges


def prepare_graph_representation(
    edges,
    graph_type,
):
    """
    Return either the complete or simple representation.
    """

    if graph_type == "complete":
        return edges.copy()

    if graph_type == "simple":
        return make_simple_graph(
            edges
        )

    raise ValueError(
        f"Unknown graph type: {graph_type}"
    )


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_node_degrees(
    edges,
):
    """
    Calculate node degrees directly from an edge list.

    The edge list should already represent either the complete or
    simple graph, as desired.
    """

    endpoints = pd.concat(
        [
            edges["node_i"],
            edges["node_j"],
        ],
        ignore_index=True,
    )

    return (
        endpoints
        .value_counts()
        .sort_index()
    )


# =====================================================================
# COMPLEMENTARY CUMULATIVE DEGREE DISTRIBUTION
# =====================================================================

def calculate_degree_distribution(
    degrees,
):
    """
    Calculate the complementary cumulative degree distribution:

        P(K >= k)

    Therefore:

        P(K >= 1) = 1
    """

    if len(degrees) == 0:

        return (
            np.array([]),
            np.array([]),
        )

    max_degree = int(
        degrees.max()
    )

    k = np.arange(
        1,
        max_degree + 1,
        dtype=float,
    )

    probability = np.array(
        [
            np.mean(
                degrees >= degree
            )
            for degree in k
        ],
        dtype=float,
    )

    return (
        k,
        probability,
    )


# =====================================================================
# EXPONENTIAL FIT
# =====================================================================

def fit_exponential_distribution(
    k,
    probability,
):
    """
    Fit:

        P(K >= k) = C * exp(-k / gamma)

    with both C and gamma free.
    """

    if len(k) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.full_like(
                k,
                np.nan,
            ),
        )

    C_initial = 1.5
    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            exponential_degree_distribution,
            k,
            probability,
            p0=[
                C_initial,
                gamma_initial,
            ],
            bounds=(
                [
                    0.0,
                    1e-8,
                ],
                [
                    np.inf,
                    np.inf,
                ],
            ),
            maxfev=100000,
        )

        C = float(
            popt[0]
        )

        gamma = float(
            popt[1]
        )

    except (
        RuntimeError,
        ValueError,
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.full_like(
                k,
                np.nan,
            ),
        )

    fitted_probability = (
        exponential_degree_distribution(
            k,
            C,
            gamma,
        )
    )

    # -------------------------------------------------------------
    # R^2
    # -------------------------------------------------------------

    residual_sum_squares = (
        np.sum(
            (
                probability
                - fitted_probability
            ) ** 2
        )
    )

    total_sum_squares = (
        np.sum(
            (
                probability
                - np.mean(
                    probability
                )
            ) ** 2
        )
    )

    if total_sum_squares > 0:

        r2 = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )

    else:

        r2 = np.nan

    return (
        C,
        gamma,
        r2,
        fitted_probability,
    )


# =====================================================================
# ANALYZE ONE EDGE SET
# =====================================================================

def analyze_network(
    original_edges,
    graph_type,
):
    """
    Analyze either the complete or simple representation of one
    edge set.
    """

    edges = (
        prepare_graph_representation(
            original_edges,
            graph_type,
        )
    )

    degrees = (
        calculate_node_degrees(
            edges
        )
    )

    (
        k,
        probability,
    ) = (
        calculate_degree_distribution(
            degrees
        )
    )

    (
        C,
        gamma,
        r2,
        fitted_probability,
    ) = (
        fit_exponential_distribution(
            k,
            probability,
        )
    )

    n_nodes = len(
        degrees
    )

    n_branches_original = len(
        original_edges
    )

    n_branches = len(
        edges
    )

    branches_removed = (
        n_branches_original
        - n_branches
    )

    if n_branches_original > 0:

        fraction_removed = (
            branches_removed
            / n_branches_original
        )

    else:

        fraction_removed = np.nan

    if n_nodes > 0:

        mean_degree = float(
            degrees.mean()
        )

    else:

        mean_degree = np.nan

    return {
        "degrees":
            degrees,

        "k":
            k,

        "probability":
            probability,

        "fitted_probability":
            fitted_probability,

        "n_nodes":
            n_nodes,

        "n_branches_original":
            n_branches_original,

        "n_branches":
            n_branches,

        "branches_removed":
            branches_removed,

        "fraction_removed":
            fraction_removed,

        "mean_degree":
            mean_degree,

        "C":
            C,

        "gamma":
            gamma,

        "r2":
            r2,
    }


# =====================================================================
# VOLTAGE GROUPING
# =====================================================================

def get_subtransmission_label(
    edges,
):
    """
    Build display label for all voltage levels below 200 kV.

    Examples
    --------
    110 kV

    132–165 kV
    """

    voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            < SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .unique()
        .tolist()
    )

    if len(voltages) == 0:
        return None

    if len(voltages) == 1:

        return (
            f"{voltages[0]} kV"
        )

    return (
        f"{voltages[0]}"
        f"\N{EN DASH}"
        f"{voltages[-1]} kV"
    )


def build_voltage_groups(
    edges,
):
    """
    Build edge sets analyzed for one country.

    Groups:
        All
        combined <200 kV
        each >=200 kV individually
    """

    groups = []

    # -------------------------------------------------------------
    # Complete country network
    # -------------------------------------------------------------

    groups.append(
        {
            "key":
                "All",

            "label":
                "All voltage levels",

            "edges":
                edges.copy(),
        }
    )

    # -------------------------------------------------------------
    # Combined sub-transmission network
    # -------------------------------------------------------------

    sub_edges = edges.loc[
        edges["voltage_kv"]
        < SUBTRANSMISSION_LIMIT_KV
    ].copy()

    if len(sub_edges) > 0:

        sub_label = (
            get_subtransmission_label(
                edges
            )
        )

        groups.append(
            {
                "key":
                    sub_label,

                "label":
                    sub_label,

                "edges":
                    sub_edges,
            }
        )

    # -------------------------------------------------------------
    # Individual >=200 kV networks
    # -------------------------------------------------------------

    transmission_voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            >= SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .unique()
        .tolist()
    )

    for voltage in (
        transmission_voltages
    ):

        voltage_edges = edges.loc[
            edges["voltage_kv"]
            == voltage
        ].copy()

        label = (
            f"{voltage} kV"
        )

        groups.append(
            {
                "key":
                    label,

                "label":
                    label,

                "edges":
                    voltage_edges,
            }
        )

    return groups


# =====================================================================
# SAFE FILENAMES
# =====================================================================

def safe_filename(
    text,
):
    """
    Convert display label to filename component.
    """

    return (
        str(text)
        .replace(
            " ",
            "_",
        )
        .replace(
            "\N{EN DASH}",
            "-",
        )
        .replace(
            "/",
            "-",
        )
    )


# =====================================================================
# PLOT
# =====================================================================

def plot_degree_distribution(
    result,
    country,
    voltage_label,
    graph_type,
    output_path,
):
    """
    Plot cumulative degree distribution and fitted exponential curve.
    """

    k = result[
        "k"
    ]

    probability = result[
        "probability"
    ]

    C = result[
        "C"
    ]

    gamma = result[
        "gamma"
    ]

    r2 = result[
        "r2"
    ]

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # -------------------------------------------------------------
    # Observed distribution
    # -------------------------------------------------------------

    ax.plot(
        k,
        probability,
        "o",
        markersize=6,
        label=(
            "Observed cumulative "
            "distribution"
        ),
    )

    # -------------------------------------------------------------
    # Smooth fitted curve
    # -------------------------------------------------------------

    if (
        np.isfinite(C)
        and np.isfinite(gamma)
    ):

        k_smooth = np.linspace(
            1,
            max(k),
            500,
        )

        p_smooth = (
            exponential_degree_distribution(
                k_smooth,
                C,
                gamma,
            )
        )

        ax.plot(
            k_smooth,
            p_smooth,
            linewidth=2.0,
            label="Exponential fit",
        )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Node degree, $k$"
    )

    ax.set_ylabel(
        r"Cumulative probability, "
        r"$P(K \geq k)$"
    )

    graph_display = (
        graph_type.capitalize()
    )

    ax.set_title(
        f"{country} — "
        f"{voltage_label} — "
        f"{graph_display} graph"
    )

    if len(k) > 0:

        max_degree = int(
            max(k)
        )

        ax.set_xticks(
            np.arange(
                1,
                max_degree + 1,
                1,
            )
        )

    ax.set_ylim(
        0,
        1.05,
    )

    # -------------------------------------------------------------
    # Annotation
    # -------------------------------------------------------------

    if (
        np.isfinite(C)
        and np.isfinite(gamma)
    ):

        fit_text = (
            r"$P(K\geq k)=Ce^{-k/\gamma}$"
            "\n"
            rf"$C = {C:.4f}$"
            "\n"
            rf"$\gamma = {gamma:.4f}$"
            "\n"
            rf"$R^2 = {r2:.4f}$"
            "\n"
            rf"$\langle k\rangle = "
            rf"{result['mean_degree']:.4f}$"
            "\n"
            rf"$N = {result['n_nodes']}$"
            "\n"
            rf"$E = {result['n_branches']}$"
        )

        if graph_type == "simple":

            fit_text += (
                "\n"
                rf"$E_{{removed}} = "
                rf"{result['branches_removed']}$"
            )

    else:

        fit_text = (
            "Fit unavailable"
        )

    ax.text(
        0.97,
        0.97,
        fit_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={
            "boxstyle":
                "round",

            "facecolor":
                "white",

            "alpha":
                0.85,
        },
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(
            1.0,
            0.53,
        ),
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# ANALYZE ONE COUNTRY
# =====================================================================

def analyze_country(
    country,
    edges,
):
    """
    Analyze every voltage group twice:

        complete graph
        simple graph
    """

    summary_rows = []

    voltage_groups = (
        build_voltage_groups(
            edges
        )
    )

    for graph_type in (
        GRAPH_TYPES
    ):

        country_output_dir = (
            OUTPUT_DIR
            / graph_type
            / country
        )

        country_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for group in (
            voltage_groups
        ):

            group_key = group[
                "key"
            ]

            group_label = group[
                "label"
            ]

            group_edges = group[
                "edges"
            ]

            result = (
                analyze_network(
                    original_edges=
                        group_edges,

                    graph_type=
                        graph_type,
                )
            )

            # -----------------------------------------------------
            # Filename
            # -----------------------------------------------------

            if group_key == "All":

                filename = (
                    f"{country}_"
                    f"all_voltages.png"
                )

            else:

                filename = (
                    f"{country}_"
                    f"{safe_filename(group_key)}"
                    ".png"
                )

            plot_path = (
                country_output_dir
                / filename
            )

            plot_degree_distribution(
                result=result,
                country=country,
                voltage_label=
                    group_label,
                graph_type=
                    graph_type,
                output_path=
                    plot_path,
            )

            # -----------------------------------------------------
            # Summary
            # -----------------------------------------------------

            summary_rows.append(
                {
                    "country":
                        country,

                    "graph_type":
                        graph_type,

                    "voltage_group":
                        group_key,

                    "n_nodes":
                        result[
                            "n_nodes"
                        ],

                    "n_branches_original":
                        result[
                            "n_branches_original"
                        ],

                    "n_branches":
                        result[
                            "n_branches"
                        ],

                    "branches_removed":
                        result[
                            "branches_removed"
                        ],

                    "fraction_removed":
                        result[
                            "fraction_removed"
                        ],

                    "mean_degree":
                        result[
                            "mean_degree"
                        ],

                    "C":
                        result[
                            "C"
                        ],

                    "gamma":
                        result[
                            "gamma"
                        ],

                    "r2":
                        result[
                            "r2"
                        ],
                }
            )

    return summary_rows


# =====================================================================
# PRINT COUNTRY SUMMARY
# =====================================================================

def print_summary(
    summary,
):
    """
    Print complete and simple graph results by country.
    """

    print("\n")
    print("=" * 130)
    print(
        "EUROPEAN NODE-DEGREE "
        "CUMULATIVE EXPONENTIAL FITS"
    )
    print("=" * 130)

    for country in (
        summary[
            "country"
        ].unique()
    ):

        print("\n")
        print(country)
        print("=" * 130)

        for graph_type in (
            GRAPH_TYPES
        ):

            country_results = (
                summary.loc[
                    (
                        summary[
                            "country"
                        ]
                        == country
                    )
                    &
                    (
                        summary[
                            "graph_type"
                        ]
                        == graph_type
                    )
                ]
            )

            print(
                f"\n"
                f"{graph_type.upper()} GRAPH"
            )

            print(
                "-" * 130
            )

            print(
                f"{'Voltage':>18} "
                f"{'Nodes':>8} "
                f"{'Edges':>8} "
                f"{'Removed':>9} "
                f"{'Removed %':>11} "
                f"{'<k>':>10} "
                f"{'C':>10} "
                f"{'gamma':>10} "
                f"{'R2':>10}"
            )

            for _, row in (
                country_results
                .iterrows()
            ):

                print(
                    f"{row['voltage_group']:>18} "
                    f"{int(row['n_nodes']):>8d} "
                    f"{int(row['n_branches']):>8d} "
                    f"{int(row['branches_removed']):>9d} "
                    f"{100 * row['fraction_removed']:>10.2f}% "
                    f"{row['mean_degree']:>10.4f} "
                    f"{row['C']:>10.4f} "
                    f"{row['gamma']:>10.4f} "
                    f"{row['r2']:>10.4f}"
                )


# =====================================================================
# PARAMETER SUMMARY TABLE
# =====================================================================

def build_parameter_table(
    summary,
):
    """
    Build a wide parameter table.

    Row:
        country

    Column hierarchy:
        graph_type
            voltage_group
                C
                gamma
    """

    parameter_long = (
        summary[
            [
                "country",
                "graph_type",
                "voltage_group",
                "C",
                "gamma",
            ]
        ]
        .copy()
    )

    parameter_table = (
        parameter_long
        .pivot(
            index="country",
            columns=[
                "graph_type",
                "voltage_group",
            ],
            values=[
                "C",
                "gamma",
            ],
        )
    )

    # Move parameter name to innermost level:
    #
    # complete
    #   110 kV
    #       C
    #       gamma
    #
    parameter_table = (
        parameter_table
        .reorder_levels(
            [
                1,
                2,
                0,
            ],
            axis=1,
        )
        .sort_index(
            axis=1,
            level=[
                0,
                1,
            ],
            sort_remaining=False,
        )
    )

    parameter_table.columns.names = [
        "graph_type",
        "voltage_group",
        "parameter",
    ]

    return parameter_table


def print_parameter_table(
    parameter_table,
):
    """
    Print wide fitted-parameter table.
    """

    print("\n")
    print("=" * 130)
    print(
        "FITTED PARAMETER SUMMARY"
    )
    print("=" * 130)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.precision",
        4,
    ):

        print(
            parameter_table
            .to_string()
        )


# =====================================================================
# COMPLETE VS SIMPLE COMPARISON TABLE
# =====================================================================

def build_graph_comparison_table(
    summary,
):
    """
    Build a compact table showing how simplification changes C and
    gamma for every country/voltage group.
    """

    complete = (
        summary.loc[
            summary[
                "graph_type"
            ]
            == "complete"
        ]
        .set_index(
            [
                "country",
                "voltage_group",
            ]
        )
    )

    simple = (
        summary.loc[
            summary[
                "graph_type"
            ]
            == "simple"
        ]
        .set_index(
            [
                "country",
                "voltage_group",
            ]
        )
    )

    comparison = pd.DataFrame(
        index=complete.index
    )

    comparison[
        "n_branches_complete"
    ] = complete[
        "n_branches"
    ]

    comparison[
        "n_branches_simple"
    ] = simple[
        "n_branches"
    ]

    comparison[
        "edges_removed_pct"
    ] = (
        100
        * (
            complete[
                "n_branches"
            ]
            - simple[
                "n_branches"
            ]
        )
        / complete[
            "n_branches"
        ]
    )

    comparison[
        "C_complete"
    ] = complete[
        "C"
    ]

    comparison[
        "C_simple"
    ] = simple[
        "C"
    ]

    comparison[
        "C_change_pct"
    ] = (
        100
        * (
            simple["C"]
            - complete["C"]
        )
        / complete["C"]
    )

    comparison[
        "gamma_complete"
    ] = complete[
        "gamma"
    ]

    comparison[
        "gamma_simple"
    ] = simple[
        "gamma"
    ]

    comparison[
        "gamma_change_pct"
    ] = (
        100
        * (
            simple["gamma"]
            - complete["gamma"]
        )
        / complete["gamma"]
    )

    comparison[
        "R2_complete"
    ] = complete[
        "r2"
    ]

    comparison[
        "R2_simple"
    ] = simple[
        "r2"
    ]

    return (
        comparison
        .reset_index()
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 112)
    print(
        "LOADING EUROPEAN NETWORK DATA"
    )
    print("=" * 112)

    print(
        f"Input:\n"
        f"  {INPUT_FILE}"
    )

    euro_networks = (
        pd.read_pickle(
            INPUT_FILE
        )
    )

    print(
        f"\nLoaded "
        f"{len(euro_networks)} "
        f"countries."
    )

    # -----------------------------------------------------------------
    # Analyze
    # -----------------------------------------------------------------

    all_summary_rows = []

    for country in sorted(
        euro_networks
    ):

        edges = (
            euro_networks[
                country
            ]
        )

        print(
            f"Analyzing "
            f"{country:<25} "
            f"({len(edges):>5} branches)"
        )

        country_summary = (
            analyze_country(
                country,
                edges,
            )
        )

        all_summary_rows.extend(
            country_summary
        )

    # -----------------------------------------------------------------
    # Long-form summary
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary_rows
    )

    print_summary(
        summary
    )

    # -----------------------------------------------------------------
    # Wide fitted-parameter table
    # -----------------------------------------------------------------

    parameter_table = (
        build_parameter_table(
            summary
        )
    )

    print_parameter_table(
        parameter_table
    )

    # -----------------------------------------------------------------
    # Direct complete/simple comparison
    # -----------------------------------------------------------------

    graph_comparison = (
        build_graph_comparison_table(
            summary
        )
    )

    print("\n")
    print("=" * 130)
    print(
        "COMPLETE VS SIMPLE GRAPH COMPARISON"
    )
    print("=" * 130)

    print(
        graph_comparison
        .round(
            {
                "edges_removed_pct":
                    2,

                "C_complete":
                    4,

                "C_simple":
                    4,

                "C_change_pct":
                    2,

                "gamma_complete":
                    4,

                "gamma_simple":
                    4,

                "gamma_change_pct":
                    2,

                "R2_complete":
                    4,

                "R2_simple":
                    4,
            }
        )
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    summary_csv_path = (
        OUTPUT_DIR
        / "node_degree_fit_summary.csv"
    )

    summary_pickle_path = (
        OUTPUT_DIR
        / "node_degree_fit_summary.pkl"
    )

    parameter_csv_path = (
        OUTPUT_DIR
        / "fitted_parameter_table.csv"
    )

    parameter_pickle_path = (
        OUTPUT_DIR
        / "fitted_parameter_table.pkl"
    )

    comparison_csv_path = (
        OUTPUT_DIR
        / "complete_vs_simple_comparison.csv"
    )

    comparison_pickle_path = (
        OUTPUT_DIR
        / "complete_vs_simple_comparison.pkl"
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    summary.to_csv(
        summary_csv_path,
        index=False,
    )

    summary.to_pickle(
        summary_pickle_path
    )

    parameter_table.to_csv(
        parameter_csv_path
    )

    parameter_table.to_pickle(
        parameter_pickle_path
    )

    graph_comparison.to_csv(
        comparison_csv_path,
        index=False,
    )

    graph_comparison.to_pickle(
        comparison_pickle_path
    )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 112)
    print("OUTPUTS")
    print("=" * 112)

    print(
        f"Plots:\n"
        f"  {OUTPUT_DIR / 'complete'}\n"
        f"  {OUTPUT_DIR / 'simple'}"
    )

    print(
        f"\nLong-form summary:\n"
        f"  {summary_csv_path}\n"
        f"  {summary_pickle_path}"
    )

    print(
        f"\nFitted parameter table:\n"
        f"  {parameter_csv_path}\n"
        f"  {parameter_pickle_path}"
    )

    print(
        f"\nComplete vs simple comparison:\n"
        f"  {comparison_csv_path}\n"
        f"  {comparison_pickle_path}"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()