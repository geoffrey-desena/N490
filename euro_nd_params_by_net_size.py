#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot anchored exponential parameters versus network size
=========================================================

Loads:

    euro-comparison/european_networks.pkl

For each country:

    - combine all voltage levels below 200 kV into one
      subtransmission network

    - treat every voltage level >= 200 kV separately

    - convert each network to a SIMPLE graph

    - calculate the complementary cumulative node-degree distribution

    - fit for k >= 2:

          P(K >= k) = A * exp(-(k - 2) / gamma)

Creates two plots:

    1. A versus number of nodes
    2. gamma versus number of nodes

Voltage colors:

    <200 kV       black
    200-299 kV    green
    300-349 kV    gold
    >=350 kV      red

Countries are not labeled.

Nothing is saved.
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

INPUT_FILE = (
    WORKING_DIR
    / "euro-comparison"
    / "european_networks.pkl"
)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200.0

MIN_FIT_DEGREE = 2

FIGSIZE = (8.5, 6.5)

# A logarithmic x-axis is useful because network sizes range from
# only a few nodes to several hundred.
USE_LOG_NODE_AXIS = True


# =====================================================================
# VOLTAGE COLORS
# =====================================================================

VOLTAGE_COLORS = {
    "<200 kV":
        "black",

    "200-299 kV":
        "green",

    "300-349 kV":
        "gold",

    ">=350 kV":
        "red",
}


# =====================================================================
# ANCHORED EXPONENTIAL MODEL
# =====================================================================

def anchored_exponential_degree_distribution(
    k,
    A,
    gamma,
):
    """
    Anchored exponential CCDF:

        P(K >= k) = A * exp(-(k - 2) / gamma)

    so that:

        P(K >= 2) = A
    """

    return (
        A
        * np.exp(
            -(k - 2.0)
            / gamma
        )
    )


# =====================================================================
# SIMPLE GRAPH
# =====================================================================

def make_simple_graph(
    edges,
):
    """
    Collapse parallel branches connecting the same unordered node pair.
    """

    if edges.empty:

        return edges.copy()

    simple_edges = edges.copy()

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


# =====================================================================
# NODE DEGREE
# =====================================================================

def calculate_node_degrees(
    edges,
):
    """
    Calculate node degree from the supplied edge list.
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
# COMPLEMENTARY CUMULATIVE DISTRIBUTION
# =====================================================================

def calculate_ccdf(
    degrees,
):
    """
    Calculate:

        P(K >= k)

    for:

        k = 1, 2, ..., max_degree
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
            for degree
            in k
        ],
        dtype=float,
    )

    return (
        k,
        probability,
    )


# =====================================================================
# ANCHORED EXPONENTIAL FIT
# =====================================================================

def fit_anchored_exponential(
    k,
    probability,
):
    """
    Fit:

        P(K >= k) = A * exp(-(k - 2) / gamma)

    using only:

        k >= 2
    """

    fit_mask = (
        k >= MIN_FIT_DEGREE
    )

    k_fit = (
        k[
            fit_mask
        ]
    )

    probability_fit = (
        probability[
            fit_mask
        ]
    )

    n_fit_points = len(
        k_fit
    )

    # Two free parameters require at least two points.
    if n_fit_points < 2:

        return {
            "A":
                np.nan,

            "gamma":
                np.nan,

            "n_fit_points":
                n_fit_points,
        }

    A_initial = float(
        probability_fit[0]
    )

    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            anchored_exponential_degree_distribution,
            k_fit,
            probability_fit,
            p0=[
                A_initial,
                gamma_initial,
            ],
            bounds=(
                [
                    0.0,
                    1e-8,
                ],
                [
                    1.0,
                    np.inf,
                ],
            ),
            maxfev=100000,
        )

        A = float(
            popt[0]
        )

        gamma = float(
            popt[1]
        )

    except (
        RuntimeError,
        ValueError,
    ):

        return {
            "A":
                np.nan,

            "gamma":
                np.nan,

            "n_fit_points":
                n_fit_points,
        }

    return {
        "A":
            A,

        "gamma":
            gamma,

        "n_fit_points":
            n_fit_points,
    }


# =====================================================================
# VOLTAGE GROUPING
# =====================================================================

def build_voltage_groups(
    edges,
):
    """
    Build:

        - one combined <200 kV network
        - each >=200 kV voltage separately
    """

    groups = []

    # -------------------------------------------------------------
    # Combined subtransmission network
    # -------------------------------------------------------------

    sub_edges = edges.loc[
        edges["voltage_kv"]
        < SUBTRANSMISSION_LIMIT_KV
    ].copy()

    if not sub_edges.empty:

        groups.append(
            {
                "voltage_class":
                    "<200 kV",

                "voltage":
                    float(
                        sub_edges[
                            "voltage_kv"
                        ].max()
                    ),

                "edges":
                    sub_edges,
            }
        )

    # -------------------------------------------------------------
    # Individual >=200 kV networks
    # -------------------------------------------------------------

    voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            >= SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .dropna()
        .unique()
    )

    for voltage in voltages:

        voltage_edges = edges.loc[
            edges["voltage_kv"]
            == voltage
        ].copy()

        if voltage < 300:

            voltage_class = (
                "200-299 kV"
            )

        elif voltage < 350:

            voltage_class = (
                "300-349 kV"
            )

        else:

            voltage_class = (
                ">=350 kV"
            )

        groups.append(
            {
                "voltage_class":
                    voltage_class,

                "voltage":
                    float(
                        voltage
                    ),

                "edges":
                    voltage_edges,
            }
        )

    return groups


# =====================================================================
# BUILD PARAMETER TABLE
# =====================================================================

def build_parameter_table(
    euro_networks,
):
    """
    Fit anchored A and gamma for all usable country-voltage networks.
    """

    rows = []

    for country in sorted(
        euro_networks
    ):

        country_edges = (
            euro_networks[
                country
            ]
            .copy()
        )

        voltage_groups = (
            build_voltage_groups(
                country_edges
            )
        )

        for group in voltage_groups:

            simple_edges = (
                make_simple_graph(
                    group[
                        "edges"
                    ]
                )
            )

            degrees = (
                calculate_node_degrees(
                    simple_edges
                )
            )

            (
                k,
                probability,
            ) = calculate_ccdf(
                degrees
            )

            fit = (
                fit_anchored_exponential(
                    k,
                    probability,
                )
            )

            rows.append(
                {
                    "country":
                        country,

                    "voltage":
                        group[
                            "voltage"
                        ],

                    "voltage_class":
                        group[
                            "voltage_class"
                        ],

                    "n_nodes":
                        len(
                            degrees
                        ),

                    "n_edges":
                        len(
                            simple_edges
                        ),

                    "n_fit_points":
                        fit[
                            "n_fit_points"
                        ],

                    "A":
                        fit[
                            "A"
                        ],

                    "gamma":
                        fit[
                            "gamma"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# GENERIC PARAMETER-VS-NODE-COUNT PLOT
# =====================================================================

def plot_parameter_vs_nodes(
    parameter_df,
    parameter,
    ylabel,
):
    """
    Plot one fitted parameter against network node count.
    """

    valid = (
        parameter_df
        .dropna(
            subset=[
                parameter,
                "n_nodes",
            ]
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    plot_order = [
        "<200 kV",
        "200-299 kV",
        "300-349 kV",
        ">=350 kV",
    ]

    for voltage_class in plot_order:

        group = valid.loc[
            valid[
                "voltage_class"
            ]
            == voltage_class
        ]

        if group.empty:

            continue

        ax.scatter(
            group[
                "n_nodes"
            ],
            group[
                parameter
            ],
            s=80,
            color=VOLTAGE_COLORS[
                voltage_class
            ],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.80,
            label=voltage_class,
        )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Number of nodes, N",
        fontsize=13,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=13,
    )

    if USE_LOG_NODE_AXIS:

        ax.set_xscale(
            "log"
        )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.legend(
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    plt.show()


# =====================================================================
# CONSOLE TABLE
# =====================================================================

def print_parameter_table(
    parameter_df,
):
    """
    Print the values used in the plots.
    """

    print("\n")
    print("=" * 110)
    print(
        "ANCHORED EXPONENTIAL PARAMETERS AND NETWORK SIZE"
    )
    print("=" * 110)

    print(
        parameter_df[
            [
                "country",
                "voltage",
                "voltage_class",
                "n_nodes",
                "n_edges",
                "n_fit_points",
                "A",
                "gamma",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "voltage":
                    "{:.0f}".format,

                "A":
                    "{:.4f}".format,

                "gamma":
                    "{:.4f}".format,
            },
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 100)
    print(
        "LOADING EUROPEAN NETWORK DATA"
    )
    print("=" * 100)

    print(
        f"Input:\n"
        f"  {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nCould not find:\n"
            f"  {INPUT_FILE}"
        )

    euro_networks = (
        pd.read_pickle(
            INPUT_FILE
        )
    )

    print(
        f"\nCountries loaded: "
        f"{len(euro_networks)}"
    )

    # -------------------------------------------------------------
    # Fit parameters
    # -------------------------------------------------------------

    parameter_df = (
        build_parameter_table(
            euro_networks
        )
    )

    print_parameter_table(
        parameter_df
    )

    # -------------------------------------------------------------
    # A versus N
    # -------------------------------------------------------------

    plot_parameter_vs_nodes(
        parameter_df=
            parameter_df,

        parameter=
            "A",

        ylabel=
            r"$A$",
    )

    # -------------------------------------------------------------
    # gamma versus N
    # -------------------------------------------------------------

    plot_parameter_vs_nodes(
        parameter_df=
            parameter_df,

        parameter=
            "gamma",

        ylabel=
            r"$\gamma$",
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()