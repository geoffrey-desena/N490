#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot anchored exponential A-gamma parameters by voltage level
=============================================================

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

Plot
----
All fitted A-gamma values on one scatter plot.

Color:
    <200 kV       black
    200-299 kV    green
    300-349 kV    yellow
    >=350 kV      red

Marker area is proportional to the number of nodes in the network.

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

FIGSIZE = (9.0, 7.0)

# Marker-area scaling.
MIN_MARKER_SIZE = 40
MAX_MARKER_SIZE = 500


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

    such that:

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
    Collapse parallel edges connecting the same unordered node pair.
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

    for k = 1 ... max degree.
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

    using only k >= 2.
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

    if len(k_fit) < 2:

        return (
            np.nan,
            np.nan,
        )

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

        return (
            np.nan,
            np.nan,
        )

    return (
        A,
        gamma,
    )


# =====================================================================
# VOLTAGE GROUPING
# =====================================================================

def build_voltage_groups(
    edges,
):
    """
    Build one aggregated <200 kV network and individual >=200 kV
    networks.
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
                "voltage":
                    float(
                        sub_edges[
                            "voltage_kv"
                        ].max()
                    ),

                "voltage_class":
                    "<200 kV",

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
                "voltage":
                    float(
                        voltage
                    ),

                "voltage_class":
                    voltage_class,

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
    Fit A and gamma for every usable country/voltage network.
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

        groups = (
            build_voltage_groups(
                country_edges
            )
        )

        for group in groups:

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

            (
                A,
                gamma,
            ) = fit_anchored_exponential(
                k,
                probability,
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

                    "A":
                        A,

                    "gamma":
                        gamma,
                }
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# MARKER SIZE
# =====================================================================

def node_count_to_marker_size(
    n_nodes,
    min_nodes,
    max_nodes,
):
    """
    Scale marker AREA according to number of network nodes.
    """

    if max_nodes <= min_nodes:

        return (
            (
                MIN_MARKER_SIZE
                + MAX_MARKER_SIZE
            )
            / 2.0
        )

    fraction = (
        (
            n_nodes
            - min_nodes
        )
        /
        (
            max_nodes
            - min_nodes
        )
    )

    return (
        MIN_MARKER_SIZE
        + fraction
        * (
            MAX_MARKER_SIZE
            - MIN_MARKER_SIZE
        )
    )


# =====================================================================
# PLOT
# =====================================================================

def plot_parameter_space(
    parameter_df,
):
    """
    Plot every fitted A-gamma pair.

    Color indicates voltage class.
    Marker area indicates node count.
    """

    valid = (
        parameter_df
        .dropna(
            subset=[
                "A",
                "gamma",
            ]
        )
        .copy()
    )

    min_nodes = (
        valid[
            "n_nodes"
        ]
        .min()
    )

    max_nodes = (
        valid[
            "n_nodes"
        ]
        .max()
    )

    valid[
        "marker_size"
    ] = [
        node_count_to_marker_size(
            n_nodes,
            min_nodes,
            max_nodes,
        )
        for n_nodes
        in valid[
            "n_nodes"
        ]
    ]

    # -------------------------------------------------------------
    # Voltage colors
    # -------------------------------------------------------------

    color_map = {
        "<200 kV":
            "black",

        "200-299 kV":
            "green",

        "300-349 kV":
            "gold",

        ">=350 kV":
            "red",
    }

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # Plot in voltage order so high-voltage points don't always hide
    # lower-voltage ones.
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
                "A"
            ],
            group[
                "gamma"
            ],
            s=group[
                "marker_size"
            ],
            color=color_map[
                voltage_class
            ],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.75,
            label=voltage_class,
        )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        r"$A$",
        fontsize=13,
    )

    ax.set_ylabel(
        r"$\gamma$",
        fontsize=13,
    )

    ax.legend(
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    plt.show()


# =====================================================================
# OPTIONAL SIZE LEGEND
# =====================================================================

def print_parameter_table(
    parameter_df,
):
    """
    Print values used in the scatter plot.
    """

    print("\n")
    print("=" * 100)
    print(
        "A-GAMMA VALUES BY VOLTAGE CLASS"
    )
    print("=" * 100)

    print(
        parameter_df[
            [
                "country",
                "voltage",
                "voltage_class",
                "n_nodes",
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

    parameter_df = (
        build_parameter_table(
            euro_networks
        )
    )

    print_parameter_table(
        parameter_df
    )

    plot_parameter_space(
        parameter_df
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()