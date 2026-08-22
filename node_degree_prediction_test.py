#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-one-country-out subtransmission parameter validation
==========================================================

Tests whether the anchored exponential node-degree CCDF parameters of an
unknown subtransmission network can be predicted from its known
higher-voltage networks.

MODEL
-----

For each network, calculate the complementary cumulative node-degree
distribution:

    P(K >= k)

The point:

    P(K >= 1) = 1

is deterministic for these data and is therefore excluded from both
fitting and fit-quality evaluation.

The fitted model is:

    P(K >= k) = A * exp(-(k - 2) / gamma),     k >= 2

so that:

    A = fitted P(K >= 2)

This gives A a direct interpretation at the first degree included in
the fit, rather than extrapolating an intercept to k = 0.

GRAPH REPRESENTATION
--------------------

This script uses SIMPLE graphs:

    - parallel circuits between the same unordered node pair are
      collapsed to one edge
    - node degrees are calculated after simplification

NETWORK GROUPS
--------------

For every country:

    1. All voltage levels below 200 kV are aggregated into one
       subtransmission network.

    2. Each voltage level >= 200 kV is treated separately.

LEAVE-ONE-COUNTRY-OUT VALIDATION
--------------------------------

For each country in turn:

    1. Hold out that country's subtransmission A and gamma.

    2. Retain its >=200 kV fitted A and gamma values as known
       information.

    3. Compare its higher-voltage networks with those of every other
       country.

    4. Compare countries only at EXACT nominal voltages they share.

    5. Standardize A and gamma differences by their across-country
       standard deviations at that voltage.

    6. Calculate RMS parameter-space distance across all common
       voltages.

    7. Select the N most similar comparison countries.

    8. Predict the held-out country's subtransmission A and gamma using
       inverse-distance weighting of those countries' actual
       subtransmission parameters.

INPUT
-----

    euro-comparison/
        european_networks.pkl

OUTPUTS
-------

Directory:

    euro-comparison/
        node-degree-prediction-test/

Files:

    anchored_parameter_summary.csv
    leave_one_country_out_predictions.csv
    leave_one_country_out_statistics.csv
    leave_one_country_out_neighbors.csv
    actual_vs_predicted_A_gamma.png

Console:

    - fitted A/gamma parameters
    - subtransmission targets
    - leave-one-out predictions
    - validation statistics
    - neighbors and weights for every prediction
"""

from pathlib import Path
import colorsys

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

OUTPUT_DIR = (
    WORKING_DIR
    / "euro-comparison"
    / "node-degree-prediction-test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200.0

MIN_FIT_DEGREE = 2

# Number of closest comparison countries used in each prediction.
#
# Set to None to use every eligible comparison country.
N_NEIGHBORS = 5

# Minimum number of exact >=200 kV voltage levels two countries must
# share to be comparable.
MIN_COMMON_VOLTAGES = 1

# Prevent division by zero in inverse-distance weighting.
DISTANCE_EPSILON = 1e-6

# Optional filter on anchored-exponential fit quality.
#
# Leave as None initially.
MIN_FIT_R2 = None

SIMILARITY_PARAMETERS = [
    "A",
    "gamma",
]

FIGSIZE = (10.5, 8.0)

DPI = 300


# =====================================================================
# ANCHORED EXPONENTIAL MODEL
# =====================================================================

def anchored_exponential_degree_distribution(
    k,
    A,
    gamma,
):
    """
    Anchored complementary cumulative exponential model:

        P(K >= k) = A * exp(-(k - 2) / gamma)

    Therefore:

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
# SIMPLE-GRAPH CONVERSION
# =====================================================================

def make_simple_graph(
    edges,
):
    """
    Collapse parallel edges connecting the same unordered node pair.

    Direction is ignored.

    Thus:

        node_i = 4, node_j = 12

    is treated as identical to:

        node_i = 12, node_j = 4
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
    # Keep one edge per unordered node pair
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
# DEGREE CALCULATION
# =====================================================================

def calculate_node_degrees(
    edges,
):
    """
    Calculate node degrees from an edge list.
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

def calculate_ccdf(
    degrees,
):
    """
    Calculate:

        P(K >= k)

    for:

        k = 1, 2, ..., max_degree

    P(K >= 1) = 1 by construction.

    The k = 1 point is calculated but is not used in the model fit.
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
# FIT METRICS
# =====================================================================

def calculate_fit_metrics(
    observed,
    predicted,
):
    """
    Calculate R^2 and RMSE.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    residuals = (
        observed
        - predicted
    )

    residual_sum_squares = np.sum(
        residuals ** 2
    )

    total_sum_squares = np.sum(
        (
            observed
            - np.mean(
                observed
            )
        ) ** 2
    )

    if total_sum_squares > 0:

        r2 = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )

    else:

        r2 = np.nan

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    return (
        float(r2),
        float(rmse),
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

    R^2 and RMSE are also calculated only for k >= 2.

    A is constrained to:

        0 <= A <= 1

    because it represents fitted P(K >= 2).
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

    # Two free parameters require at least two CCDF points.
    if n_fit_points < 2:

        return {
            "A":
                np.nan,

            "gamma":
                np.nan,

            "r2":
                np.nan,

            "rmse":
                np.nan,

            "n_fit_points":
                n_fit_points,
        }

    # -------------------------------------------------------------
    # Initial estimates
    # -------------------------------------------------------------

    A_initial = float(
        probability_fit[0]
    )

    gamma_initial = 2.0

    # -------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------

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

            "r2":
                np.nan,

            "rmse":
                np.nan,

            "n_fit_points":
                n_fit_points,
        }

    # -------------------------------------------------------------
    # Evaluate on exactly the same k >= 2 domain
    # -------------------------------------------------------------

    predicted = (
        anchored_exponential_degree_distribution(
            k_fit,
            A,
            gamma,
        )
    )

    (
        r2,
        rmse,
    ) = calculate_fit_metrics(
        observed=
            probability_fit,

        predicted=
            predicted,
    )

    return {
        "A":
            A,

        "gamma":
            gamma,

        "r2":
            r2,

        "rmse":
            rmse,

        "n_fit_points":
            n_fit_points,
    }


# =====================================================================
# VOLTAGE GROUPS
# =====================================================================

def get_subtransmission_label(
    edges,
):
    """
    Build display label for the combined network below 200 kV.

    Examples
    --------

    110 kV

    110-150 kV

    132-165 kV
    """

    voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            < SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if len(voltages) == 0:

        return None

    if len(voltages) == 1:

        return (
            f"{voltages[0]:g} kV"
        )

    return (
        f"{voltages[0]:g}-"
        f"{voltages[-1]:g} kV"
    )


def build_voltage_groups(
    edges,
):
    """
    Build network groups for one country.

    Groups:

        1. All <200 kV edges aggregated together.
        2. Each >=200 kV voltage separately.

    Simplification is applied AFTER selecting the voltage group.
    """

    groups = []

    # -------------------------------------------------------------
    # Combined subtransmission network
    # -------------------------------------------------------------

    sub_edges = edges.loc[
        edges["voltage_kv"]
        < SUBTRANSMISSION_LIMIT_KV
    ].copy()

    if len(sub_edges) > 0:

        groups.append(
            {
                "voltage_group":
                    get_subtransmission_label(
                        edges
                    ),

                "comparison_voltage":
                    np.nan,

                "is_subtransmission":
                    True,

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
        .dropna()
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

        groups.append(
            {
                "voltage_group":
                    f"{voltage:g} kV",

                "comparison_voltage":
                    float(
                        voltage
                    ),

                "is_subtransmission":
                    False,

                "edges":
                    voltage_edges,
            }
        )

    return groups


# =====================================================================
# FIT ALL EUROPEAN NETWORKS
# =====================================================================

def build_parameter_table(
    euro_networks,
):
    """
    Calculate anchored A/gamma parameters directly from the raw network
    data.

    Only simple graphs are analyzed.
    """

    rows = []

    print("\n")
    print("=" * 105)
    print(
        "FITTING ANCHORED EXPONENTIAL PARAMETERS"
    )
    print("=" * 105)

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

        print(
            f"\n{country}"
        )

        for group in (
            voltage_groups
        ):

            original_edges = (
                group[
                    "edges"
                ]
            )

            simple_edges = (
                make_simple_graph(
                    original_edges
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

            n_nodes = len(
                degrees
            )

            n_edges_original = len(
                original_edges
            )

            n_edges_simple = len(
                simple_edges
            )

            edges_removed = (
                n_edges_original
                - n_edges_simple
            )

            if n_nodes > 0:

                mean_degree = float(
                    degrees.mean()
                )

            else:

                mean_degree = np.nan

            rows.append(
                {
                    "country":
                        country,

                    "voltage_group":
                        group[
                            "voltage_group"
                        ],

                    "comparison_voltage":
                        group[
                            "comparison_voltage"
                        ],

                    "is_subtransmission":
                        group[
                            "is_subtransmission"
                        ],

                    "n_nodes":
                        n_nodes,

                    "n_edges_original":
                        n_edges_original,

                    "n_edges_simple":
                        n_edges_simple,

                    "edges_removed":
                        edges_removed,

                    "mean_degree":
                        mean_degree,

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

                    "r2":
                        fit[
                            "r2"
                        ],

                    "rmse":
                        fit[
                            "rmse"
                        ],
                }
            )

            print(
                f"  "
                f"{group['voltage_group']:<15} "
                f"N={n_nodes:>4d}  "
                f"E={n_edges_simple:>4d}  "
                f"A={fit['A']:>7.4f}  "
                f"gamma={fit['gamma']:>7.4f}  "
                f"R2={fit['r2']:>7.4f}  "
                f"RMSE={fit['rmse']:>8.5f}  "
                f"points={fit['n_fit_points']}"
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# PREPARE TARGET AND PREDICTOR NETWORKS
# =====================================================================

def get_subtransmission_rows(
    parameter_df,
):
    """
    Extract the combined <200 kV target network for each country.
    """

    sub = parameter_df.loc[
        parameter_df[
            "is_subtransmission"
        ]
    ].copy()

    # Require valid anchored fit.
    sub = sub.dropna(
        subset=[
            "A",
            "gamma",
        ]
    ).copy()

    if MIN_FIT_R2 is not None:

        sub = sub.loc[
            sub["r2"]
            >= MIN_FIT_R2
        ].copy()

    counts = (
        sub.groupby(
            "country"
        )
        .size()
    )

    bad_counts = (
        counts.loc[
            counts != 1
        ]
    )

    if not bad_counts.empty:

        raise ValueError(
            "Expected exactly one valid combined "
            "<200 kV network per country.\n\n"
            + bad_counts.to_string()
        )

    sub = sub.rename(
        columns={
            "voltage_group":
                "subtransmission_voltage_group",

            "A":
                "actual_A",

            "gamma":
                "actual_gamma",

            "r2":
                "actual_r2",

            "rmse":
                "actual_rmse",

            "n_fit_points":
                "actual_n_fit_points",
        }
    )

    return (
        sub
        .sort_values(
            "country"
        )
        .reset_index(
            drop=True
        )
    )


def prepare_high_voltage_rows(
    parameter_df,
):
    """
    Extract valid individual >=200 kV network fits.
    """

    high = parameter_df.loc[
        ~parameter_df[
            "is_subtransmission"
        ]
    ].copy()

    high = high.dropna(
        subset=[
            "comparison_voltage",
            "A",
            "gamma",
        ]
    ).copy()

    if MIN_FIT_R2 is not None:

        high = high.loc[
            high["r2"]
            >= MIN_FIT_R2
        ].copy()

    return (
        high
        .sort_values(
            [
                "country",
                "comparison_voltage",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =====================================================================
# INPUT DIAGNOSTICS
# =====================================================================

def print_parameter_table(
    parameter_df,
):
    """
    Print all fitted anchored parameters.
    """

    print("\n")
    print("=" * 135)
    print(
        "AVAILABLE SIMPLE-GRAPH ANCHORED EXPONENTIAL PARAMETERS"
    )
    print("=" * 135)

    display = (
        parameter_df[
            [
                "country",
                "voltage_group",
                "n_nodes",
                "n_edges_simple",
                "n_fit_points",
                "A",
                "gamma",
                "r2",
                "rmse",
            ]
        ]
        .copy()
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "A":
                    "{:.4f}".format,

                "gamma":
                    "{:.4f}".format,

                "r2":
                    "{:.4f}".format,

                "rmse":
                    "{:.5f}".format,
            },
        )
    )


def print_subtransmission_targets(
    subtransmission,
):
    """
    Print actual held-out target parameters.
    """

    print("\n")
    print("=" * 115)
    print(
        "SUBTRANSMISSION TARGET PARAMETERS"
    )
    print("=" * 115)

    print(
        subtransmission[
            [
                "country",
                "subtransmission_voltage_group",
                "actual_A",
                "actual_gamma",
                "actual_r2",
                "actual_rmse",
                "actual_n_fit_points",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "actual_A":
                    "{:.4f}".format,

                "actual_gamma":
                    "{:.4f}".format,

                "actual_r2":
                    "{:.4f}".format,

                "actual_rmse":
                    "{:.5f}".format,
            },
        )
    )


# =====================================================================
# COUNTRY HIGH-VOLTAGE SIGNATURE
# =====================================================================

def get_country_high_voltage_signature(
    high_voltage_df,
    country,
):
    """
    Return A/gamma parameters indexed by exact nominal voltage.
    """

    country_rows = high_voltage_df.loc[
        high_voltage_df[
            "country"
        ]
        == country
    ]

    return (
        country_rows
        .groupby(
            "comparison_voltage"
        )[
            SIMILARITY_PARAMETERS
        ]
        .mean()
        .sort_index()
    )


# =====================================================================
# PARAMETER SCALING
# =====================================================================

def calculate_voltage_scaling(
    comparison_high_voltage,
):
    """
    Calculate across-country standard deviations of A and gamma at
    every exact nominal voltage.

    Scaling is recalculated after the held-out country is removed.
    """

    scaling = {}

    for voltage, group in (
        comparison_high_voltage
        .groupby(
            "comparison_voltage"
        )
    ):

        scaling[
            voltage
        ] = {}

        for parameter in (
            SIMILARITY_PARAMETERS
        ):

            std = (
                group[
                    parameter
                ]
                .std(
                    ddof=1
                )
            )

            if (
                not np.isfinite(
                    std
                )
                or std <= 0
            ):

                std = 1.0

            scaling[
                voltage
            ][
                parameter
            ] = std

    return scaling


# =====================================================================
# COUNTRY SIMILARITY
# =====================================================================

def calculate_country_distance(
    target_country,
    comparison_country,
    high_voltage_df,
    scaling,
):
    """
    Calculate high-voltage topological distance between two countries.

    For every exact nominal voltage V shared by both countries:

        d_V^2 =
            ((A_target - A_comparison) / sigma_A,V)^2
            +
            ((gamma_target - gamma_comparison) / sigma_gamma,V)^2

    Overall distance:

        d = sqrt(mean(d_V^2))

    Lower distance means greater similarity.
    """

    target_signature = (
        get_country_high_voltage_signature(
            high_voltage_df,
            target_country,
        )
    )

    comparison_signature = (
        get_country_high_voltage_signature(
            high_voltage_df,
            comparison_country,
        )
    )

    common_voltages = (
        target_signature.index
        .intersection(
            comparison_signature.index
        )
    )

    if (
        len(
            common_voltages
        )
        < MIN_COMMON_VOLTAGES
    ):

        return {
            "distance":
                np.nan,

            "n_common_voltages":
                len(
                    common_voltages
                ),

            "common_voltages":
                "",
        }

    voltage_distance_squared = []

    used_voltages = []

    for voltage in (
        common_voltages
    ):

        if voltage not in scaling:

            continue

        distance_squared = 0.0

        valid_voltage = True

        for parameter in (
            SIMILARITY_PARAMETERS
        ):

            target_value = (
                target_signature.loc[
                    voltage,
                    parameter,
                ]
            )

            comparison_value = (
                comparison_signature.loc[
                    voltage,
                    parameter,
                ]
            )

            if (
                not np.isfinite(
                    target_value
                )
                or not np.isfinite(
                    comparison_value
                )
            ):

                valid_voltage = False

                break

            scale = (
                scaling[
                    voltage
                ][
                    parameter
                ]
            )

            difference = (
                target_value
                - comparison_value
            )

            distance_squared += (
                difference
                / scale
            ) ** 2

        if valid_voltage:

            voltage_distance_squared.append(
                distance_squared
            )

            used_voltages.append(
                voltage
            )

    if (
        len(
            voltage_distance_squared
        )
        < MIN_COMMON_VOLTAGES
    ):

        return {
            "distance":
                np.nan,

            "n_common_voltages":
                len(
                    voltage_distance_squared
                ),

            "common_voltages":
                "",
        }

    distance = np.sqrt(
        np.mean(
            voltage_distance_squared
        )
    )

    voltage_string = ", ".join(
        f"{voltage:g}"
        for voltage
        in used_voltages
    )

    return {
        "distance":
            float(
                distance
            ),

        "n_common_voltages":
            len(
                used_voltages
            ),

        "common_voltages":
            voltage_string,
    }


# =====================================================================
# ONE LEAVE-ONE-OUT PREDICTION
# =====================================================================

def predict_one_country(
    target_country,
    high_voltage_df,
    subtransmission_df,
):
    """
    Predict held-out subtransmission A and gamma.
    """

    # -------------------------------------------------------------
    # Remove target from scaling population
    # -------------------------------------------------------------

    comparison_high_voltage = (
        high_voltage_df.loc[
            high_voltage_df[
                "country"
            ]
            != target_country
        ]
        .copy()
    )

    scaling = (
        calculate_voltage_scaling(
            comparison_high_voltage
        )
    )

    countries_with_subtransmission = set(
        subtransmission_df[
            "country"
        ]
    )

    candidate_countries = sorted(
        (
            set(
                comparison_high_voltage[
                    "country"
                ]
            )
            & countries_with_subtransmission
        )
        - {
            target_country
        }
    )

    candidate_rows = []

    # -------------------------------------------------------------
    # Calculate similarity to every candidate country
    # -------------------------------------------------------------

    for comparison_country in (
        candidate_countries
    ):

        distance_result = (
            calculate_country_distance(
                target_country=
                    target_country,

                comparison_country=
                    comparison_country,

                high_voltage_df=
                    high_voltage_df,

                scaling=
                    scaling,
            )
        )

        distance = (
            distance_result[
                "distance"
            ]
        )

        if not np.isfinite(
            distance
        ):

            continue

        sub_row = (
            subtransmission_df.loc[
                subtransmission_df[
                    "country"
                ]
                == comparison_country
            ]
            .iloc[0]
        )

        candidate_rows.append(
            {
                "target_country":
                    target_country,

                "comparison_country":
                    comparison_country,

                "distance":
                    distance,

                "n_common_voltages":
                    distance_result[
                        "n_common_voltages"
                    ],

                "common_voltages_kv":
                    distance_result[
                        "common_voltages"
                    ],

                "comparison_A":
                    sub_row[
                        "actual_A"
                    ],

                "comparison_gamma":
                    sub_row[
                        "actual_gamma"
                    ],
            }
        )

    candidates = pd.DataFrame(
        candidate_rows
    )

    if candidates.empty:

        raise RuntimeError(
            f"No eligible comparison countries "
            f"were found for {target_country}."
        )

    candidates = (
        candidates
        .sort_values(
            [
                "distance",
                "comparison_country",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Keep nearest neighbors
    # -------------------------------------------------------------

    if N_NEIGHBORS is not None:

        candidates = (
            candidates
            .head(
                N_NEIGHBORS
            )
            .copy()
        )

    candidates[
        "rank"
    ] = np.arange(
        1,
        len(
            candidates
        ) + 1,
    )

    # -------------------------------------------------------------
    # Inverse-distance weighting
    # -------------------------------------------------------------

    candidates[
        "raw_weight"
    ] = (
        1.0
        / (
            candidates[
                "distance"
            ]
            + DISTANCE_EPSILON
        )
    )

    candidates[
        "weight"
    ] = (
        candidates[
            "raw_weight"
        ]
        / candidates[
            "raw_weight"
        ].sum()
    )

    # -------------------------------------------------------------
    # Weighted prediction
    # -------------------------------------------------------------

    predicted_A = np.sum(
        candidates[
            "weight"
        ]
        * candidates[
            "comparison_A"
        ]
    )

    predicted_gamma = np.sum(
        candidates[
            "weight"
        ]
        * candidates[
            "comparison_gamma"
        ]
    )

    return (
        float(
            predicted_A
        ),
        float(
            predicted_gamma
        ),
        candidates,
    )


# =====================================================================
# COMPLETE LEAVE-ONE-OUT VALIDATION
# =====================================================================

def run_leave_one_out(
    high_voltage_df,
    subtransmission_df,
):
    """
    Run validation once for every eligible country.
    """

    eligible_countries = sorted(
        set(
            high_voltage_df[
                "country"
            ]
        )
        & set(
            subtransmission_df[
                "country"
            ]
        )
    )

    results = []

    neighbor_tables = []

    print("\n")
    print("=" * 110)
    print(
        "LEAVE-ONE-COUNTRY-OUT "
        "SUBTRANSMISSION VALIDATION"
    )
    print("=" * 110)

    print(
        f"Eligible countries: "
        f"{len(eligible_countries)}"
    )

    print(
        f"Neighbor count: "
        f"{N_NEIGHBORS}"
    )

    print(
        f"Minimum common >=200 kV voltages: "
        f"{MIN_COMMON_VOLTAGES}"
    )

    print()

    for target_country in (
        eligible_countries
    ):

        (
            predicted_A,
            predicted_gamma,
            neighbors,
        ) = predict_one_country(
            target_country=
                target_country,

            high_voltage_df=
                high_voltage_df,

            subtransmission_df=
                subtransmission_df,
        )

        actual = (
            subtransmission_df.loc[
                subtransmission_df[
                    "country"
                ]
                == target_country
            ]
            .iloc[0]
        )

        actual_A = float(
            actual[
                "actual_A"
            ]
        )

        actual_gamma = float(
            actual[
                "actual_gamma"
            ]
        )

        difference_A = (
            predicted_A
            - actual_A
        )

        difference_gamma = (
            predicted_gamma
            - actual_gamma
        )

        absolute_difference_A = abs(
            difference_A
        )

        absolute_difference_gamma = abs(
            difference_gamma
        )

        if actual_A != 0:

            percent_difference_A = (
                100.0
                * difference_A
                / actual_A
            )

        else:

            percent_difference_A = np.nan

        if actual_gamma != 0:

            percent_difference_gamma = (
                100.0
                * difference_gamma
                / actual_gamma
            )

        else:

            percent_difference_gamma = np.nan

        A_gamma_distance = np.sqrt(
            difference_A ** 2
            + difference_gamma ** 2
        )

        results.append(
            {
                "country":
                    target_country,

                "subtransmission_voltage_group":
                    actual[
                        "subtransmission_voltage_group"
                    ],

                "actual_A":
                    actual_A,

                "predicted_A":
                    predicted_A,

                "difference_A":
                    difference_A,

                "absolute_difference_A":
                    absolute_difference_A,

                "percent_difference_A":
                    percent_difference_A,

                "actual_gamma":
                    actual_gamma,

                "predicted_gamma":
                    predicted_gamma,

                "difference_gamma":
                    difference_gamma,

                "absolute_difference_gamma":
                    absolute_difference_gamma,

                "percent_difference_gamma":
                    percent_difference_gamma,

                "A_gamma_distance":
                    A_gamma_distance,
            }
        )

        neighbor_tables.append(
            neighbors
        )

        print(
            f"{target_country:<22} "
            f"A: "
            f"{actual_A:>7.4f} -> "
            f"{predicted_A:>7.4f} "
            f"({difference_A:+7.4f})    "
            f"gamma: "
            f"{actual_gamma:>7.4f} -> "
            f"{predicted_gamma:>7.4f} "
            f"({difference_gamma:+7.4f})"
        )

    results = pd.DataFrame(
        results
    )

    neighbor_table = pd.concat(
        neighbor_tables,
        ignore_index=True,
    )

    return (
        results,
        neighbor_table,
    )


# =====================================================================
# VALIDATION STATISTICS
# =====================================================================

def calculate_r_squared(
    actual,
    predicted,
):
    """
    Calculate prediction R^2.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    residual_sum_squares = np.sum(
        (
            actual
            - predicted
        ) ** 2
    )

    total_sum_squares = np.sum(
        (
            actual
            - np.mean(
                actual
            )
        ) ** 2
    )

    if total_sum_squares <= 0:

        return np.nan

    return (
        1.0
        - residual_sum_squares
        / total_sum_squares
    )


def parameter_statistics(
    actual,
    predicted,
):
    """
    Calculate prediction statistics for one parameter.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = (
        predicted
        - actual
    )

    absolute_error = np.abs(
        error
    )

    valid_percent = (
        actual != 0
    )

    if np.any(
        valid_percent
    ):

        mape = (
            np.mean(
                np.abs(
                    error[
                        valid_percent
                    ]
                    / actual[
                        valid_percent
                    ]
                )
            )
            * 100.0
        )

    else:

        mape = np.nan

    if len(
        actual
    ) >= 2:

        pearson_r = np.corrcoef(
            actual,
            predicted,
        )[0, 1]

    else:

        pearson_r = np.nan

    return {
        "mean_difference_bias":
            np.mean(
                error
            ),

        "mean_absolute_error":
            np.mean(
                absolute_error
            ),

        "median_absolute_error":
            np.median(
                absolute_error
            ),

        "rmse":
            np.sqrt(
                np.mean(
                    error ** 2
                )
            ),

        "std_difference":
            np.std(
                error,
                ddof=1,
            ),

        "max_absolute_error":
            np.max(
                absolute_error
            ),

        "mean_absolute_percent_error":
            mape,

        "r2_actual_vs_predicted":
            calculate_r_squared(
                actual,
                predicted,
            ),

        "pearson_r":
            pearson_r,
    }


def build_statistics_table(
    results,
):
    """
    Build aggregate A/gamma prediction statistics.
    """

    A_stats = parameter_statistics(
        actual=
            results[
                "actual_A"
            ],

        predicted=
            results[
                "predicted_A"
            ],
    )

    gamma_stats = parameter_statistics(
        actual=
            results[
                "actual_gamma"
            ],

        predicted=
            results[
                "predicted_gamma"
            ],
    )

    rows = []

    for statistic_name in (
        A_stats
    ):

        rows.append(
            {
                "statistic":
                    statistic_name,

                "A":
                    A_stats[
                        statistic_name
                    ],

                "gamma":
                    gamma_stats[
                        statistic_name
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# CONSOLE TABLES
# =====================================================================

def print_results_table(
    results,
):
    """
    Print country-level prediction results.
    """

    print("\n")
    print("=" * 150)
    print(
        "COUNTRY-BY-COUNTRY "
        "SUBTRANSMISSION PARAMETER VALIDATION"
    )
    print("=" * 150)

    display_columns = [
        "country",
        "subtransmission_voltage_group",
        "actual_A",
        "predicted_A",
        "difference_A",
        "percent_difference_A",
        "actual_gamma",
        "predicted_gamma",
        "difference_gamma",
        "percent_difference_gamma",
        "A_gamma_distance",
    ]

    print(
        results[
            display_columns
        ]
        .to_string(
            index=False,
            formatters={
                "actual_A":
                    "{:.4f}".format,

                "predicted_A":
                    "{:.4f}".format,

                "difference_A":
                    "{:+.4f}".format,

                "percent_difference_A":
                    "{:+.2f}".format,

                "actual_gamma":
                    "{:.4f}".format,

                "predicted_gamma":
                    "{:.4f}".format,

                "difference_gamma":
                    "{:+.4f}".format,

                "percent_difference_gamma":
                    "{:+.2f}".format,

                "A_gamma_distance":
                    "{:.4f}".format,
            },
        )
    )


def print_statistics_table(
    statistics,
):
    """
    Print aggregate prediction statistics.
    """

    print("\n")
    print("=" * 95)
    print(
        "LEAVE-ONE-COUNTRY-OUT ERROR STATISTICS"
    )
    print("=" * 95)

    print(
        statistics.to_string(
            index=False,
            formatters={
                "A":
                    "{:.5f}".format,

                "gamma":
                    "{:.5f}".format,
            },
        )
    )


def print_neighbor_summary(
    neighbor_table,
):
    """
    Print countries and weights used for each prediction.
    """

    print("\n")
    print("=" * 120)
    print(
        "COMPARISON COUNTRIES USED "
        "FOR EACH PREDICTION"
    )
    print("=" * 120)

    for target_country, group in (
        neighbor_table
        .groupby(
            "target_country",
            sort=True,
        )
    ):

        print(
            f"\n{target_country}"
        )

        display = group[
            [
                "rank",
                "comparison_country",
                "distance",
                "n_common_voltages",
                "common_voltages_kv",
                "weight",
                "comparison_A",
                "comparison_gamma",
            ]
        ]

        print(
            display.to_string(
                index=False,
                formatters={
                    "distance":
                        "{:.4f}".format,

                    "weight":
                        "{:.4f}".format,

                    "comparison_A":
                        "{:.4f}".format,

                    "comparison_gamma":
                        "{:.4f}".format,
                },
            )
        )


# =====================================================================
# COLOR UTILITIES
# =====================================================================

def adjust_lightness(
    color,
    factor,
):
    """
    Darken or lighten an RGB/RGBA matplotlib color.
    """

    red, green, blue = (
        color[
            :3
        ]
    )

    hue, lightness, saturation = (
        colorsys.rgb_to_hls(
            red,
            green,
            blue,
        )
    )

    lightness = np.clip(
        lightness
        * factor,
        0.0,
        1.0,
    )

    return colorsys.hls_to_rgb(
        hue,
        lightness,
        saturation,
    )


# =====================================================================
# A-GAMMA PLOT
# =====================================================================

def plot_actual_vs_predicted(
    results,
):
    """
    Plot actual and predicted subtransmission parameters in A-gamma
    space.

    Each country has one color.

    Actual:
        darker shade

    Predicted:
        lighter shade

    Solid line:
        actual to predicted
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    countries = (
        results[
            "country"
        ]
        .tolist()
    )

    cmap = plt.get_cmap(
        "tab20"
    )

    base_colors = [
        cmap(
            index
            / max(
                len(
                    countries
                ) - 1,
                1,
            )
        )
        for index
        in range(
            len(
                countries
            )
        )
    ]

    for (
        color,
        (_, row),
    ) in zip(
        base_colors,
        results.iterrows(),
    ):

        actual_color = (
            adjust_lightness(
                color,
                0.70,
            )
        )

        predicted_color = (
            adjust_lightness(
                color,
                1.30,
            )
        )

        # ---------------------------------------------------------
        # Connecting line
        # ---------------------------------------------------------

        ax.plot(
            [
                row[
                    "actual_A"
                ],
                row[
                    "predicted_A"
                ],
            ],
            [
                row[
                    "actual_gamma"
                ],
                row[
                    "predicted_gamma"
                ],
            ],
            color=color,
            linewidth=1.7,
            zorder=1,
        )

        # ---------------------------------------------------------
        # Actual
        # ---------------------------------------------------------

        ax.scatter(
            row[
                "actual_A"
            ],
            row[
                "actual_gamma"
            ],
            s=90,
            marker="o",
            color=actual_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        # ---------------------------------------------------------
        # Predicted
        # ---------------------------------------------------------

        ax.scatter(
            row[
                "predicted_A"
            ],
            row[
                "predicted_gamma"
            ],
            s=90,
            marker="o",
            color=predicted_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        # ---------------------------------------------------------
        # Country legend
        # ---------------------------------------------------------

        ax.plot(
            [],
            [],
            color=color,
            linewidth=2.2,
            label=row[
                "country"
            ],
        )

    # -------------------------------------------------------------
    # Actual/predicted legend
    # -------------------------------------------------------------

    ax.scatter(
        [],
        [],
        s=90,
        marker="o",
        facecolor="0.30",
        edgecolor="black",
        linewidth=0.5,
        label="Actual — darker shade",
    )

    ax.scatter(
        [],
        [],
        s=90,
        marker="o",
        facecolor="0.80",
        edgecolor="black",
        linewidth=0.5,
        label="Predicted — lighter shade",
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

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "actual_vs_predicted_A_gamma.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"\nSaved plot:\n"
        f"  {output_path}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load raw network data
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 105)
    print(
        "LOADING EUROPEAN NETWORK DATA"
    )
    print("=" * 105)

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

    # -----------------------------------------------------------------
    # Fit anchored exponential parameters directly from raw data
    # -----------------------------------------------------------------

    parameter_df = (
        build_parameter_table(
            euro_networks
        )
    )

    print_parameter_table(
        parameter_df
    )

    # -----------------------------------------------------------------
    # Separate explanatory and target networks
    # -----------------------------------------------------------------

    high_voltage = (
        prepare_high_voltage_rows(
            parameter_df
        )
    )

    subtransmission = (
        get_subtransmission_rows(
            parameter_df
        )
    )

    print_subtransmission_targets(
        subtransmission
    )

    # -----------------------------------------------------------------
    # Eligibility
    # -----------------------------------------------------------------

    eligible_countries = sorted(
        set(
            high_voltage[
                "country"
            ]
        )
        & set(
            subtransmission[
                "country"
            ]
        )
    )

    print("\n")
    print("=" * 105)
    print(
        "ELIGIBILITY"
    )
    print("=" * 105)

    print(
        f"Countries with both >=200 kV data "
        f"and a valid subtransmission fit: "
        f"{len(eligible_countries)}"
    )

    print(
        ", ".join(
            eligible_countries
        )
    )

    # -----------------------------------------------------------------
    # Leave-one-country-out validation
    # -----------------------------------------------------------------

    (
        results,
        neighbors,
    ) = run_leave_one_out(
        high_voltage_df=
            high_voltage,

        subtransmission_df=
            subtransmission,
    )

    # -----------------------------------------------------------------
    # Aggregate prediction statistics
    # -----------------------------------------------------------------

    statistics = (
        build_statistics_table(
            results
        )
    )

    # -----------------------------------------------------------------
    # Console summaries
    # -----------------------------------------------------------------

    print_results_table(
        results
    )

    print_statistics_table(
        statistics
    )

    print_neighbor_summary(
        neighbors
    )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    parameter_path = (
        OUTPUT_DIR
        / "anchored_parameter_summary.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "leave_one_country_out_predictions.csv"
    )

    statistics_path = (
        OUTPUT_DIR
        / "leave_one_country_out_statistics.csv"
    )

    neighbors_path = (
        OUTPUT_DIR
        / "leave_one_country_out_neighbors.csv"
    )

    parameter_df.to_csv(
        parameter_path,
        index=False,
    )

    results.to_csv(
        predictions_path,
        index=False,
    )

    statistics.to_csv(
        statistics_path,
        index=False,
    )

    neighbors.to_csv(
        neighbors_path,
        index=False,
    )

    print("\n")
    print("=" * 105)
    print(
        "SAVED OUTPUT"
    )
    print("=" * 105)

    print(
        f"Anchored fitted parameters:\n"
        f"  {parameter_path}"
    )

    print(
        f"\nPredictions:\n"
        f"  {predictions_path}"
    )

    print(
        f"\nPrediction statistics:\n"
        f"  {statistics_path}"
    )

    print(
        f"\nNeighbor weights:\n"
        f"  {neighbors_path}"
    )

    # -----------------------------------------------------------------
    # A-gamma plot
    # -----------------------------------------------------------------

    plot_actual_vs_predicted(
        results
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()