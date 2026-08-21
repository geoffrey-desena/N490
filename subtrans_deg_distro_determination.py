#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate N490 sub-transmission exponential-fit parameters
=========================================================

Use fitted cumulative node-degree parameters from 15 European
networks to estimate plausible sub-transmission parameters for N490.

The cumulative degree model is:

    P(K >= k) = C * exp(-k / gamma)

with both C and gamma free.

Method
------
1. Load N490 fits for:
       220 kV
       300 kV
       380 kV

2. Load European country fits.

3. For each N490 voltage level, identify European voltage layers
   within +/-10%:

       ~220 kV
       ~300 kV
       ~380 kV

4. Ignore any network with fewer than MIN_NODES nodes.

5. Treat every (C, gamma) pair as a point in parameter space.

6. Calculate a GLOBAL covariance matrix from all eligible European
   transmission-layer comparison points.

7. Measure similarity to the appropriate N490 voltage-layer point
   using Mahalanobis distance:

       d^2 = (x - x_N490)^T Sigma^-1 (x - x_N490)

   where:

       x = [C, gamma]

   This explicitly accounts for correlation between C and gamma.

8. Convert Mahalanobis distance to a Gaussian similarity weight:

       w = exp(-d^2 / 2)

9. For each country, average squared Mahalanobis distance across
   all N490 voltage bands for which it has a suitable comparison
   layer:

       w_country = exp(-mean(d^2) / 2)

10. Use those country weights to calculate a weighted center of mass
    of the countries' combined sub-transmission (<200 kV) C-gamma
    points.

11. Also calculate the ordinary unweighted centroid of the same
    sub-transmission points for comparison.

Outputs
-------
Console:
    - N490 reference parameters
    - covariance/correlation diagnostics
    - eligible European comparison points
    - country similarity weights
    - unweighted sub-transmission centroid
    - weighted sub-transmission centroid
    - difference between the two

Files:
    euro-comparison/
        n490-subtransmission-estimate/
            voltage_comparison_weights.csv
            country_similarity_weights.csv
            weighted_subtransmission_points.csv
            subtransmission_estimate.csv
            subtransmission_estimate.pkl
            transmission_parameter_similarity.png
            subtransmission_center_of_mass.png
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

WORKING_DIR = Path.cwd()

EURO_DIR = (
    WORKING_DIR
    / "euro-comparison"
)

EURO_RESULTS_FILE = (
    EURO_DIR
    / "node-degree-analysis"
    / "node_degree_fit_summary.pkl"
)

N490_RESULTS_FILE = (
    WORKING_DIR
    / "n490_degree_decay_analysis"
    / "N490_degree_decay_fit_summary.pkl"
)

OUTPUT_DIR = (
    EURO_DIR
    / "n490-subtransmission-estimate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MIN_NODES = 10

VOLTAGE_TOLERANCE = 0.10


TARGET_VOLTAGES = {
    "220 kV": 220.0,
    "300 kV": 300.0,
    "380 kV": 380.0,
}


VOLTAGE_COLORS = {
    "220 kV": "green",
    "300 kV": "gold",
    "380 kV": "red",
}


FIGSIZE = (9.0, 7.0)
DPI = 300


# =====================================================================
# VOLTAGE LABEL PARSING
# =====================================================================

def parse_voltage_group(label):
    """
    Extract numerical voltage values from a voltage-group label.

    Examples
    --------
    '220 kV'
        -> [220.0]

    '132–165 kV'
        -> [132.0, 165.0]

    'All'
        -> []
    """

    if pd.isna(label):
        return []

    label = str(label)

    if label.lower() == "all":
        return []

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        label,
    )

    return [
        float(value)
        for value in numbers
    ]


def representative_voltage(label):
    """
    Return a representative voltage for a voltage-group label.

    A single voltage is returned directly.

    For a range, return the midpoint.
    """

    values = parse_voltage_group(
        label
    )

    if len(values) == 0:
        return np.nan

    if len(values) == 1:
        return values[0]

    return float(
        np.mean(values)
    )


def is_subtransmission_group(label):
    """
    Return True if all voltages represented by the label are <200 kV.
    """

    values = parse_voltage_group(
        label
    )

    if len(values) == 0:
        return False

    return max(values) < 200.0


# =====================================================================
# LOAD DATA
# =====================================================================

def load_results():
    """
    Load European and N490 fitted-parameter results.
    """

    if not EURO_RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"European results not found:\n"
            f"{EURO_RESULTS_FILE}"
        )

    if not N490_RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"N490 results not found:\n"
            f"{N490_RESULTS_FILE}"
        )

    euro = pd.read_pickle(
        EURO_RESULTS_FILE
    )

    n490 = pd.read_pickle(
        N490_RESULTS_FILE
    )

    return euro, n490


# =====================================================================
# STANDARDIZE INPUT TABLES
# =====================================================================

def standardize_euro_results(
    euro,
):
    """
    Check expected European result columns and add helper columns.
    """

    required = {
        "country",
        "voltage_group",
        "n_nodes",
        "C",
        "gamma",
        "r2",
    }

    missing = (
        required
        - set(euro.columns)
    )

    if missing:

        raise ValueError(
            "European results are missing columns:\n"
            f"{sorted(missing)}\n\n"
            f"Available columns:\n"
            f"{euro.columns.tolist()}"
        )

    euro = euro.copy()

    euro["representative_voltage"] = (
        euro["voltage_group"]
        .apply(
            representative_voltage
        )
    )

    euro["is_subtransmission"] = (
        euro["voltage_group"]
        .apply(
            is_subtransmission_group
        )
    )

    return euro


def standardize_n490_results(
    n490,
):
    """
    Check expected N490 result columns.
    """

    required = {
        "network",
        "n_nodes",
        "C",
        "gamma",
        "R2",
    }

    missing = (
        required
        - set(n490.columns)
    )

    if missing:

        raise ValueError(
            "N490 results are missing columns:\n"
            f"{sorted(missing)}\n\n"
            f"Available columns:\n"
            f"{n490.columns.tolist()}"
        )

    return n490.copy()


# =====================================================================
# N490 REFERENCE POINTS
# =====================================================================

def get_n490_targets(
    n490,
):
    """
    Extract the 220, 300, and 380 kV N490 reference fits.
    """

    targets = {}

    for label, voltage in (
        TARGET_VOLTAGES.items()
    ):

        matches = n490.loc[
            n490["network"] == label
        ]

        if len(matches) != 1:

            raise ValueError(
                f"Expected exactly one N490 "
                f"result for {label}; "
                f"found {len(matches)}."
            )

        row = matches.iloc[0]

        if int(row["n_nodes"]) < MIN_NODES:

            raise ValueError(
                f"N490 {label} has fewer "
                f"than {MIN_NODES} nodes."
            )

        targets[label] = {
            "voltage": voltage,
            "C": float(row["C"]),
            "gamma": float(
                row["gamma"]
            ),
            "n_nodes": int(
                row["n_nodes"]
            ),
            "r2": float(
                row["R2"]
            ),
        }

    return targets


# =====================================================================
# BUILD EUROPEAN COMPARISON SET
# =====================================================================

def assign_voltage_band(
    voltage,
):
    """
    Assign a European voltage layer to an N490 target voltage if
    it lies within +/- VOLTAGE_TOLERANCE.

    If multiple bands qualify, use the closest relative difference.
    """

    if not np.isfinite(voltage):
        return None

    candidates = []

    for label, target in (
        TARGET_VOLTAGES.items()
    ):

        relative_difference = abs(
            voltage - target
        ) / target

        if (
            relative_difference
            <= VOLTAGE_TOLERANCE
        ):

            candidates.append(
                (
                    relative_difference,
                    label,
                )
            )

    if not candidates:
        return None

    candidates.sort()

    return candidates[0][1]


def build_comparison_points(
    euro,
):
    """
    Build European transmission-layer comparison points.

    Excludes:
        - complete-network rows
        - sub-transmission rows
        - N < MIN_NODES
        - voltages outside +/-10% of an N490 target
    """

    comparisons = euro.loc[
        euro["n_nodes"] >= MIN_NODES
    ].copy()

    comparisons = comparisons.loc[
        ~comparisons[
            "is_subtransmission"
        ]
    ].copy()

    comparisons = comparisons.loc[
        comparisons[
            "voltage_group"
        ].astype(str).str.lower()
        != "all"
    ].copy()

    comparisons["target_band"] = (
        comparisons[
            "representative_voltage"
        ]
        .apply(
            assign_voltage_band
        )
    )

    comparisons = comparisons.loc[
        comparisons[
            "target_band"
        ].notna()
    ].copy()

    return comparisons


# =====================================================================
# GLOBAL COVARIANCE STRUCTURE
# =====================================================================

def calculate_global_covariance(
    comparisons,
):
    """
    Calculate the global covariance matrix of C and gamma.

    Returns
    -------
    covariance_matrix
        2x2 covariance matrix.

    inverse_covariance
        Pseudoinverse of covariance matrix.

        np.linalg.pinv is used rather than inv because strong
        C-gamma correlation can make the covariance matrix close
        to singular.

    correlation
        Pearson correlation between C and gamma.
    """

    parameter_values = (
        comparisons[
            [
                "C",
                "gamma",
            ]
        ]
        .astype(float)
        .to_numpy()
    )

    covariance_matrix = np.cov(
        parameter_values,
        rowvar=False,
        ddof=1,
    )

    inverse_covariance = (
        np.linalg.pinv(
            covariance_matrix
        )
    )

    correlation = float(
        comparisons[
            ["C", "gamma"]
        ]
        .corr()
        .loc[
            "C",
            "gamma",
        ]
    )

    condition_number = float(
        np.linalg.cond(
            covariance_matrix
        )
    )

    return (
        covariance_matrix,
        inverse_covariance,
        correlation,
        condition_number,
    )


# =====================================================================
# MAHALANOBIS DISTANCE
# =====================================================================

def mahalanobis_distance_squared(
    point,
    target,
    inverse_covariance,
):
    """
    Calculate squared Mahalanobis distance:

        d^2 = delta.T @ Sigma^-1 @ delta
    """

    delta = (
        np.asarray(
            point,
            dtype=float,
        )
        - np.asarray(
            target,
            dtype=float,
        )
    )

    distance_squared = float(
        delta.T
        @ inverse_covariance
        @ delta
    )

    # Numerical noise can occasionally produce a tiny negative value.
    return max(
        distance_squared,
        0.0,
    )


# =====================================================================
# VOLTAGE-LEVEL WEIGHTS
# =====================================================================

def calculate_point_weights(
    comparisons,
    n490_targets,
    inverse_covariance,
):
    """
    Calculate Mahalanobis distance from each European transmission
    point to the corresponding N490 target.

    Gaussian point weight:

        w = exp(-d^2 / 2)
    """

    rows = []

    for _, row in (
        comparisons.iterrows()
    ):

        target_label = row[
            "target_band"
        ]

        target = n490_targets[
            target_label
        ]

        point = np.array(
            [
                float(row["C"]),
                float(row["gamma"]),
            ]
        )

        target_point = np.array(
            [
                target["C"],
                target["gamma"],
            ]
        )

        distance_squared = (
            mahalanobis_distance_squared(
                point=point,
                target=target_point,
                inverse_covariance=
                    inverse_covariance,
            )
        )

        distance = float(
            np.sqrt(
                distance_squared
            )
        )

        weight = float(
            np.exp(
                -0.5
                * distance_squared
            )
        )

        rows.append(
            {
                "country":
                    row["country"],

                "voltage_group":
                    row[
                        "voltage_group"
                    ],

                "representative_voltage":
                    row[
                        "representative_voltage"
                    ],

                "target_band":
                    target_label,

                "n_nodes":
                    int(row["n_nodes"]),

                "C":
                    float(row["C"]),

                "gamma":
                    float(
                        row["gamma"]
                    ),

                "R2":
                    float(row["r2"]),

                "N490_C":
                    target["C"],

                "N490_gamma":
                    target["gamma"],

                "mahalanobis_distance":
                    distance,

                "mahalanobis_distance_squared":
                    distance_squared,

                "point_weight":
                    weight,
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# COUNTRY-LEVEL SIMILARITY
# =====================================================================

def calculate_country_weights(
    comparison_weights,
):
    """
    Combine voltage-layer similarities into one country similarity.

    For each country:

        mean_d2 = mean(d_M^2)

        weight = exp(-mean_d2 / 2)

    Averaging squared distance prevents countries with more available
    comparison layers from being automatically penalized.
    """

    rows = []

    for country, group in (
        comparison_weights.groupby(
            "country"
        )
    ):

        mean_d2 = float(
            group[
                "mahalanobis_distance_squared"
            ].mean()
        )

        rms_distance = float(
            np.sqrt(
                mean_d2
            )
        )

        weight = float(
            np.exp(
                -0.5
                * mean_d2
            )
        )

        bands = ", ".join(
            sorted(
                group[
                    "target_band"
                ].unique()
            )
        )

        rows.append(
            {
                "country":
                    country,

                "n_matching_layers":
                    len(group),

                "matching_bands":
                    bands,

                "mean_mahalanobis_distance_squared":
                    mean_d2,

                "rms_mahalanobis_distance":
                    rms_distance,

                "weight":
                    weight,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "normalized_weight"
    ] = (
        result["weight"]
        / result["weight"].sum()
    )

    return (
        result
        .sort_values(
            "weight",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =====================================================================
# SUB-TRANSMISSION POINTS
# =====================================================================

def get_subtransmission_points(
    euro,
    country_weights,
):
    """
    Extract each country's combined <200 kV parameter point and
    attach its country similarity weight.
    """

    sub = euro.loc[
        (
            euro[
                "is_subtransmission"
            ]
        )
        & (
            euro["n_nodes"]
            >= MIN_NODES
        )
    ].copy()

    counts = (
        sub.groupby(
            "country"
        )
        .size()
    )

    duplicates = counts[
        counts > 1
    ]

    if len(duplicates) > 0:

        raise ValueError(
            "More than one sub-transmission "
            "group found for:\n"
            f"{duplicates}"
        )

    sub = sub.merge(
        country_weights[
            [
                "country",
                "weight",
            ]
        ],
        on="country",
        how="inner",
    )

    # Re-normalize after restricting to countries with valid
    # sub-transmission fits.
    sub[
        "normalized_weight"
    ] = (
        sub["weight"]
        / sub["weight"].sum()
    )

    return sub


# =====================================================================
# CENTROIDS
# =====================================================================

def calculate_unweighted_centroid(
    sub,
):
    """
    Ordinary arithmetic mean of all eligible sub-transmission points.
    """

    centroid_C = float(
        sub["C"].mean()
    )

    centroid_gamma = float(
        sub["gamma"].mean()
    )

    return (
        centroid_C,
        centroid_gamma,
    )


def calculate_weighted_centroid(
    sub,
):
    """
    Weighted C-gamma center of mass.
    """

    centroid_C = float(
        np.sum(
            sub[
                "normalized_weight"
            ]
            * sub["C"]
        )
    )

    centroid_gamma = float(
        np.sum(
            sub[
                "normalized_weight"
            ]
            * sub["gamma"]
        )
    )

    return (
        centroid_C,
        centroid_gamma,
    )


# =====================================================================
# PLOT 1: TRANSMISSION PARAMETER SIMILARITY
# =====================================================================

def plot_transmission_similarity(
    comparison_weights,
    n490_targets,
):
    """
    Plot eligible European transmission-layer parameter points.

    Circle size corresponds to point-specific Gaussian weight.

    N490 reference points are stars.
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for band in TARGET_VOLTAGES:

        group = comparison_weights.loc[
            comparison_weights[
                "target_band"
            ]
            == band
        ]

        if group.empty:
            continue

        sizes = (
            40
            + 500
            * group[
                "point_weight"
            ]
        )

        ax.scatter(
            group["C"],
            group["gamma"],
            s=sizes,
            color=VOLTAGE_COLORS[
                band
            ],
            alpha=0.50,
            edgecolors="black",
            linewidths=0.6,
            label=(
                f"European "
                f"{band}"
            ),
        )

        for _, row in (
            group.iterrows()
        ):

            ax.annotate(
                row["country"],
                (
                    row["C"],
                    row["gamma"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.75,
            )

    # -------------------------------------------------------------
    # N490 reference stars
    # -------------------------------------------------------------

    for band, target in (
        n490_targets.items()
    ):

        ax.scatter(
            target["C"],
            target["gamma"],
            s=280,
            marker="*",
            color=VOLTAGE_COLORS[
                band
            ],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.95,
            label=f"N490 {band}",
            zorder=10,
        )

    ax.set_xlabel(
        "Exponential coefficient $C$"
    )

    ax.set_ylabel(
        r"Decay parameter $\gamma$"
    )

    ax.set_title(
        "Transmission-network similarity "
        "in $C$–$\\gamma$ parameter space"
    )

    ax.grid(
        False
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "transmission_parameter_similarity.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved:\n  {output_path}"
    )


# =====================================================================
# PLOT 2: SUB-TRANSMISSION CENTER OF MASS
# =====================================================================

def plot_subtransmission_center_of_mass(
    sub,
    unweighted_C,
    unweighted_gamma,
    weighted_C,
    weighted_gamma,
):
    """
    Plot country sub-transmission parameter points.

    Circle size corresponds to country similarity weight.

    Show both:
        - unweighted centroid
        - Mahalanobis-weighted centroid
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    sizes = (
        50
        + 650
        * sub[
            "normalized_weight"
        ]
        / sub[
            "normalized_weight"
        ].max()
    )

    ax.scatter(
        sub["C"],
        sub["gamma"],
        s=sizes,
        alpha=0.45,
        edgecolors="black",
        linewidths=0.7,
        label="Country sub-transmission fits",
    )

    for _, row in (
        sub.iterrows()
    ):

        ax.annotate(
            row["country"],
            (
                row["C"],
                row["gamma"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # -------------------------------------------------------------
    # Unweighted centroid
    # -------------------------------------------------------------

    ax.scatter(
        unweighted_C,
        unweighted_gamma,
        marker="X",
        s=220,
        color="gray",
        edgecolors="black",
        linewidths=1.0,
        label="Unweighted centroid",
        zorder=9,
    )

    # -------------------------------------------------------------
    # Weighted centroid
    # -------------------------------------------------------------

    ax.scatter(
        weighted_C,
        weighted_gamma,
        marker="*",
        s=350,
        color="black",
        label="Weighted N490 estimate",
        zorder=10,
    )

    # -------------------------------------------------------------
    # Connect the two estimates
    # -------------------------------------------------------------

    ax.plot(
        [
            unweighted_C,
            weighted_C,
        ],
        [
            unweighted_gamma,
            weighted_gamma,
        ],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.set_xlabel(
        "Exponential coefficient $C$"
    )

    ax.set_ylabel(
        r"Decay parameter $\gamma$"
    )

    ax.set_title(
        "Sub-transmission $C$–$\\gamma$ "
        "centroid comparison"
    )

    ax.grid(
        False
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "subtransmission_center_of_mass.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved:\n  {output_path}"
    )


# =====================================================================
# PRINTING
# =====================================================================

def print_n490_targets(
    targets,
):
    """
    Print N490 reference parameters.
    """

    print("\n")
    print("=" * 90)
    print("N490 REFERENCE PARAMETERS")
    print("=" * 90)

    print(
        f"{'Network':<12}"
        f"{'N':>8}"
        f"{'C':>12}"
        f"{'gamma':>12}"
        f"{'R2':>12}"
    )

    for label, values in (
        targets.items()
    ):

        print(
            f"{label:<12}"
            f"{values['n_nodes']:>8d}"
            f"{values['C']:>12.4f}"
            f"{values['gamma']:>12.4f}"
            f"{values['r2']:>12.4f}"
        )


def print_country_weights(
    country_weights,
):
    """
    Print final country similarity weights.
    """

    print("\n")
    print("=" * 115)
    print("COUNTRY SIMILARITY WEIGHTS")
    print("=" * 115)

    display = country_weights[
        [
            "country",
            "n_matching_layers",
            "matching_bands",
            "rms_mahalanobis_distance",
            "weight",
            "normalized_weight",
        ]
    ].copy()

    print(
        display
        .round(4)
        .to_string(
            index=False
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load results
    # -----------------------------------------------------------------

    euro, n490 = (
        load_results()
    )

    euro = (
        standardize_euro_results(
            euro
        )
    )

    n490 = (
        standardize_n490_results(
            n490
        )
    )

    # -----------------------------------------------------------------
    # N490 targets
    # -----------------------------------------------------------------

    n490_targets = (
        get_n490_targets(
            n490
        )
    )

    print_n490_targets(
        n490_targets
    )

    # -----------------------------------------------------------------
    # European comparison points
    # -----------------------------------------------------------------

    comparisons = (
        build_comparison_points(
            euro
        )
    )

    print("\n")
    print("=" * 100)
    print("ELIGIBLE EUROPEAN COMPARISON POINTS")
    print("=" * 100)

    print(
        comparisons[
            [
                "country",
                "voltage_group",
                "representative_voltage",
                "n_nodes",
                "C",
                "gamma",
                "r2",
                "target_band",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Global covariance structure
    # -----------------------------------------------------------------

    (
        covariance_matrix,
        inverse_covariance,
        correlation,
        condition_number,
    ) = calculate_global_covariance(
        comparisons
    )

    print("\n")
    print("=" * 100)
    print("GLOBAL C-GAMMA COVARIANCE STRUCTURE")
    print("=" * 100)

    covariance_df = pd.DataFrame(
        covariance_matrix,
        index=[
            "C",
            "gamma",
        ],
        columns=[
            "C",
            "gamma",
        ],
    )

    print(
        "\nCovariance matrix:"
    )

    print(
        covariance_df
        .round(6)
        .to_string()
    )

    print(
        f"\nPearson correlation "
        f"(C, gamma) = "
        f"{correlation:.6f}"
    )

    print(
        f"Covariance matrix "
        f"condition number = "
        f"{condition_number:.4f}"
    )

    # -----------------------------------------------------------------
    # Point-specific Mahalanobis weights
    # -----------------------------------------------------------------

    comparison_weights = (
        calculate_point_weights(
            comparisons=comparisons,
            n490_targets=n490_targets,
            inverse_covariance=
                inverse_covariance,
        )
    )

    # -----------------------------------------------------------------
    # Country weights
    # -----------------------------------------------------------------

    country_weights = (
        calculate_country_weights(
            comparison_weights
        )
    )

    print_country_weights(
        country_weights
    )

    # -----------------------------------------------------------------
    # Sub-transmission points
    # -----------------------------------------------------------------

    sub = (
        get_subtransmission_points(
            euro=euro,
            country_weights=
                country_weights,
        )
    )

    # -----------------------------------------------------------------
    # Unweighted centroid
    # -----------------------------------------------------------------

    (
        unweighted_C,
        unweighted_gamma,
    ) = (
        calculate_unweighted_centroid(
            sub
        )
    )

    # -----------------------------------------------------------------
    # Weighted centroid
    # -----------------------------------------------------------------

    (
        weighted_C,
        weighted_gamma,
    ) = (
        calculate_weighted_centroid(
            sub
        )
    )

    # -----------------------------------------------------------------
    # Difference
    # -----------------------------------------------------------------

    delta_C = (
        weighted_C
        - unweighted_C
    )

    delta_gamma = (
        weighted_gamma
        - unweighted_gamma
    )

    euclidean_shift = float(
        np.sqrt(
            delta_C ** 2
            + delta_gamma ** 2
        )
    )

    percent_shift_C = (
        100.0
        * delta_C
        / unweighted_C
    )

    percent_shift_gamma = (
        100.0
        * delta_gamma
        / unweighted_gamma
    )

    print("\n")
    print("=" * 100)
    print("SUB-TRANSMISSION CENTROID COMPARISON")
    print("=" * 100)

    print(
        "\nUnweighted centroid:"
    )

    print(
        f"  C     = "
        f"{unweighted_C:.6f}"
    )

    print(
        f"  gamma = "
        f"{unweighted_gamma:.6f}"
    )

    print(
        "\nMahalanobis-weighted centroid:"
    )

    print(
        f"  C     = "
        f"{weighted_C:.6f}"
    )

    print(
        f"  gamma = "
        f"{weighted_gamma:.6f}"
    )

    print(
        "\nWeighted - unweighted:"
    )

    print(
        f"  delta C     = "
        f"{delta_C:+.6f} "
        f"({percent_shift_C:+.2f} %)"
    )

    print(
        f"  delta gamma = "
        f"{delta_gamma:+.6f} "
        f"({percent_shift_gamma:+.2f} %)"
    )

    print(
        f"  Euclidean parameter-space shift = "
        f"{euclidean_shift:.6f}"
    )

    print(
        f"\nCountries contributing: "
        f"{len(sub)}"
    )

    # -----------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------

    comparison_weights.to_csv(
        OUTPUT_DIR
        / "voltage_comparison_weights.csv",
        index=False,
    )

    country_weights.to_csv(
        OUTPUT_DIR
        / "country_similarity_weights.csv",
        index=False,
    )

    sub_output = sub[
        [
            "country",
            "voltage_group",
            "n_nodes",
            "C",
            "gamma",
            "r2",
            "weight",
            "normalized_weight",
        ]
    ].copy()

    sub_output.to_csv(
        OUTPUT_DIR
        / "weighted_subtransmission_points.csv",
        index=False,
    )

    estimate = pd.DataFrame(
        [
            {
                "unweighted_C":
                    unweighted_C,

                "unweighted_gamma":
                    unweighted_gamma,

                "weighted_C":
                    weighted_C,

                "weighted_gamma":
                    weighted_gamma,

                "delta_C":
                    delta_C,

                "delta_gamma":
                    delta_gamma,

                "percent_shift_C":
                    percent_shift_C,

                "percent_shift_gamma":
                    percent_shift_gamma,

                "euclidean_shift":
                    euclidean_shift,

                "C_gamma_correlation":
                    correlation,

                "covariance_condition_number":
                    condition_number,

                "voltage_tolerance":
                    VOLTAGE_TOLERANCE,

                "minimum_nodes":
                    MIN_NODES,

                "n_countries":
                    len(sub),
            }
        ]
    )

    estimate.to_csv(
        OUTPUT_DIR
        / "subtransmission_estimate.csv",
        index=False,
    )

    estimate.to_pickle(
        OUTPUT_DIR
        / "subtransmission_estimate.pkl"
    )

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    plot_transmission_similarity(
        comparison_weights=
            comparison_weights,
        n490_targets=
            n490_targets,
    )

    plot_subtransmission_center_of_mass(
        sub=sub,
        unweighted_C=
            unweighted_C,
        unweighted_gamma=
            unweighted_gamma,
        weighted_C=
            weighted_C,
        weighted_gamma=
            weighted_gamma,
    )

    # -----------------------------------------------------------------
    # Final output paths
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        OUTPUT_DIR
        / "voltage_comparison_weights.csv"
    )

    print(
        OUTPUT_DIR
        / "country_similarity_weights.csv"
    )

    print(
        OUTPUT_DIR
        / "weighted_subtransmission_points.csv"
    )

    print(
        OUTPUT_DIR
        / "subtransmission_estimate.csv"
    )

    print(
        OUTPUT_DIR
        / "subtransmission_estimate.pkl"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()