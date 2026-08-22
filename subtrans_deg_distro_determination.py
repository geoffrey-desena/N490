#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate N490 sub-transmission degree-distribution parameters
using SIMPLE-GRAPH European comparison networks.

Model
-----
All cumulative node-degree distributions are fitted as:

    P(K >= k) = C * exp(-k / gamma)

with both C and gamma free.

Method
------
1. Load the European node-degree fit results.

2. Keep only SIMPLE-GRAPH European results.

3. Load Nordic490 directly and construct simple graphs for:
       220 kV
       300 kV
       380 kV

   Parallel lines between the same unordered pair of buses are
   collapsed to one edge before degree calculation.

4. Fit C and gamma for the three simple N490 voltage networks.

5. Match European transmission voltage layers to the N490 layers
   within +/-10%.

6. Ignore networks with fewer than MIN_NODES nodes.

7. Treat each fitted (C, gamma) pair as a point in parameter space.

8. Calculate one global covariance matrix from the eligible European
   transmission-layer simple-graph points.

9. Calculate Mahalanobis distance to the corresponding N490 point:

       d^2 = (x - x_N490)^T Sigma^-1 (x - x_N490)

10. Convert distance to a Gaussian similarity weight:

       w = exp(-d^2 / 2)

11. For each country, average squared Mahalanobis distance over its
    available matching transmission layers:

       w_country = exp(-mean(d^2) / 2)

12. Apply those country weights to the SIMPLE-GRAPH sub-transmission
    (<200 kV) C-gamma points.

13. Calculate:
       - unweighted sub-transmission centroid
       - weighted sub-transmission centroid
       - difference between them

Outputs
-------
euro-comparison/
    n490-subtransmission-estimate/
        simple-graph/
            voltage_comparison_weights.csv
            country_similarity_weights.csv
            weighted_subtransmission_points.csv
            subtransmission_estimate.csv
            subtransmission_estimate.pkl
            n490_simple_graph_fits.csv
            transmission_parameter_similarity.png
            subtransmission_center_of_mass.png
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, OptimizeWarning

from nordic490 import N490


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

OUTPUT_DIR = (
    EURO_DIR
    / "n490-subtransmission-estimate"
    / "simple-graph"
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
    """

    return (
        C
        * np.exp(
            -k / gamma
        )
    )


# =====================================================================
# N490 ENDPOINT DETECTION
# =====================================================================

def resolve_line_endpoint_columns(
    lines,
):
    """
    Identify the two endpoint columns in model.line.
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
            return (
                col0,
                col1,
            )

    raise ValueError(
        "Could not identify N490 line endpoint columns.\n"
        f"Available columns:\n"
        f"{lines.columns.tolist()}"
    )


# =====================================================================
# N490 SIMPLE GRAPH
# =====================================================================

def select_n490_voltage_lines(
    lines,
    voltage,
):
    """
    Select one N490 voltage layer.
    """

    if "Vbase" not in lines.columns:

        raise ValueError(
            "model.line does not contain Vbase."
        )

    voltage_values = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    selected = lines.loc[
        np.isclose(
            voltage_values,
            float(voltage),
            equal_nan=False,
        )
    ].copy()

    if selected.empty:

        raise ValueError(
            f"No N490 lines found at {voltage} kV."
        )

    return selected


def make_n490_simple_graph(
    lines,
):
    """
    Collapse multiple N490 lines connecting the same unordered
    pair of buses to one edge.
    """

    bus0_col, bus1_col = (
        resolve_line_endpoint_columns(
            lines
        )
    )

    edges = lines[
        [
            bus0_col,
            bus1_col,
        ]
    ].dropna().copy()

    # Convert endpoints to strings before sorting.
    #
    # This avoids relying on numeric bus IDs while still producing
    # a stable canonical unordered pair.
    endpoint_array = np.sort(
        edges[
            [
                bus0_col,
                bus1_col,
            ]
        ]
        .astype(str)
        .to_numpy(),
        axis=1,
    )

    edges["_node_i"] = (
        endpoint_array[:, 0]
    )

    edges["_node_j"] = (
        endpoint_array[:, 1]
    )

    simple_edges = (
        edges
        .drop_duplicates(
            subset=[
                "_node_i",
                "_node_j",
            ],
            keep="first",
        )
        .rename(
            columns={
                "_node_i": "node_i",
                "_node_j": "node_j",
            }
        )
        [
            [
                "node_i",
                "node_j",
            ]
        ]
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
    Calculate degree from a simple edge list.
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

def calculate_degree_distribution(
    degrees,
):
    """
    Calculate:

        P(K >= k)

    from k = 1 through maximum observed degree.
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

        P(K >= k) = C exp(-k / gamma)

    with free C and gamma.
    """

    if len(k) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    try:

        # Small networks can produce an OptimizeWarning because
        # curve_fit cannot estimate parameter covariance. We do not
        # use that covariance matrix here, so suppress only that
        # specific warning.
        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                OptimizeWarning,
            )

            popt, _ = curve_fit(
                exponential_degree_distribution,
                k,
                probability,
                p0=[
                    1.5,
                    2.0,
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
        )

    fitted = (
        exponential_degree_distribution(
            k,
            C,
            gamma,
        )
    )

    ss_res = np.sum(
        (
            probability
            - fitted
        ) ** 2
    )

    ss_tot = np.sum(
        (
            probability
            - np.mean(
                probability
            )
        ) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1
            - ss_res / ss_tot
        )

    else:

        r2 = np.nan

    return (
        C,
        gamma,
        float(r2),
    )


# =====================================================================
# BUILD N490 SIMPLE-GRAPH TARGETS
# =====================================================================

def calculate_n490_targets():
    """
    Load N490 and calculate simple-graph C-gamma fits for
    220, 300, and 380 kV.
    """

    model = N490(
        year=2018
    )

    lines = model.line.copy()

    targets = {}
    rows = []

    for label, voltage in (
        TARGET_VOLTAGES.items()
    ):

        complete_lines = (
            select_n490_voltage_lines(
                lines,
                voltage,
            )
        )

        simple_edges = (
            make_n490_simple_graph(
                complete_lines
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
        ) = (
            calculate_degree_distribution(
                degrees
            )
        )

        (
            C,
            gamma,
            r2,
        ) = (
            fit_exponential_distribution(
                k,
                probability,
            )
        )

        n_nodes = len(
            degrees
        )

        n_complete = len(
            complete_lines
        )

        n_simple = len(
            simple_edges
        )

        if n_nodes < MIN_NODES:

            raise ValueError(
                f"N490 {label} has fewer "
                f"than {MIN_NODES} nodes."
            )

        targets[label] = {
            "voltage":
                voltage,

            "C":
                C,

            "gamma":
                gamma,

            "n_nodes":
                n_nodes,

            "n_edges_complete":
                n_complete,

            "n_edges_simple":
                n_simple,

            "edges_removed":
                n_complete
                - n_simple,

            "r2":
                r2,
        }

        rows.append(
            {
                "network":
                    label,

                "voltage_kv":
                    voltage,

                "n_nodes":
                    n_nodes,

                "n_edges_complete":
                    n_complete,

                "n_edges_simple":
                    n_simple,

                "edges_removed":
                    n_complete
                    - n_simple,

                "C":
                    C,

                "gamma":
                    gamma,

                "R2":
                    r2,
            }
        )

    n490_table = pd.DataFrame(
        rows
    )

    return (
        targets,
        n490_table,
    )


# =====================================================================
# VOLTAGE LABEL PARSING
# =====================================================================

def parse_voltage_group(
    label,
):
    """
    Extract numerical voltage values from a European voltage label.

    Examples:
        220 kV      -> [220]
        132–165 kV  -> [132, 165]
        All         -> []
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


def representative_voltage(
    label,
):
    """
    Single voltage -> itself.
    Range -> midpoint.
    """

    values = (
        parse_voltage_group(
            label
        )
    )

    if len(values) == 0:
        return np.nan

    if len(values) == 1:
        return values[0]

    return float(
        np.mean(values)
    )


def is_subtransmission_group(
    label,
):
    """
    True if the voltage group contains only values below 200 kV.
    """

    values = (
        parse_voltage_group(
            label
        )
    )

    if len(values) == 0:
        return False

    return (
        max(values) < 200
    )


# =====================================================================
# LOAD EUROPEAN SIMPLE-GRAPH RESULTS
# =====================================================================

def load_european_results():
    """
    Load European results and retain only simple-graph rows.
    """

    if not EURO_RESULTS_FILE.exists():

        raise FileNotFoundError(
            "European result file not found:\n"
            f"{EURO_RESULTS_FILE}"
        )

    euro = pd.read_pickle(
        EURO_RESULTS_FILE
    )

    required = {
        "country",
        "graph_type",
        "voltage_group",
        "n_nodes",
        "C",
        "gamma",
        "r2",
    }

    missing = (
        required
        - set(
            euro.columns
        )
    )

    if missing:

        raise ValueError(
            "European results are missing columns:\n"
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------
    # IMPORTANT:
    # only SIMPLE-GRAPH fitted parameters are retained.
    # -------------------------------------------------------------

    euro = euro.loc[
        euro["graph_type"]
        .astype(str)
        .str.lower()
        == "simple"
    ].copy()

    if euro.empty:

        raise ValueError(
            "No simple-graph rows found in "
            "European result table."
        )

    euro[
        "representative_voltage"
    ] = (
        euro[
            "voltage_group"
        ]
        .apply(
            representative_voltage
        )
    )

    euro[
        "is_subtransmission"
    ] = (
        euro[
            "voltage_group"
        ]
        .apply(
            is_subtransmission_group
        )
    )

    return euro


# =====================================================================
# VOLTAGE MATCHING
# =====================================================================

def assign_voltage_band(
    voltage,
):
    """
    Match a European voltage layer to one N490 target within +/-10%.
    """

    if not np.isfinite(
        voltage
    ):
        return None

    candidates = []

    for label, target in (
        TARGET_VOLTAGES.items()
    ):

        relative_difference = (
            abs(
                voltage
                - target
            )
            / target
        )

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


# =====================================================================
# EUROPEAN TRANSMISSION COMPARISON POINTS
# =====================================================================

def build_comparison_points(
    euro,
):
    """
    Keep eligible European simple-graph transmission layers.
    """

    comparisons = euro.loc[
        euro["n_nodes"]
        >= MIN_NODES
    ].copy()

    # Remove combined sub-transmission rows.
    comparisons = comparisons.loc[
        ~comparisons[
            "is_subtransmission"
        ]
    ].copy()

    # Remove complete-country rows.
    comparisons = comparisons.loc[
        comparisons[
            "voltage_group"
        ]
        .astype(str)
        .str.lower()
        != "all"
    ].copy()

    comparisons[
        "target_band"
    ] = (
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
        ]
        .notna()
    ].copy()

    return comparisons


# =====================================================================
# GLOBAL COVARIANCE
# =====================================================================

def calculate_global_covariance(
    comparisons,
):
    """
    Calculate global covariance structure of simple-graph C-gamma
    comparison points.
    """

    values = (
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
        values,
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
            [
                "C",
                "gamma",
            ]
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
    Squared Mahalanobis distance.
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

    d2 = float(
        delta.T
        @ inverse_covariance
        @ delta
    )

    return max(
        d2,
        0.0,
    )


# =====================================================================
# POINT WEIGHTS
# =====================================================================

def calculate_point_weights(
    comparisons,
    n490_targets,
    inverse_covariance,
):
    """
    Calculate simple-graph European-to-N490 similarity weights.
    """

    rows = []

    for _, row in (
        comparisons.iterrows()
    ):

        target_label = row[
            "target_band"
        ]

        target = (
            n490_targets[
                target_label
            ]
        )

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

        d2 = (
            mahalanobis_distance_squared(
                point,
                target_point,
                inverse_covariance,
            )
        )

        distance = float(
            np.sqrt(
                d2
            )
        )

        weight = float(
            np.exp(
                -0.5
                * d2
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
                    int(
                        row["n_nodes"]
                    ),

                "C":
                    float(
                        row["C"]
                    ),

                "gamma":
                    float(
                        row["gamma"]
                    ),

                "R2":
                    float(
                        row["r2"]
                    ),

                "N490_C":
                    target["C"],

                "N490_gamma":
                    target["gamma"],

                "mahalanobis_distance":
                    distance,

                "mahalanobis_distance_squared":
                    d2,

                "point_weight":
                    weight,
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# COUNTRY WEIGHTS
# =====================================================================

def calculate_country_weights(
    comparison_weights,
):
    """
    Combine available transmission-layer matches into one weight
    per country.

    Similarity component:

        similarity_weight = exp(-mean(d_M^2) / 2)

    Evidence component:

        evidence_factor = sqrt(m / M)

    where:
        m = number of matching N490 voltage bands for the country
        M = total number of N490 target voltage bands

    Final weight:

        weight = similarity_weight * evidence_factor

    This prevents a country with only one matching voltage layer
    from receiving the same evidentiary weight as a country with
    comparable similarity across multiple voltage layers.
    """

    rows = []

    max_matching_layers = len(
        TARGET_VOLTAGES
    )

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

        # ---------------------------------------------------------
        # Similarity component
        # ---------------------------------------------------------

        similarity_weight = float(
            np.exp(
                -0.5
                * mean_d2
            )
        )

        # ---------------------------------------------------------
        # Evidence component
        #
        # Count distinct matched N490 voltage bands.
        #
        # For three target bands:
        #
        #   1 match -> sqrt(1/3) = 0.577
        #   2 match -> sqrt(2/3) = 0.816
        #   3 match -> 1.000
        # ---------------------------------------------------------

        n_matching_layers = len(
            group[
                "target_band"
            ].unique()
        )

        evidence_factor = float(
            np.sqrt(
                n_matching_layers
                / max_matching_layers
            )
        )

        # ---------------------------------------------------------
        # Combined weight
        # ---------------------------------------------------------

        weight = float(
            similarity_weight
            * evidence_factor
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
                    n_matching_layers,

                "matching_bands":
                    bands,

                "mean_mahalanobis_distance_squared":
                    mean_d2,

                "rms_mahalanobis_distance":
                    rms_distance,

                "similarity_weight":
                    similarity_weight,

                "evidence_factor":
                    evidence_factor,

                "weight":
                    weight,
            }
        )

    result = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------
    # Normalize final weights
    # -------------------------------------------------------------

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
# SIMPLE-GRAPH SUB-TRANSMISSION POINTS
# =====================================================================

def get_subtransmission_points(
    euro,
    country_weights,
):
    """
    Extract each country's SIMPLE-GRAPH combined <200 kV fit.
    """

    sub = euro.loc[
        (
            euro[
                "is_subtransmission"
            ]
        )
        &
        (
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

    duplicates = counts.loc[
        counts > 1
    ]

    if len(
        duplicates
    ) > 0:

        raise ValueError(
            "More than one simple-graph "
            "sub-transmission group found for:\n"
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

    # Re-normalize weights among countries actually contributing
    # sub-transmission points.
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
    Arithmetic mean of simple-graph sub-transmission points.
    """

    return (
        float(
            sub["C"].mean()
        ),
        float(
            sub["gamma"].mean()
        ),
    )


def calculate_weighted_centroid(
    sub,
):
    """
    Weighted center of mass.
    """

    weighted_C = float(
        np.sum(
            sub[
                "normalized_weight"
            ]
            * sub["C"]
        )
    )

    weighted_gamma = float(
        np.sum(
            sub[
                "normalized_weight"
            ]
            * sub["gamma"]
        )
    )

    return (
        weighted_C,
        weighted_gamma,
    )


# =====================================================================
# PLOT: TRANSMISSION PARAMETER SPACE
# =====================================================================

def plot_transmission_similarity(
    comparison_weights,
    n490_targets,
):
    """
    Plot eligible European simple-graph transmission fits.

    Circle size corresponds to point-specific similarity weight.
    N490 simple-graph points are stars.
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for band in (
        TARGET_VOLTAGES
    ):

        group = (
            comparison_weights.loc[
                comparison_weights[
                    "target_band"
                ]
                == band
            ]
        )

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
                f"European {band}"
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
                xytext=(
                    4,
                    4,
                ),
                textcoords=(
                    "offset points"
                ),
                fontsize=7,
                alpha=0.75,
            )

    # -------------------------------------------------------------
    # N490 simple-graph stars
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
            label=(
                f"N490 {band}"
            ),
            zorder=10,
        )

    ax.set_xlabel(
        "Exponential coefficient $C$"
    )

    ax.set_ylabel(
        r"Decay parameter $\gamma$"
    )

    ax.set_title(
        "Simple-graph transmission similarity "
        "in $C$–$\\gamma$ parameter space"
    )

    ax.grid(
        False
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

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
    plt.close(
        fig
    )

    print(
        f"Saved:\n  {output_path}"
    )


# =====================================================================
# PLOT: SUB-TRANSMISSION CENTER OF MASS
# =====================================================================

def plot_subtransmission_center_of_mass(
    sub,
    unweighted_C,
    unweighted_gamma,
    weighted_C,
    weighted_gamma,
):
    """
    Plot simple-graph European sub-transmission fits and both
    centroid estimates.
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
        label=(
            "Country sub-transmission "
            "simple-graph fits"
        ),
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
            xytext=(
                5,
                5,
            ),
            textcoords=(
                "offset points"
            ),
            fontsize=8,
        )

    # Unweighted
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

    # Weighted
    ax.scatter(
        weighted_C,
        weighted_gamma,
        marker="*",
        s=350,
        color="black",
        label=(
            "Weighted N490 estimate"
        ),
        zorder=10,
    )

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
        "Simple-graph sub-transmission "
        "$C$–$\\gamma$ centroid comparison"
    )

    ax.grid(
        False
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

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
    plt.close(
        fig
    )

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
    Print simple-graph N490 reference parameters.
    """

    print("\n")
    print("=" * 105)
    print(
        "N490 SIMPLE-GRAPH REFERENCE PARAMETERS"
    )
    print("=" * 105)

    print(
        f"{'Network':<12}"
        f"{'N':>8}"
        f"{'E original':>12}"
        f"{'E simple':>12}"
        f"{'Removed':>10}"
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
            f"{values['n_edges_complete']:>12d}"
            f"{values['n_edges_simple']:>12d}"
            f"{values['edges_removed']:>10d}"
            f"{values['C']:>12.4f}"
            f"{values['gamma']:>12.4f}"
            f"{values['r2']:>12.4f}"
        )


def print_country_weights(
    country_weights,
):
    """
    Print final country weights.
    """

    print("\n")
    print("=" * 115)
    print(
        "SIMPLE-GRAPH COUNTRY SIMILARITY WEIGHTS"
    )
    print("=" * 115)

    display = country_weights[
        [
            "country",
            "n_matching_layers",
            "matching_bands",
            "rms_mahalanobis_distance",
            "similarity_weight",
            "evidence_factor",
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
    # Load European SIMPLE-GRAPH results
    # -----------------------------------------------------------------

    euro = (
        load_european_results()
    )

    print("\n")
    print("=" * 100)
    print(
        "EUROPEAN INPUT"
    )
    print("=" * 100)

    print(
        "Using ONLY graph_type == 'simple'."
    )

    print(
        f"Rows loaded: {len(euro)}"
    )

    # -----------------------------------------------------------------
    # Calculate N490 SIMPLE-GRAPH fits
    # -----------------------------------------------------------------

    (
        n490_targets,
        n490_table,
    ) = (
        calculate_n490_targets()
    )

    print_n490_targets(
        n490_targets
    )

    n490_table.to_csv(
        OUTPUT_DIR
        / "n490_simple_graph_fits.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # European transmission comparison set
    # -----------------------------------------------------------------

    comparisons = (
        build_comparison_points(
            euro
        )
    )

    print("\n")
    print("=" * 110)
    print(
        "ELIGIBLE EUROPEAN SIMPLE-GRAPH COMPARISON POINTS"
    )
    print("=" * 110)

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
    # Global covariance
    # -----------------------------------------------------------------

    (
        covariance_matrix,
        inverse_covariance,
        correlation,
        condition_number,
    ) = (
        calculate_global_covariance(
            comparisons
        )
    )

    print("\n")
    print("=" * 100)
    print(
        "GLOBAL SIMPLE-GRAPH C-GAMMA COVARIANCE STRUCTURE"
    )
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
    # Mahalanobis point weights
    # -----------------------------------------------------------------

    comparison_weights = (
        calculate_point_weights(
            comparisons=
                comparisons,

            n490_targets=
                n490_targets,

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
    # Simple-graph sub-transmission points
    # -----------------------------------------------------------------

    sub = (
        get_subtransmission_points(
            euro=
                euro,

            country_weights=
                country_weights,
        )
    )

    # -----------------------------------------------------------------
    # Centroids
    # -----------------------------------------------------------------

    (
        unweighted_C,
        unweighted_gamma,
    ) = (
        calculate_unweighted_centroid(
            sub
        )
    )

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
        100
        * delta_C
        / unweighted_C
    )

    percent_shift_gamma = (
        100
        * delta_gamma
        / unweighted_gamma
    )

    print("\n")
    print("=" * 100)
    print(
        "SIMPLE-GRAPH SUB-TRANSMISSION CENTROID COMPARISON"
    )
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
                "graph_type":
                    "simple",

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
    # Figures
    # -----------------------------------------------------------------

    plot_transmission_similarity(
        comparison_weights=
            comparison_weights,

        n490_targets=
            n490_targets,
    )

    plot_subtransmission_center_of_mass(
        sub=
            sub,

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
    # Outputs
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        OUTPUT_DIR
        / "n490_simple_graph_fits.csv"
    )

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