#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N490 A-gamma bivariate reference-ellipse analysis
==================================================

Purpose
-------
Read the model-parameter table produced by:

    n490_europe_node_degree_parameters.py

and assess whether the N490 anchored-exponential node-degree parameters are
reasonable relative to the European comparison networks.

For each graph representation and voltage class:

    <200 kV
    200-299 kV
    300-349 kV
    >=350 kV

this script uses ONLY the European A-gamma observations to estimate:

    - the bivariate mean (centroid),
    - the sample covariance matrix,
    - the A-gamma correlation,
    - a 95% bivariate normal reference ellipse.

The N490 point does NOT contribute to the centroid, covariance, or ellipse.
Its squared Mahalanobis distance from the European centroid is then evaluated
against the chi-square distribution with 2 degrees of freedom.

Terminology
-----------
The plotted regions are called "95% reference ellipses" rather than
"95% confidence ellipses". They describe the dispersion of individual
European network parameter pairs under a bivariate-normal reference model;
they are NOT confidence regions for the European mean.

Reference ellipse
-----------------
For x = [A, gamma]^T, European mean mu, and sample covariance Sigma:

    D^2 = (x - mu)^T Sigma^(-1) (x - mu)

The 95% reference ellipse is:

    D^2 <= chi2.ppf(0.95, df=2)

which is approximately 5.991.

The percentile reported for N490 is:

    100 * chi2.cdf(D_N490^2, df=2)

and can be interpreted as the fraction of the fitted bivariate-normal
reference distribution expected to lie closer to the European centroid than
the N490 point.

Input
-----
    euro-comparison/node-degree-parameter-comparison/
        euro_n490_node_degree_parameters.pkl

Outputs
-------
Nothing is saved.

The script:
    - prints diagnostics and statistical summary tables,
    - displays one A-gamma figure for complete networks,
    - displays one A-gamma figure for simple graphs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


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

# =====================================================================
# DEGREE-DISTRIBUTION OUTPUT
# =====================================================================

OUTPUT_DIR = (
    WORKING_DIR
    / "euro-comparison"
    / "node-degree-parameter-comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Continue calculating explicit degree probabilities until the
# probability remaining above the largest listed degree is smaller
# than this value.
PMF_TAIL_TOLERANCE = 1e-6


# =====================================================================
# SETTINGS
# =====================================================================

CONFIDENCE_LEVEL = 0.95
CHI2_DF = 2

FIGSIZE = (9.0, 7.0)

EURO_ALPHA = 0.35
EURO_MARKER_SIZE = 65

N490_ALPHA = 1.00
N490_MARKER_SIZE = 190

CENTROID_MARKER_SIZE = 95

ELLIPSE_LINEWIDTH = 2.0
ELLIPSE_FILL_ALPHA = 0.07

# At least three two-dimensional observations are needed before a covariance
# ellipse is useful. A full-rank covariance matrix is also required.
MIN_EUROPEAN_POINTS = 3


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

GRAPH_TYPE_LABELS = {
    "complete": "Complete networks",
    "simple": "Simple graphs",
}


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def load_parameter_table():
    """Load and validate the parameter table from the previous script."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nCould not find the saved parameter table:\n"
            f"  {INPUT_FILE}\n\n"
            "Run n490_europe_node_degree_parameters.py first."
        )

    parameter_df = pd.read_pickle(INPUT_FILE)

    required_columns = {
        "source",
        "system",
        "graph_type",
        "voltage",
        "voltage_label",
        "voltage_class",
        "A",
        "gamma",
    }

    missing = required_columns - set(parameter_df.columns)

    if missing:
        raise KeyError(
            "The parameter pickle does not have the expected schema.\n"
            f"Missing columns: {sorted(missing)}"
        )

    return parameter_df.copy()


# =====================================================================
# BIVARIATE STATISTICS
# =====================================================================

def calculate_reference_statistics(european_group):
    """
    Estimate the bivariate center/covariance of one European voltage class.

    Returns None if there are too few observations or if the covariance matrix
    is singular / not positive definite.
    """

    valid = (
        european_group
        .dropna(subset=["A", "gamma"])
        .copy()
    )

    n = len(valid)

    if n < MIN_EUROPEAN_POINTS:
        return None

    values = valid[["A", "gamma"]].to_numpy(dtype=float)

    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)

    if covariance.shape != (2, 2):
        return None

    if not np.all(np.isfinite(covariance)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # A valid 2-D ellipse requires positive variance along both principal axes.
    if np.any(eigenvalues <= 0.0):
        return None

    covariance_inverse = np.linalg.inv(covariance)

    std_A = float(np.sqrt(covariance[0, 0]))
    std_gamma = float(np.sqrt(covariance[1, 1]))

    if std_A > 0.0 and std_gamma > 0.0:
        correlation = float(
            covariance[0, 1]
            / (std_A * std_gamma)
        )
    else:
        correlation = np.nan

    return {
        "n": n,
        "mean": mean,
        "covariance": covariance,
        "covariance_inverse": covariance_inverse,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "correlation": correlation,
    }


def mahalanobis_squared(point, mean, covariance_inverse):
    """Squared Mahalanobis distance of one [A, gamma] point."""

    difference = np.asarray(point, dtype=float) - np.asarray(mean, dtype=float)

    return float(
        difference.T
        @ covariance_inverse
        @ difference
    )


# =====================================================================
# ANALYSIS TABLES
# =====================================================================

def build_statistical_summary(parameter_df):
    """
    Build one row per graph type and voltage class.

    European observations define the reference distribution. N490 is then
    evaluated against it.
    """

    chi2_threshold = float(
        chi2.ppf(
            CONFIDENCE_LEVEL,
            df=CHI2_DF,
        )
    )

    summary_rows = []
    stats_lookup = {}

    graph_types = [
        graph_type
        for graph_type in ["complete", "simple"]
        if graph_type in set(parameter_df["graph_type"])
    ]

    for graph_type in graph_types:

        graph_df = parameter_df.loc[
            parameter_df["graph_type"] == graph_type
        ].copy()

        for voltage_class in VOLTAGE_CLASS_ORDER:

            class_df = graph_df.loc[
                graph_df["voltage_class"] == voltage_class
            ].copy()

            european = class_df.loc[
                class_df["source"] == "Europe"
            ].copy()

            n490 = class_df.loc[
                class_df["source"] == "N490"
            ].copy()

            reference_stats = calculate_reference_statistics(european)

            lookup_key = (graph_type, voltage_class)
            stats_lookup[lookup_key] = reference_stats

            base_row = {
                "graph_type": graph_type,
                "voltage_class": voltage_class,
                "european_n": int(
                    european[["A", "gamma"]]
                    .dropna()
                    .shape[0]
                ),
                "ellipse_available": reference_stats is not None,
                "chi2_95_threshold": chi2_threshold,
            }

            if reference_stats is None:
                base_row.update(
                    {
                        "mean_A": np.nan,
                        "mean_gamma": np.nan,
                        "sd_A": np.nan,
                        "sd_gamma": np.nan,
                        "cov_A_gamma": np.nan,
                        "correlation": np.nan,
                    }
                )
            else:
                covariance = reference_stats["covariance"]

                base_row.update(
                    {
                        "mean_A": float(reference_stats["mean"][0]),
                        "mean_gamma": float(reference_stats["mean"][1]),
                        "sd_A": float(np.sqrt(covariance[0, 0])),
                        "sd_gamma": float(np.sqrt(covariance[1, 1])),
                        "cov_A_gamma": float(covariance[0, 1]),
                        "correlation": float(reference_stats["correlation"]),
                    }
                )

            if n490.empty:
                row = base_row.copy()
                row.update(
                    {
                        "n490_voltage": np.nan,
                        "n490_A": np.nan,
                        "n490_gamma": np.nan,
                        "mahalanobis_D2": np.nan,
                        "reference_percentile": np.nan,
                        "inside_95_reference_ellipse": np.nan,
                    }
                )
                summary_rows.append(row)
                continue

            # There should normally be exactly one N490 observation in each
            # voltage class. Loop rather than silently discarding extras.
            for _, n490_row in n490.iterrows():

                row = base_row.copy()

                row.update(
                    {
                        "n490_voltage": float(n490_row["voltage"]),
                        "n490_A": float(n490_row["A"]),
                        "n490_gamma": float(n490_row["gamma"]),
                    }
                )

                if (
                    reference_stats is None
                    or pd.isna(n490_row["A"])
                    or pd.isna(n490_row["gamma"])
                ):
                    D2 = np.nan
                    percentile = np.nan
                    inside = np.nan
                else:
                    D2 = mahalanobis_squared(
                        point=[n490_row["A"], n490_row["gamma"]],
                        mean=reference_stats["mean"],
                        covariance_inverse=reference_stats[
                            "covariance_inverse"
                        ],
                    )

                    percentile = float(
                        100.0
                        * chi2.cdf(
                            D2,
                            df=CHI2_DF,
                        )
                    )

                    inside = bool(D2 <= chi2_threshold)

                row.update(
                    {
                        "mahalanobis_D2": D2,
                        "reference_percentile": percentile,
                        "inside_95_reference_ellipse": inside,
                    }
                )

                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, stats_lookup

# =====================================================================
# DEGREE PROBABILITIES FROM MEAN A-GAMMA PARAMETERS
# =====================================================================

def anchored_exponential_ccdf(
    k,
    A,
    gamma,
):
    """
    Anchored exponential complementary cumulative distribution:

        P(K >= 1) = 1

        P(K >= k) =
            A * exp(-(k - 2) / gamma),   k >= 2

    Thus A = P(K >= 2).
    """

    k = np.asarray(
        k,
        dtype=float,
    )

    probability = np.ones_like(
        k,
        dtype=float,
    )

    mask = (
        k >= 2
    )

    probability[mask] = (
        A
        * np.exp(
            -(k[mask] - 2.0)
            / gamma
        )
    )

    return probability


def degree_probability_mass(
    k,
    A,
    gamma,
):
    """
    Calculate P(K = k) from the anchored exponential CCDF.

    Since:

        P(K = k)
        =
        P(K >= k) - P(K >= k + 1)

    we obtain:

        P(K = 1) = 1 - A

    and for k >= 2:

        P(K = k)
        =
        A * exp(-(k - 2) / gamma)
          * (1 - exp(-1 / gamma))
    """

    k = np.asarray(
        k,
        dtype=float,
    )

    ccdf_k = anchored_exponential_ccdf(
        k,
        A,
        gamma,
    )

    ccdf_next = anchored_exponential_ccdf(
        k + 1.0,
        A,
        gamma,
    )

    return (
        ccdf_k
        - ccdf_next
    )


def build_mean_parameter_degree_distributions(
    summary_df,
):
    """
    Convert the mean European A-gamma parameters for every graph type
    and voltage class into ordinary degree probability distributions.

    The explicit distribution is extended until the remaining
    probability above the largest listed degree is less than
    PMF_TAIL_TOLERANCE.

    Returns
    -------
    pandas.DataFrame

        One row per graph type, voltage class, and degree k.
    """

    # Mean European parameters are repeated if a voltage class has more
    # than one N490 comparison row, so retain only one copy of each
    # European reference distribution.
    parameter_means = (
        summary_df[
            [
                "graph_type",
                "voltage_class",
                "european_n",
                "mean_A",
                "mean_gamma",
            ]
        ]
        .drop_duplicates(
            subset=[
                "graph_type",
                "voltage_class",
            ]
        )
        .dropna(
            subset=[
                "mean_A",
                "mean_gamma",
            ]
        )
        .copy()
    )

    rows = []

    for _, row in parameter_means.iterrows():

        A = float(
            row["mean_A"]
        )

        gamma = float(
            row["mean_gamma"]
        )

        # ---------------------------------------------------------
        # Determine a sufficiently large maximum degree.
        #
        # We require:
        #
        #   P(K >= k_max + 1) < PMF_TAIL_TOLERANCE
        #
        # so that essentially all model probability mass appears
        # explicitly in the saved table.
        # ---------------------------------------------------------

        k_max = 2

        while (
            anchored_exponential_ccdf(
                np.array(
                    [k_max + 1],
                    dtype=float,
                ),
                A,
                gamma,
            )[0]
            >= PMF_TAIL_TOLERANCE
        ):
            k_max += 1

        k_values = np.arange(
            1,
            k_max + 1,
            dtype=int,
        )

        ccdf = anchored_exponential_ccdf(
            k_values,
            A,
            gamma,
        )

        probability_mass = degree_probability_mass(
            k_values,
            A,
            gamma,
        )

        remaining_tail = float(
            anchored_exponential_ccdf(
                np.array(
                    [k_max + 1],
                    dtype=float,
                ),
                A,
                gamma,
            )[0]
        )

        for (
            degree,
            cumulative_probability,
            probability,
        ) in zip(
            k_values,
            ccdf,
            probability_mass,
        ):

            rows.append(
                {
                    "graph_type":
                        row["graph_type"],

                    "voltage_class":
                        row["voltage_class"],

                    "european_n":
                        int(row["european_n"]),

                    "mean_A":
                        A,

                    "mean_gamma":
                        gamma,

                    "degree_k":
                        int(degree),

                    "ccdf_P_K_ge_k":
                        float(
                            cumulative_probability
                        ),

                    "probability_P_K_eq_k":
                        float(
                            probability
                        ),

                    "percentage_P_K_eq_k":
                        float(
                            100.0
                            * probability
                        ),

                    "remaining_probability_above_max_k":
                        (
                            remaining_tail
                            if degree == k_max
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# ELLIPSE GEOMETRY
# =====================================================================

def ellipse_geometry(reference_stats, probability=CONFIDENCE_LEVEL):
    """
    Convert covariance eigenstructure to matplotlib Ellipse dimensions.

    For a bivariate-normal probability contour:

        axis semi-length_i = sqrt(chi2_quantile * eigenvalue_i)

    matplotlib expects full width and height, so each semi-length is doubled.
    """

    threshold = float(
        chi2.ppf(
            probability,
            df=CHI2_DF,
        )
    )

    eigenvalues = reference_stats["eigenvalues"]
    eigenvectors = reference_stats["eigenvectors"]

    # eigh returns ascending eigenvalues. Use the largest principal axis first.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    semi_axes = np.sqrt(threshold * eigenvalues)

    width = 2.0 * float(semi_axes[0])
    height = 2.0 * float(semi_axes[1])

    major_axis_vector = eigenvectors[:, 0]

    angle_degrees = float(
        np.degrees(
            np.arctan2(
                major_axis_vector[1],
                major_axis_vector[0],
            )
        )
    )

    return width, height, angle_degrees


# =====================================================================
# PLOTTING
# =====================================================================

def plot_reference_ellipses(parameter_df, stats_lookup, graph_type):
    """
    Reproduce the A-gamma parameter-space plot with European 95% reference
    ellipses and emphasized N490 observations.
    """

    graph_df = parameter_df.loc[
        parameter_df["graph_type"] == graph_type
    ].dropna(subset=["A", "gamma"]).copy()

    if graph_df.empty:
        print(
            f"\nNo valid data available for graph_type={graph_type!r}."
        )
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for voltage_class in VOLTAGE_CLASS_ORDER:

        color = VOLTAGE_COLOR_MAP[voltage_class]

        group = graph_df.loc[
            graph_df["voltage_class"] == voltage_class
        ].copy()

        european = group.loc[
            group["source"] == "Europe"
        ].copy()

        n490 = group.loc[
            group["source"] == "N490"
        ].copy()

        # ---------------------------------------------------------
        # European observations
        # ---------------------------------------------------------

        if not european.empty:
            ax.scatter(
                european["A"],
                european["gamma"],
                s=EURO_MARKER_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=EURO_ALPHA,
                label=f"{voltage_class} Europe",
                zorder=3,
            )

        # ---------------------------------------------------------
        # 95% European reference ellipse + centroid
        # ---------------------------------------------------------

        reference_stats = stats_lookup.get(
            (graph_type, voltage_class)
        )

        if reference_stats is not None:

            width, height, angle = ellipse_geometry(reference_stats)
            mean_A, mean_gamma = reference_stats["mean"]

            ellipse = Ellipse(
                xy=(mean_A, mean_gamma),
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                linewidth=ELLIPSE_LINEWIDTH,
                alpha=ELLIPSE_FILL_ALPHA,
                zorder=1,
            )

            ax.add_patch(ellipse)

            # Add an opaque boundary on top because the patch alpha also
            # affects its edge.
            ellipse_outline = Ellipse(
                xy=(mean_A, mean_gamma),
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor=color,
                linewidth=ELLIPSE_LINEWIDTH,
                zorder=2,
            )

            ax.add_patch(ellipse_outline)

            ax.scatter(
                [mean_A],
                [mean_gamma],
                s=CENTROID_MARKER_SIZE,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
            )

        # ---------------------------------------------------------
        # N490 observation
        # ---------------------------------------------------------

        if not n490.empty:
            ax.scatter(
                n490["A"],
                n490["gamma"],
                s=N490_MARKER_SIZE,
                marker="*",
                color=color,
                edgecolor="black",
                linewidth=1.0,
                alpha=N490_ALPHA,
                label=f"{voltage_class} N490",
                zorder=6,
            )

            for _, row in n490.iterrows():
                ax.annotate(
                    f"N490 {row['voltage']:g} kV",
                    xy=(row["A"], row["gamma"]),
                    xytext=(7, 6),
                    textcoords="offset points",
                    fontsize=9,
                    zorder=7,
                )

    ax.set_xlabel(r"$A$", fontsize=13)
    ax.set_ylabel(r"$\gamma$", fontsize=13)

    graph_label = GRAPH_TYPE_LABELS.get(
        graph_type,
        graph_type,
    )

    ax.set_title(
        f"{graph_label}: European 95% reference ellipses and N490"
    )

    ax.grid(alpha=0.20)

    # The scatter labels provide an explicit Europe/N490 distinction while the
    # color simultaneously identifies the voltage class.
    handles, labels = ax.get_legend_handles_labels()

    # Keep only the first occurrence of each label.
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        fontsize=8.5,
        ncol=2,
    )

    fig.tight_layout()
    plt.show()


# =====================================================================
# CONSOLE REPORTING
# =====================================================================

def print_reference_distribution_summary(summary_df):
    """Print European reference-distribution statistics."""

    print("\n")
    print("=" * 122)
    print("EUROPEAN BIVARIATE REFERENCE DISTRIBUTIONS")
    print("=" * 122)
    print(
        f"Reference probability: {100 * CONFIDENCE_LEVEL:.1f}%\n"
        f"Chi-square threshold (df={CHI2_DF}): "
        f"{chi2.ppf(CONFIDENCE_LEVEL, df=CHI2_DF):.4f}"
    )

    columns = [
        "graph_type",
        "voltage_class",
        "european_n",
        "mean_A",
        "mean_gamma",
        "sd_A",
        "sd_gamma",
        "correlation",
        "ellipse_available",
    ]

    display = (
        summary_df[columns]
        .drop_duplicates(
            subset=["graph_type", "voltage_class"]
        )
        .copy()
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_A": "{:.4f}".format,
                "mean_gamma": "{:.4f}".format,
                "sd_A": "{:.4f}".format,
                "sd_gamma": "{:.4f}".format,
                "correlation": "{:.4f}".format,
            },
        )
    )


def print_n490_comparison(summary_df):
    """Print N490 location relative to each European reference ellipse."""

    print("\n")
    print("=" * 122)
    print("N490 POSITION WITHIN EUROPEAN A-GAMMA REFERENCE DISTRIBUTIONS")
    print("=" * 122)

    columns = [
        "graph_type",
        "voltage_class",
        "n490_voltage",
        "european_n",
        "n490_A",
        "n490_gamma",
        "mahalanobis_D2",
        "reference_percentile",
        "inside_95_reference_ellipse",
    ]

    print(
        summary_df[columns].to_string(
            index=False,
            formatters={
                "n490_voltage": "{:.0f}".format,
                "n490_A": "{:.4f}".format,
                "n490_gamma": "{:.4f}".format,
                "mahalanobis_D2": "{:.4f}".format,
                "reference_percentile": "{:.1f}".format,
            },
        )
    )

    print(
        "\nInterpretation:\n"
        "  - D^2 is the squared Mahalanobis distance from the European centroid.\n"
        "  - The reference percentile is 100 * chi2.cdf(D^2, df=2).\n"
        "  - A point inside the 95% reference ellipse has D^2 <= "
        f"{chi2.ppf(CONFIDENCE_LEVEL, df=CHI2_DF):.4f}.\n"
        "  - The ellipses describe between-network dispersion, not uncertainty "
        "in the fitted European mean."
    )


def print_covariance_matrices(parameter_df, stats_lookup):
    """Print each European covariance matrix for diagnostic inspection."""

    print("\n")
    print("=" * 122)
    print("EUROPEAN A-GAMMA SAMPLE COVARIANCE MATRICES")
    print("=" * 122)

    graph_types = [
        graph_type
        for graph_type in ["complete", "simple"]
        if graph_type in set(parameter_df["graph_type"])
    ]

    for graph_type in graph_types:
        for voltage_class in VOLTAGE_CLASS_ORDER:

            stats = stats_lookup.get((graph_type, voltage_class))

            print(
                f"\n{graph_type.upper()} | {voltage_class}"
            )

            if stats is None:
                print(
                    "  Reference ellipse unavailable "
                    "(too few observations or singular covariance)."
                )
                continue

            covariance = stats["covariance"]

            print(
                f"  n = {stats['n']}\n"
                f"  corr(A, gamma) = {stats['correlation']:.4f}\n"
                "  covariance =\n"
                f"    [[{covariance[0, 0]: .8f}, {covariance[0, 1]: .8f}],\n"
                f"     [{covariance[1, 0]: .8f}, {covariance[1, 1]: .8f}]]"
            )

def print_mean_degree_probability_distributions(
    distribution_df,
):
    """
    Print degree-probability quotas derived from the mean European
    A-gamma parameters.
    """

    print("\n")
    print("=" * 122)
    print(
        "DEGREE PROBABILITY DISTRIBUTIONS FROM "
        "MEAN EUROPEAN A-GAMMA PARAMETERS"
    )
    print("=" * 122)

    for graph_type in [
        "complete",
        "simple",
    ]:

        graph_df = distribution_df.loc[
            distribution_df["graph_type"]
            == graph_type
        ]

        if graph_df.empty:
            continue

        print(
            f"\n{graph_type.upper()}"
        )

        # Wide table is easier to interpret as a set of quotas:
        #
        #              k=1    k=2    k=3 ...
        # <200 kV
        # 200-299 kV
        # ...
        wide = (
            graph_df
            .pivot(
                index="voltage_class",
                columns="degree_k",
                values="probability_P_K_eq_k",
            )
            .reindex(
                VOLTAGE_CLASS_ORDER
            )
        )

        wide.columns = [
            f"k={int(k)}"
            for k in wide.columns
        ]

        print(
            wide.to_string(
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
                na_rep="",
            )
        )

# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 122)
    print("N490 A-GAMMA BIVARIATE REFERENCE-ELLIPSE ANALYSIS")
    print("=" * 122)
    print(f"Input:\n  {INPUT_FILE}")

    parameter_df = load_parameter_table()

    print(
        f"\nRows loaded: {len(parameter_df)}\n"
        f"European rows: "
        f"{int((parameter_df['source'] == 'Europe').sum())}\n"
        f"N490 rows: "
        f"{int((parameter_df['source'] == 'N490').sum())}"
    )

    summary_df, stats_lookup = build_statistical_summary(parameter_df)
    
    # -------------------------------------------------------------
    # Degree probability distributions implied by mean A-gamma
    # -------------------------------------------------------------

    degree_distribution_df = (
        build_mean_parameter_degree_distributions(
            summary_df
        )
    )

    print_mean_degree_probability_distributions(
        degree_distribution_df
    )

    degree_distribution_csv = (
        OUTPUT_DIR
        / "mean_A_gamma_degree_probability_distributions.csv"
    )

    degree_distribution_pickle = (
        OUTPUT_DIR
        / "mean_A_gamma_degree_probability_distributions.pkl"
    )

    degree_distribution_df.to_csv(
        degree_distribution_csv,
        index=False,
    )

    degree_distribution_df.to_pickle(
        degree_distribution_pickle
    )

    print(
        "\nSaved mean-parameter degree distributions:"
        f"\n  {degree_distribution_csv}"
        f"\n  {degree_distribution_pickle}"
    )

    print_reference_distribution_summary(summary_df)
    print_covariance_matrices(parameter_df, stats_lookup)
    print_n490_comparison(summary_df)

    plot_reference_ellipses(
        parameter_df,
        stats_lookup,
        graph_type="complete",
    )

    plot_reference_ellipses(
        parameter_df,
        stats_lookup,
        graph_type="simple",
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
