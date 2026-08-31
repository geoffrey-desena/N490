#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two exponential models for node-degree CCDFs
====================================================

Loads:
    euro-comparison/european_networks.pkl

For each complete country network:

    1. Calculate node degrees, retaining parallel circuits.
    2. Calculate the empirical complementary cumulative distribution:

           P(K >= k)

    3. Fit two alternative exponential models:

       MODEL 1: one-parameter CCDF
       --------------------------
           P(K >= k) = exp(-k / gamma)

       fitted over:
           k >= 1

       gamma is the only fitted parameter.


       MODEL 2: exponential anchored at k = 2
       --------------------------------------
           P(K >= k) = A * exp(-(k - 2) / gamma)

       fitted over:
           k >= 2

       A and gamma are both fitted parameters.

       Here A is the fitted value of the CCDF at k = 2.


    4. Compare goodness of fit.

       Because the two models are fitted over different native domains,
       the main side-by-side comparison is evaluated on the COMMON DOMAIN:

           k >= 2

       Metrics reported on the common domain:
           - R^2
           - adjusted R^2
           - RMSE
           - MAE
           - AIC
           - AICc

       The script also saves native-domain metrics for reference.

    5. Plot, for each country:
           - empirical CCDF
           - fitted one-parameter CCDF
           - fitted k=2 anchored exponential

Parallel circuits are retained in the node-degree calculation.
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

DATA_DIR = WORKING_DIR / "euro-comparison"
INPUT_FILE = DATA_DIR / "european_networks.pkl"

OUTPUT_DIR = DATA_DIR / "hartmann-investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "model-comparison-plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "ccdf_model_comparison_summary.csv"
)

FULL_RESULTS_FILE = (
    OUTPUT_DIR
    / "ccdf_model_comparison_full.csv"
)


# =====================================================================
# PLOT SETTINGS
# =====================================================================

FIGSIZE = (7.2, 5.2)
DPI = 300
TEXT_SIZE = 12


# =====================================================================
# MODELS
# =====================================================================

def model_ccdf(k, gamma):
    """
    Model 1:

        P(K >= k) = exp(-k / gamma)

    gamma is the only fitted parameter.
    """

    return np.exp(
        -k / gamma
    )


def model_anchored_k2(k, A, gamma):
    """
    Model 2:

        P(K >= k)
        =
        A * exp(-(k - 2) / gamma)

    for k >= 2.

    A is the fitted CCDF value at k = 2.
    """

    return (
        A
        * np.exp(
            -(k - 2.0) / gamma
        )
    )


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_node_degrees(edges):
    """
    Calculate node degree directly from the edge list.

    Every row is treated as one physical branch, so parallel circuits
    are retained and contribute separately to node degree.
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

def calculate_cumulative_distribution(degrees):
    """
    Calculate:

        P(K >= k)

    for every integer degree from 1 through max degree.
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
# GOODNESS-OF-FIT METRICS
# =====================================================================

def goodness_of_fit(
    observed,
    fitted,
    n_parameters,
):
    """
    Calculate goodness-of-fit metrics.

    Returns:
        R^2
        adjusted R^2
        RMSE
        MAE
        AIC
        AICc

    AIC is calculated under the usual Gaussian residual assumption:

        AIC = n * ln(RSS / n) + 2p

    where p is the number of fitted model parameters.

    AICc is the small-sample corrected AIC.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    fitted = np.asarray(
        fitted,
        dtype=float,
    )

    valid = (
        np.isfinite(observed)
        & np.isfinite(fitted)
    )

    observed = observed[valid]
    fitted = fitted[valid]

    n = len(observed)
    p = int(
        n_parameters
    )

    if n == 0:
        return {
            "n_points": 0,
            "r2": np.nan,
            "adjusted_r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "aic": np.nan,
            "aicc": np.nan,
        }

    residuals = (
        observed
        - fitted
    )

    rss = np.sum(
        residuals ** 2
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    mae = np.mean(
        np.abs(
            residuals
        )
    )

    ss_tot = np.sum(
        (
            observed
            - np.mean(observed)
        ) ** 2
    )

    if ss_tot > 0:
        r2 = (
            1.0
            - rss / ss_tot
        )
    else:
        r2 = np.nan

    if (
        np.isfinite(r2)
        and n > p + 1
    ):
        adjusted_r2 = (
            1.0
            - (1.0 - r2)
            * (n - 1.0)
            / (n - p - 1.0)
        )
    else:
        adjusted_r2 = np.nan

    # Guard against log(0) for a numerically perfect fit.
    rss_for_aic = max(
        float(rss),
        np.finfo(float).tiny,
    )

    aic = (
        n
        * np.log(
            rss_for_aic / n
        )
        + 2.0 * p
    )

    if n > p + 1:
        aicc = (
            aic
            + (
                2.0
                * p
                * (p + 1.0)
                / (n - p - 1.0)
            )
        )
    else:
        aicc = np.nan

    return {
        "n_points": n,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "rmse": rmse,
        "mae": mae,
        "aic": aic,
        "aicc": aicc,
    }


# =====================================================================
# FIT MODEL 1
# =====================================================================

def fit_model_ccdf(
    k,
    probability,
):
    """
    Fit:

        P(K >= k) = exp(-k / gamma)

    over k >= 1.

    gamma is the only fitted parameter.
    """

    valid = (
        np.isfinite(k)
        & np.isfinite(probability)
        & (k >= 1)
    )

    k_fit = k[valid]
    p_fit = probability[valid]

    if len(k_fit) < 2:
        return (
            np.nan,
            k_fit,
            p_fit,
        )

    try:

        popt, _ = curve_fit(
            model_ccdf,
            k_fit,
            p_fit,
            p0=[
                2.0,
            ],
            bounds=(
                [1e-8],
                [np.inf],
            ),
            maxfev=100000,
        )

        gamma = float(
            popt[0]
        )

    except (
        RuntimeError,
        ValueError,
    ):

        gamma = np.nan

    return (
        gamma,
        k_fit,
        p_fit,
    )


# =====================================================================
# FIT MODEL 2
# =====================================================================

def fit_model_anchored_k2(
    k,
    probability,
):
    """
    Fit:

        P(K >= k)
        =
        A * exp(-(k - 2) / gamma)

    over k >= 2.

    A and gamma are both fitted.
    """

    valid = (
        np.isfinite(k)
        & np.isfinite(probability)
        & (k >= 2)
    )

    k_fit = k[valid]
    p_fit = probability[valid]

    if len(k_fit) < 2:
        return (
            np.nan,
            np.nan,
            k_fit,
            p_fit,
        )

    # The empirical probability at k=2 is a natural initial value for A.
    A_initial = float(
        p_fit[0]
    )

    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            model_anchored_k2,
            k_fit,
            p_fit,
            p0=[
                A_initial,
                gamma_initial,
            ],
            bounds=(
                [0.0, 1e-8],
                [1.0, np.inf],
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

        A = np.nan
        gamma = np.nan

    return (
        A,
        gamma,
        k_fit,
        p_fit,
    )


# =====================================================================
# PLOTTING
# =====================================================================

def plot_country_comparison(
    country,
    k,
    probability,
    gamma_ccdf,
    A_anchored,
    gamma_anchored,
):
    """
    Plot the empirical CCDF and both fitted exponential models.
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    # -------------------------------------------------------------
    # Empirical distribution
    # -------------------------------------------------------------

    ax.scatter(
        k,
        probability,
        s=48,
        label="Empirical CCDF",
        zorder=4,
    )

    # -------------------------------------------------------------
    # Model 1: exp(-k/gamma)
    # -------------------------------------------------------------

    k_model1 = np.linspace(
        1.0,
        float(k.max()),
        500,
    )

    if np.isfinite(
        gamma_ccdf
    ):

        p_model1 = (
            model_ccdf(
                k_model1,
                gamma_ccdf,
            )
        )

        ax.plot(
            k_model1,
            p_model1,
            linewidth=2.0,
            label=(
                r"$e^{-k/\gamma}$"
                f", $\\gamma={gamma_ccdf:.3f}$"
            ),
        )

    # -------------------------------------------------------------
    # Model 2: A exp(-(k-2)/gamma), k >= 2
    # -------------------------------------------------------------

    if (
        np.isfinite(
            A_anchored
        )
        and np.isfinite(
            gamma_anchored
        )
        and k.max() >= 2
    ):

        k_model2 = np.linspace(
            2.0,
            float(k.max()),
            500,
        )

        p_model2 = (
            model_anchored_k2(
                k_model2,
                A_anchored,
                gamma_anchored,
            )
        )

        ax.plot(
            k_model2,
            p_model2,
            linewidth=2.0,
            linestyle="--",
            label=(
                r"$A e^{-(k-2)/\gamma}$"
                f", $A={A_anchored:.3f}$"
                f", $\\gamma={gamma_anchored:.3f}$"
            ),
        )

    # -------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Node degree, $k$",
        fontsize=TEXT_SIZE,
    )

    ax.set_ylabel(
        r"$P(K \geq k)$",
        fontsize=TEXT_SIZE,
    )

    ax.set_title(
        country,
        fontsize=TEXT_SIZE + 1,
    )

    ax.tick_params(
        axis="both",
        labelsize=TEXT_SIZE - 1,
    )

    ax.set_xlim(
        0.8,
        float(k.max()) + 0.2,
    )

    ax.set_ylim(
        0.0,
        1.05,
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
        frameon=False,
        fontsize=TEXT_SIZE - 2,
    )

    fig.tight_layout()

    safe_country = (
        country
        .replace(
            "&",
            "and",
        )
        .replace(
            " ",
            "_",
        )
        .replace(
            "/",
            "_",
        )
    )

    output_file = (
        PLOT_DIR
        / (
            f"{safe_country}"
            "_ccdf_model_comparison.png"
        )
    )

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_file


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 150)
    print("NODE-DEGREE CCDF MODEL COMPARISON")
    print("=" * 150)

    print("\nLoading:")
    print(
        f"  {INPUT_FILE}"
    )

    print("\n")
    print("=" * 150)
    print("METHOD")
    print("=" * 150)

    print(
        """
For each country, node degree is calculated from the complete edge list,
so parallel circuits are retained.

The empirical complementary cumulative distribution is:

    P(K >= k)

Two exponential models are compared.

Model 1:
    P(K >= k) = exp(-k / gamma)

    fitted over k >= 1
    fitted parameters: gamma

Model 2:
    P(K >= k) = A * exp(-(k - 2) / gamma)

    fitted over k >= 2
    fitted parameters: A and gamma

Both fits use nonlinear least squares in ordinary probability space.

Because the models have different native fitting domains, the primary
goodness-of-fit comparison is calculated over the common domain k >= 2.
This permits direct comparison of R2, adjusted R2, RMSE, MAE, AIC, and AICc.
"""
    )

    euro_networks = pd.read_pickle(
        INPUT_FILE
    )

    full_results = []
    summary_rows = []

    # -----------------------------------------------------------------
    # Analyze countries
    # -----------------------------------------------------------------

    for country in sorted(
        euro_networks
    ):

        edges = (
            euro_networks[
                country
            ]
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
            calculate_cumulative_distribution(
                degrees
            )
        )

        # -------------------------------------------------------------
        # Fit Model 1
        # -------------------------------------------------------------

        (
            gamma_ccdf,
            k_ccdf,
            p_ccdf,
        ) = fit_model_ccdf(
            k,
            probability,
        )

        fitted_ccdf_native = (
            model_ccdf(
                k_ccdf,
                gamma_ccdf,
            )
            if np.isfinite(
                gamma_ccdf
            )
            else np.full_like(
                p_ccdf,
                np.nan,
            )
        )

        metrics_ccdf_native = (
            goodness_of_fit(
                p_ccdf,
                fitted_ccdf_native,
                n_parameters=1,
            )
        )

        # -------------------------------------------------------------
        # Fit Model 2
        # -------------------------------------------------------------

        (
            A_anchored,
            gamma_anchored,
            k_anchored,
            p_anchored,
        ) = (
            fit_model_anchored_k2(
                k,
                probability,
            )
        )

        fitted_anchored_native = (
            model_anchored_k2(
                k_anchored,
                A_anchored,
                gamma_anchored,
            )
            if (
                np.isfinite(
                    A_anchored
                )
                and np.isfinite(
                    gamma_anchored
                )
            )
            else np.full_like(
                p_anchored,
                np.nan,
            )
        )

        metrics_anchored_native = (
            goodness_of_fit(
                p_anchored,
                fitted_anchored_native,
                n_parameters=2,
            )
        )

        # -------------------------------------------------------------
        # Common-domain comparison: k >= 2
        # -------------------------------------------------------------

        common_mask = (
            k >= 2
        )

        k_common = (
            k[
                common_mask
            ]
        )

        p_common = (
            probability[
                common_mask
            ]
        )

        fitted_ccdf_common = (
            model_ccdf(
                k_common,
                gamma_ccdf,
            )
            if np.isfinite(
                gamma_ccdf
            )
            else np.full_like(
                p_common,
                np.nan,
            )
        )

        fitted_anchored_common = (
            model_anchored_k2(
                k_common,
                A_anchored,
                gamma_anchored,
            )
            if (
                np.isfinite(
                    A_anchored
                )
                and np.isfinite(
                    gamma_anchored
                )
            )
            else np.full_like(
                p_common,
                np.nan,
            )
        )

        metrics_ccdf_common = (
            goodness_of_fit(
                p_common,
                fitted_ccdf_common,
                n_parameters=1,
            )
        )

        metrics_anchored_common = (
            goodness_of_fit(
                p_common,
                fitted_anchored_common,
                n_parameters=2,
            )
        )

        # -------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------

        plot_file = (
            plot_country_comparison(
                country=country,
                k=k,
                probability=probability,
                gamma_ccdf=gamma_ccdf,
                A_anchored=A_anchored,
                gamma_anchored=gamma_anchored,
            )
        )

        # -------------------------------------------------------------
        # Full results
        # -------------------------------------------------------------

        full_row = {
            "country": country,
            "n_nodes": len(
                degrees
            ),
            "n_branches": len(
                edges
            ),
            "mean_degree": (
                degrees.mean()
            ),

            "ccdf_gamma": (
                gamma_ccdf
            ),

            "anchored_A": (
                A_anchored
            ),

            "anchored_gamma": (
                gamma_anchored
            ),

            "plot_file": str(
                plot_file
            ),
        }

        for key, value in (
            metrics_ccdf_native.items()
        ):
            full_row[
                f"ccdf_native_{key}"
            ] = value

        for key, value in (
            metrics_anchored_native.items()
        ):
            full_row[
                f"anchored_native_{key}"
            ] = value

        for key, value in (
            metrics_ccdf_common.items()
        ):
            full_row[
                f"ccdf_common_{key}"
            ] = value

        for key, value in (
            metrics_anchored_common.items()
        ):
            full_row[
                f"anchored_common_{key}"
            ] = value

        full_results.append(
            full_row
        )

        # -------------------------------------------------------------
        # Compact comparison row
        # -------------------------------------------------------------

        if (
            np.isfinite(
                metrics_ccdf_common[
                    "aicc"
                ]
            )
            and np.isfinite(
                metrics_anchored_common[
                    "aicc"
                ]
            )
        ):
            better_aicc = (
                "anchored k>=2"
                if (
                    metrics_anchored_common[
                        "aicc"
                    ]
                    <
                    metrics_ccdf_common[
                        "aicc"
                    ]
                )
                else "exp(-k/gamma)"
            )
        else:
            better_aicc = ""

        summary_rows.append(
            {
                "country": country,

                "gamma_exp_k": (
                    gamma_ccdf
                ),

                "A_anchored": (
                    A_anchored
                ),

                "gamma_anchored": (
                    gamma_anchored
                ),

                "exp_r2": (
                    metrics_ccdf_common[
                        "r2"
                    ]
                ),

                "anchored_r2": (
                    metrics_anchored_common[
                        "r2"
                    ]
                ),

                "exp_adjusted_r2": (
                    metrics_ccdf_common[
                        "adjusted_r2"
                    ]
                ),

                "anchored_adjusted_r2": (
                    metrics_anchored_common[
                        "adjusted_r2"
                    ]
                ),

                "exp_rmse": (
                    metrics_ccdf_common[
                        "rmse"
                    ]
                ),

                "anchored_rmse": (
                    metrics_anchored_common[
                        "rmse"
                    ]
                ),

                "exp_mae": (
                    metrics_ccdf_common[
                        "mae"
                    ]
                ),

                "anchored_mae": (
                    metrics_anchored_common[
                        "mae"
                    ]
                ),

                "exp_aic": (
                    metrics_ccdf_common[
                        "aic"
                    ]
                ),

                "anchored_aic": (
                    metrics_anchored_common[
                        "aic"
                    ]
                ),

                "exp_aicc": (
                    metrics_ccdf_common[
                        "aicc"
                    ]
                ),

                "anchored_aicc": (
                    metrics_anchored_common[
                        "aicc"
                    ]
                ),

                "lower_aicc_model": (
                    better_aicc
                ),
            }
        )

    full_results = pd.DataFrame(
        full_results
    )

    summary = pd.DataFrame(
        summary_rows
    )

    # -----------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 150)
    print("COMMON-DOMAIN COMPARISON: k >= 2")
    print("=" * 150)

    print(
        f"{'Country':<24}"
        f"{'g exp':>9}"
        f"{'A':>9}"
        f"{'g anch':>9}"
        f"{'R2 exp':>10}"
        f"{'R2 anch':>10}"
        f"{'RMSE exp':>11}"
        f"{'RMSE anch':>11}"
        f"{'AICc exp':>11}"
        f"{'AICc anch':>12}"
        f"{'Lower AICc':>18}"
    )

    print(
        "-" * 150
    )

    for _, row in (
        summary.iterrows()
    ):

        print(
            f"{row['country']:<24}"
            f"{row['gamma_exp_k']:>9.3f}"
            f"{row['A_anchored']:>9.3f}"
            f"{row['gamma_anchored']:>9.3f}"
            f"{row['exp_r2']:>10.4f}"
            f"{row['anchored_r2']:>10.4f}"
            f"{row['exp_rmse']:>11.4f}"
            f"{row['anchored_rmse']:>11.4f}"
            f"{row['exp_aicc']:>11.2f}"
            f"{row['anchored_aicc']:>12.2f}"
            f"{row['lower_aicc_model']:>18}"
        )

    # -----------------------------------------------------------------
    # Overall model comparison
    # -----------------------------------------------------------------

    n_exp_better = int(
        (
            summary[
                "lower_aicc_model"
            ]
            == "exp(-k/gamma)"
        ).sum()
    )

    n_anchored_better = int(
        (
            summary[
                "lower_aicc_model"
            ]
            == "anchored k>=2"
        ).sum()
    )

    print("\n")
    print("=" * 150)
    print("OVERALL SUMMARY")
    print("=" * 150)

    print(
        f"Countries with lower AICc for exp(-k/gamma) : "
        f"{n_exp_better}"
    )

    print(
        f"Countries with lower AICc for anchored k>=2 : "
        f"{n_anchored_better}"
    )

    print(
        f"\nMean common-domain R2, exp(-k/gamma)        : "
        f"{summary['exp_r2'].mean():.4f}"
    )

    print(
        f"Mean common-domain R2, anchored k>=2        : "
        f"{summary['anchored_r2'].mean():.4f}"
    )

    print(
        f"\nMean common-domain RMSE, exp(-k/gamma)      : "
        f"{summary['exp_rmse'].mean():.4f}"
    )

    print(
        f"Mean common-domain RMSE, anchored k>=2      : "
        f"{summary['anchored_rmse'].mean():.4f}"
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    full_results.to_csv(
        FULL_RESULTS_FILE,
        index=False,
    )

    print("\n")
    print("=" * 150)
    print("SAVED")
    print("=" * 150)

    print(
        f"Compact comparison table:\n"
        f"  {SUMMARY_FILE}"
    )

    print(
        f"\nFull diagnostic table:\n"
        f"  {FULL_RESULTS_FILE}"
    )

    print(
        f"\nCountry plots:\n"
        f"  {PLOT_DIR}"
    )

    print("\n")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
