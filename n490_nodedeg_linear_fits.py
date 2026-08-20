#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:22:13 2026

@author: geoffreydesena
"""

# -*- coding: utf-8 -*-
"""
Analyze how nodal-degree probabilities vary with voltage in Nordic490.

For each line-only voltage network (220, 300, 380 kV):

1. Calculate the nodal degree of every node appearing in model.line.
2. Calculate the probability P(k) for node degrees k = 1, ..., 7.
3. For each degree k, form a three-point series:

       (220, P_220(k))
       (300, P_300(k))
       (380, P_380(k))

4. Fit a linear model:

       P(k) = slope * V + intercept

5. Plot all seven empirical series and their fitted lines.
6. Annotate each fitted series with its equation and R-squared.
7. Save probability and fit-summary tables.

Parallel lines are counted independently.
Transformer-only buses are excluded because this analysis uses model.line only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VOLTAGE_LEVELS = [220, 300, 380]

DEGREES = list(
    range(1, 8)
)

OUTPUT_DIR = Path(
    "n490_degree_probability_voltage_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASE_FONTSIZE = (
    plt.rcParams["font.size"]
    * 1.30
)


# ---------------------------------------------------------------------
# Endpoint detection
# ---------------------------------------------------------------------

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two bus-endpoint columns in model.line.
    """
    candidate_pairs = [
        ("bus0", "bus1"),
        ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"),
        ("fbus", "tbus"),
        ("from", "to"),
    ]

    for bus0_col, bus1_col in candidate_pairs:

        if (
            bus0_col in lines.columns
            and bus1_col in lines.columns
        ):
            return (
                bus0_col,
                bus1_col,
            )

    raise ValueError(
        "Could not identify line endpoint columns.\n"
        f"Available columns:\n"
        f"{lines.columns.tolist()}"
    )


# ---------------------------------------------------------------------
# Degree distribution
# ---------------------------------------------------------------------

def calculate_degree_distribution(
    lines: pd.DataFrame,
    voltage_kv: int,
) -> pd.DataFrame:
    """
    Calculate P(k) for one voltage-specific line graph.

    The node set consists only of buses that occur as endpoints of
    model.line branches at the selected voltage.
    """
    if "Vbase" not in lines.columns:
        raise ValueError(
            "model.line does not contain 'Vbase'."
        )

    bus0_col, bus1_col = (
        resolve_line_endpoint_columns(
            lines
        )
    )

    line_voltage = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    mask = np.isclose(
        line_voltage,
        float(voltage_kv),
        equal_nan=False,
    )

    voltage_lines = lines.loc[
        mask,
        [bus0_col, bus1_col],
    ].copy()

    if voltage_lines.empty:
        raise ValueError(
            f"No lines found at {voltage_kv} kV."
        )

    endpoints = pd.concat(
        [
            voltage_lines[bus0_col],
            voltage_lines[bus1_col],
        ],
        ignore_index=True,
    ).dropna()

    degrees = (
        endpoints
        .value_counts()
        .astype(int)
    )

    # Handshaking-lemma sanity check.
    if degrees.sum() != 2 * len(voltage_lines):

        raise RuntimeError(
            f"Degree check failed at {voltage_kv} kV."
        )

    counts = (
        degrees
        .value_counts()
        .sort_index()
    )

    n_nodes = len(
        degrees
    )

    rows = []

    for degree in DEGREES:

        n_degree = int(
            counts.get(
                degree,
                0,
            )
        )

        probability = (
            n_degree
            / n_nodes
        )

        rows.append(
            {
                "Vbase":
                    int(voltage_kv),

                "degree":
                    int(degree),

                "n_nodes_degree":
                    n_degree,

                "n_nodes_total":
                    int(n_nodes),

                "probability":
                    float(probability),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Linear fitting
# ---------------------------------------------------------------------

def fit_degree_probability_series(
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit probability versus voltage independently for each node degree.

    Model:

        P(k) = slope * V + intercept
    """
    rows = []

    for degree in DEGREES:

        data = (
            probabilities.loc[
                probabilities["degree"]
                == degree
            ]
            .sort_values(
                "Vbase"
            )
        )

        x = data[
            "Vbase"
        ].to_numpy(
            dtype=float
        )

        y = data[
            "probability"
        ].to_numpy(
            dtype=float
        )

        if len(x) != 3:
            raise ValueError(
                f"Expected three voltage observations "
                f"for degree {degree}; got {len(x)}."
            )

        slope, intercept = np.polyfit(
            x,
            y,
            deg=1,
        )

        fitted = (
            slope * x
            + intercept
        )

        residuals = (
            y - fitted
        )

        ss_res = float(
            np.sum(
                residuals ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    y
                    - np.mean(y)
                ) ** 2
            )
        )

        r_squared = (
            1.0
            - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        rows.append(
            {
                "degree":
                    int(degree),

                "slope":
                    float(slope),

                "intercept":
                    float(intercept),

                "R2":
                    float(r_squared),

                "RMSE":
                    rmse,

                "P_220":
                    float(y[0]),

                "P_300":
                    float(y[1]),

                "P_380":
                    float(y[2]),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_probability_voltage_fits(
    probabilities: pd.DataFrame,
    fits: pd.DataFrame,
) -> None:
    """
    Plot P(k) as a function of voltage for degrees 1 through 7.
    """
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    voltage_fit = np.linspace(
        min(VOLTAGE_LEVELS),
        max(VOLTAGE_LEVELS),
        300,
    )

    # -------------------------------------------------------------
    # Plot all seven series
    # -------------------------------------------------------------
    for degree in DEGREES:

        data = (
            probabilities.loc[
                probabilities["degree"]
                == degree
            ]
            .sort_values(
                "Vbase"
            )
        )

        fit = fits.loc[
            fits["degree"]
            == degree
        ].iloc[0]

        x = data[
            "Vbase"
        ].to_numpy(
            dtype=float
        )

        y = data[
            "probability"
        ].to_numpy(
            dtype=float
        )

        y_fit = (
            fit["slope"]
            * voltage_fit
            + fit["intercept"]
        )

        # Empirical trio.
        ax.scatter(
            x,
            y,
            s=60,
            zorder=5,
            label=f"$k={degree}$",
        )

        # Linear fit.
        ax.plot(
            voltage_fit,
            y_fit,
            linewidth=1.8,
            zorder=3,
        )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------
    ax.set_xlabel(
        "Voltage level [kV]",
        fontsize=BASE_FONTSIZE,
    )

    ax.set_ylabel(
        "Node-degree probability $P(k)$",
        fontsize=BASE_FONTSIZE,
    )

    ax.set_xticks(
        VOLTAGE_LEVELS
    )

    ax.tick_params(
        axis="both",
        labelsize=BASE_FONTSIZE,
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

    # -------------------------------------------------------------
    # Equation / R2 text
    # -------------------------------------------------------------
    fit_lines = []

    for _, fit in fits.iterrows():

        degree = int(
            fit["degree"]
        )

        fit_lines.append(
            (
                rf"$k={degree}:$ "
                rf"$P={fit['slope']:.6f}V"
                rf"{fit['intercept']:+.4f}$"
                "\n"
                rf"$R^2={fit['R2']:.4f}$"
            )
        )

    fit_text = "\n\n".join(
        fit_lines
    )

    ax.text(
        1.02,
        0.98,
        fit_text,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=BASE_FONTSIZE * 0.72,
    )

    ax.legend(
        title="Node degree",
        fontsize=BASE_FONTSIZE * 0.8,
        title_fontsize=BASE_FONTSIZE * 0.8,
        loc="upper right",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "N490_degree_probability_vs_voltage.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"\nSaved plot:\n  {output_path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    model = N490(
        year=2018
    )

    lines = model.line.copy()

    # -------------------------------------------------------------
    # Calculate all three voltage-specific degree distributions
    # -------------------------------------------------------------
    probability_tables = []

    for voltage_kv in VOLTAGE_LEVELS:

        distribution = (
            calculate_degree_distribution(
                lines=lines,
                voltage_kv=voltage_kv,
            )
        )

        probability_tables.append(
            distribution
        )

    probabilities = pd.concat(
        probability_tables,
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Fit each degree series
    # -------------------------------------------------------------
    fits = (
        fit_degree_probability_series(
            probabilities
        )
    )

    # -------------------------------------------------------------
    # Print probability table
    # -------------------------------------------------------------
    probability_matrix = (
        probabilities
        .pivot(
            index="degree",
            columns="Vbase",
            values="probability",
        )
        .reset_index()
    )

    probability_matrix.columns.name = None

    print("\n")
    print("=" * 80)
    print("N490 node-degree probabilities by voltage")
    print("=" * 80)

    print(
        probability_matrix
        .round(6)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Print fit results
    # -------------------------------------------------------------
    print("\n")
    print("=" * 100)
    print("Linear fits: node-degree probability versus voltage")
    print("=" * 100)

    print(
        fits[
            [
                "degree",
                "slope",
                "intercept",
                "R2",
                "RMSE",
            ]
        ]
        .round(
            {
                "slope": 8,
                "intercept": 6,
                "R2": 4,
                "RMSE": 6,
            }
        )
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    plot_probability_voltage_fits(
        probabilities=probabilities,
        fits=fits,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    probabilities.to_csv(
        OUTPUT_DIR
        / "N490_degree_probabilities_by_voltage.csv",
        index=False,
    )

    probabilities.to_pickle(
        OUTPUT_DIR
        / "N490_degree_probabilities_by_voltage.pkl"
    )

    fits.to_csv(
        OUTPUT_DIR
        / "N490_degree_probability_linear_fits.csv",
        index=False,
    )

    fits.to_pickle(
        OUTPUT_DIR
        / "N490_degree_probability_linear_fits.pkl"
    )

    print("\nSaved results in:")
    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()