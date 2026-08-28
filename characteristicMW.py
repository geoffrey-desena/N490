"""
Linear fit of characteristic MW as a function of nominal voltage.

Source:
Birchfield et al. (2018), Table 2.

Model:
    MW = a * V + b

Outputs:
    characteristic_mw_linear_fit.png
    characteristic_mw_linear_fit_parameters.csv
    characteristic_mw_estimated_values.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =============================================================================
# USER SETTINGS
# =============================================================================

OUTPUT_DIR = Path("characteristic_mw_fit")

FIG_DPI = 300

# Figure appearance
TEXT_SIZE = 12
MARKER_SIZE = 45
LINE_WIDTH = 1.5

# Nominal voltages relevant to the Nordic-grid project
TARGET_VOLTAGES_KV = [132, 220, 300, 400]


# =============================================================================
# INPUT DATA
# =============================================================================

# Birchfield et al. (2018), Table 2
DATA = pd.DataFrame(
    {
        "nominal_kv": [
            115,
            138,
            161,
            230,
            345,
            500,
            765,
        ],
        "characteristic_mw": [
            160,
            223,
            265,
            541,
            1195,
            2598,
            4100,
        ],
    }
)


# =============================================================================
# MODEL
# =============================================================================

def linear_model(voltage_kv, slope, intercept):
    """
    Linear relationship:

        MW = slope * V + intercept
    """
    return slope * voltage_kv + intercept


# =============================================================================
# FIT
# =============================================================================

def fit_linear_model(data):
    voltage = data["nominal_kv"].to_numpy(dtype=float)
    mw = data["characteristic_mw"].to_numpy(dtype=float)

    parameters, covariance = curve_fit(
        linear_model,
        voltage,
        mw,
    )

    slope, intercept = parameters
    slope_std, intercept_std = np.sqrt(np.diag(covariance))

    predicted = linear_model(
        voltage,
        slope,
        intercept,
    )

    residuals = mw - predicted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mw - np.mean(mw))**2)

    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals**2))

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_std": slope_std,
        "intercept_std": intercept_std,
        "r2": r2,
        "rmse_mw": rmse,
    }


# =============================================================================
# ESTIMATED PROJECT VALUES
# =============================================================================

def estimate_project_values(fit):
    voltages = np.asarray(
        TARGET_VOLTAGES_KV,
        dtype=float,
    )

    estimated_mw = linear_model(
        voltages,
        fit["slope"],
        fit["intercept"],
    )

    return pd.DataFrame(
        {
            "nominal_kv": voltages.astype(int),
            "estimated_characteristic_mw": estimated_mw,
        }
    )


# =============================================================================
# PLOT
# =============================================================================

def plot_linear_fit(data, fit, output_dir):
    voltage = data["nominal_kv"].to_numpy(dtype=float)
    mw = data["characteristic_mw"].to_numpy(dtype=float)

    slope = fit["slope"]
    intercept = fit["intercept"]

    voltage_curve = np.linspace(
        0.95 * voltage.min(),
        1.02 * voltage.max(),
        500,
    )

    mw_curve = linear_model(
        voltage_curve,
        slope,
        intercept,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0),
    )

    # Birchfield observations
    ax.scatter(
        voltage,
        mw,
        s=MARKER_SIZE,
        zorder=3,
    )

    # Linear regression
    ax.plot(
        voltage_curve,
        mw_curve,
        linewidth=LINE_WIDTH,
        zorder=2,
    )

    ax.set_xlabel(
        "Nominal voltage (kV)",
        fontsize=TEXT_SIZE,
        color="#000000",
    )

    ax.set_ylabel(
        "Characteristic capacity (MW)",
        fontsize=TEXT_SIZE,
        color="#000000",
    )

    ax.tick_params(
        axis="both",
        labelsize=TEXT_SIZE,
        colors="#000000",
    )

    # Minimal figure styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    ax.grid(False)

    # Fit information
    annotation = (
        r"$MW = aV + b$"
        "\n"
        rf"$a = {fit['slope']:.3f} \pm {fit['slope_std']:.3f}$ MW/kV"
        "\n"
        rf"$b = {fit['intercept']:.1f} \pm {fit['intercept_std']:.1f}$ MW"
        "\n"
        rf"$R^2 = {fit['r2']:.4f}$"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_SIZE,
        color="#000000",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "characteristic_mw_linear_fit.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Fit regression
    fit = fit_linear_model(DATA)

    # Estimate characteristic MW at project voltage levels
    estimates = estimate_project_values(fit)

    # Plot
    plot_linear_fit(
        DATA,
        fit,
        OUTPUT_DIR,
    )

    # Save fit parameters
    fit_table = pd.DataFrame(
        [
            {
                "model": "MW = a * V + b",
                "slope_mw_per_kv": fit["slope"],
                "slope_std": fit["slope_std"],
                "intercept_mw": fit["intercept"],
                "intercept_std": fit["intercept_std"],
                "r2": fit["r2"],
                "rmse_mw": fit["rmse_mw"],
            }
        ]
    )

    fit_table.to_csv(
        OUTPUT_DIR / "characteristic_mw_linear_fit_parameters.csv",
        index=False,
    )

    # Save estimated project values
    estimates.to_csv(
        OUTPUT_DIR / "characteristic_mw_estimated_values.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CHARACTERISTIC MW LINEAR REGRESSION")
    print("=" * 80)

    print()
    print(
        f"MW = {fit['slope']:.6f} * V "
        f"{fit['intercept']:+.6f}"
    )

    print()
    print(
        f"Slope     = {fit['slope']:.6f} "
        f"+/- {fit['slope_std']:.6f} MW/kV"
    )
    print(
        f"Intercept = {fit['intercept']:.6f} "
        f"+/- {fit['intercept_std']:.6f} MW"
    )
    print(f"R2        = {fit['r2']:.6f}")
    print(f"RMSE      = {fit['rmse_mw']:.3f} MW")

    print()
    print("=" * 80)
    print("ESTIMATED PROJECT VALUES")
    print("=" * 80)
    print()

    print(
        estimates.to_string(
            index=False,
            formatters={
                "estimated_characteristic_mw":
                    "{:.1f}".format,
            },
        )
    )

    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()