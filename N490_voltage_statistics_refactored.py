# -*- coding: utf-8 -*-
"""
N490 voltage statistics analysis.

This script extracts voltage-related validation statistics from the Nordic490
(N490) reference model. It is intended to be run inside the dedicated N490
virtual environment, from the root directory containing the N490 scripts and
the external ``Data/`` folder.

The script performs three independent analyses:

1. Bus voltage relationship with generator capacity
   Combines conventional generators from ``Data/gen.pkl`` and renewable farms
   from ``Data/farms.pkl``, maps each unit to the nominal voltage of its
   connected bus, fits a simple linear regression between voltage and capacity,
   and saves both the underlying data and a scatter plot.

2. Bus voltage relationship with load share
   Uses the ``load_share`` column in ``m.bus`` as a normalized proxy for bus
   load size, fits a simple linear regression between nominal bus voltage and
   load share, and saves both the underlying data and a scatter plot.

3. Voltage level proportions by bidding zone
   Constructs a substation-level table by grouping buses that share rounded
   latitude/longitude coordinates. For each substation, the script records
   whether it hosts 132, 220, 300, and/or 380 kV buses. It then computes, for
   each bidding zone, the percentage of substations hosting each nominal voltage.

Outputs are written to ``N490_OUTPUT_DIR``. By default, this is set to the
N490 raw-data directory used by the synthetic Nordic grid project:

    /Users/geoffreydesena/Documents/nordic-grid/data/raw/n490

The outputs are deliberately saved as both human-readable CSV files and
machine-readable pickle files where useful, so that later scripts can reuse
these N490-derived statistics without rerunning the analysis.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

from nordic490 import N490


N490_OUTPUT_DIR = Path("/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490")
VOLTAGES_KV = [132.0, 220.0, 300.0, 380.0]


def ensure_output_dir(output_dir: Path = N490_OUTPUT_DIR) -> Path:
    """Create the output directory if needed and return it as a ``Path``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def regression_to_dict(reg) -> dict:
    """Convert a scipy ``linregress`` result into a serializable dictionary."""

    return {
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "r": float(reg.rvalue),
        "r_squared": float(reg.rvalue**2),
        "p_value": float(reg.pvalue),
        "stderr": float(reg.stderr),
        "intercept_stderr": float(reg.intercept_stderr)
        if getattr(reg, "intercept_stderr", None) is not None
        else None,
    }


def save_regression(name: str, reg, output_dir: Path) -> pd.DataFrame:
    """Save regression statistics as CSV and JSON, then return the table."""

    stats = regression_to_dict(reg)
    df = pd.DataFrame([stats])
    df.to_csv(output_dir / f"{name}_regression.csv", index=False)

    with open(output_dir / f"{name}_regression.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return df


def load_n490_model(year: int = 2018) -> N490:
    """
    Load the N490 model and restore local saved generator/farm tables.

    The upstream model may attempt to update or filter generator information
    through online data sources. For this analysis, we explicitly overwrite
    ``m.gen`` and ``m.farms`` with the saved pickle files in ``Data/`` so that
    the analysis is reproducible and does not depend on the current ENTSO-E
    website structure.
    """

    model = N490(year=year)
    model.gen = pd.read_pickle("Data/gen.pkl")
    model.farms = pd.read_pickle("Data/farms.pkl")
    return model


def identify_first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Return the first candidate column found in ``df`` or raise a clear error."""

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify {label} column. "
        f"Tried {candidates}; available columns are {list(df.columns)}"
    )


def analyze_generator_capacity_by_voltage(
    model: N490,
    output_dir: Path = N490_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze the relationship between generator capacity and connected bus voltage.

    Conventional generators are taken from ``model.gen`` and renewable farms are
    taken from ``model.farms``. The farm dataframe may use different column names
    for bus ID and capacity, so this function detects common alternatives and
    standardizes them to ``bus`` and ``Pmax``.

    For each generator/farm, the script maps the connected bus ID to
    ``model.bus['Vbase']``. It then fits a simple linear regression:

        Pmax = intercept + slope * bus_voltage_kv

    Saved outputs:
      - ``N490_generator_capacity_by_voltage_data.csv``
      - ``N490_generator_capacity_by_voltage_data.pkl``
      - ``N490_generator_capacity_by_voltage_summary.csv``
      - ``N490_generator_capacity_by_voltage_regression.csv``
      - ``N490_generator_capacity_by_voltage_regression.json``
      - ``N490_generator_capacity_by_voltage.png``
    """

    output_dir = ensure_output_dir(output_dir)

    bus = model.bus.copy()
    gen = model.gen.copy()
    farms = model.farms.copy()

    gen["source"] = "conventional"
    farms["source"] = "renewable"

    farm_bus_col = identify_first_existing_column(
        farms,
        ["bus", "Bus", "bus_id"],
        "farm bus",
    )
    farm_cap_col = identify_first_existing_column(
        farms,
        ["Pmax", "pmax", "MW", "capacity", "max_p_mw"],
        "farm capacity",
    )

    farms = farms.rename(columns={farm_bus_col: "bus", farm_cap_col: "Pmax"})

    all_gen = pd.concat(
        [
            gen[["bus", "Pmax", "source"]],
            farms[["bus", "Pmax", "source"]],
        ],
        ignore_index=True,
    )

    all_gen["bus_voltage_kv"] = all_gen["bus"].map(bus["Vbase"])

    plot_df = all_gen.dropna(subset=["bus_voltage_kv", "Pmax"]).copy()
    plot_df["bus_voltage_kv"] = pd.to_numeric(plot_df["bus_voltage_kv"], errors="coerce")
    plot_df["Pmax"] = pd.to_numeric(plot_df["Pmax"], errors="coerce")
    plot_df = plot_df.dropna(subset=["bus_voltage_kv", "Pmax"])

    reg = linregress(plot_df["bus_voltage_kv"], plot_df["Pmax"])

    summary = (
        plot_df.groupby(["source", "bus_voltage_kv"])["Pmax"]
        .agg(
            n_units="size",
            total_pmax_mw="sum",
            mean_pmax_mw="mean",
            median_pmax_mw="median",
            max_pmax_mw="max",
        )
        .reset_index()
        .sort_values(["source", "bus_voltage_kv"])
    )

    plot_df.to_csv(output_dir / "N490_generator_capacity_by_voltage_data.csv", index=False)
    plot_df.to_pickle(output_dir / "N490_generator_capacity_by_voltage_data.pkl")
    summary.to_csv(output_dir / "N490_generator_capacity_by_voltage_summary.csv", index=False)
    regression_df = save_regression("N490_generator_capacity_by_voltage", reg, output_dir)

    fig, ax = plt.subplots(figsize=(9, 6))

    for source, marker in [("conventional", "o"), ("renewable", "^")]:
        subset = plot_df[plot_df["source"] == source]
        ax.scatter(
            subset["bus_voltage_kv"],
            subset["Pmax"],
            s=45,
            alpha=0.7,
            marker=marker,
            label=source,
        )

    xline = sorted(plot_df["bus_voltage_kv"].unique())
    yline = [reg.intercept + reg.slope * x for x in xline]

    ax.plot(
        xline,
        yline,
        linewidth=2,
        label=f"Linear fit (r = {reg.rvalue:.2f})",
    )

    ax.set_xlabel("Bus nominal voltage (kV)")
    ax.set_ylabel("Generator capacity Pmax (MW)")
    ax.set_title("N490 generation capacity by connected bus voltage")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "N490_generator_capacity_by_voltage.png", dpi=200)
    plt.close(fig)

    return plot_df, summary, regression_df


def analyze_load_share_by_voltage(
    model: N490,
    output_dir: Path = N490_OUTPUT_DIR,
    positive_load_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze the relationship between bus load share and nominal bus voltage.

    The N490 bus table stores load as ``load_share``, a normalized share of total
    demand assigned to each bus. This function uses ``model.bus[['Vbase',
    'load_share']]`` and fits a simple linear regression:

        load_share = intercept + slope * Vbase

    By default, zero-load buses are excluded. This matches the intended analysis
    of whether load magnitude, among buses that actually host load, varies with
    nominal voltage. Set ``positive_load_only=False`` to include zero-load buses.

    Saved outputs:
      - ``N490_load_share_by_voltage_data.csv``
      - ``N490_load_share_by_voltage_data.pkl``
      - ``N490_load_share_by_voltage_summary.csv``
      - ``N490_load_share_by_voltage_regression.csv``
      - ``N490_load_share_by_voltage_regression.json``
      - ``N490_load_share_by_voltage.png``
    """

    output_dir = ensure_output_dir(output_dir)

    bus = model.bus.copy()
    plot_df = bus[["Vbase", "load_share"]].copy()
    plot_df["Vbase"] = pd.to_numeric(plot_df["Vbase"], errors="coerce")
    plot_df["load_share"] = pd.to_numeric(plot_df["load_share"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Vbase", "load_share"])

    if positive_load_only:
        plot_df = plot_df[plot_df["load_share"] > 0].copy()

    reg = linregress(plot_df["Vbase"], plot_df["load_share"])

    summary = (
        plot_df.groupby("Vbase")["load_share"]
        .agg(
            n_buses="size",
            total_load_share="sum",
            mean_load_share="mean",
            median_load_share="median",
            max_load_share="max",
        )
        .reset_index()
        .sort_values("Vbase")
    )

    plot_df.to_csv(output_dir / "N490_load_share_by_voltage_data.csv", index=False)
    plot_df.to_pickle(output_dir / "N490_load_share_by_voltage_data.pkl")
    summary.to_csv(output_dir / "N490_load_share_by_voltage_summary.csv", index=False)
    regression_df = save_regression("N490_load_share_by_voltage", reg, output_dir)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        plot_df["Vbase"],
        plot_df["load_share"],
        s=45,
        alpha=0.7,
        marker="o",
    )

    xline = sorted(plot_df["Vbase"].unique())
    yline = [reg.intercept + reg.slope * x for x in xline]

    ax.plot(
        xline,
        yline,
        linewidth=2,
        label=f"Linear fit (r = {reg.rvalue:.2f})",
    )

    ax.set_xlabel("Bus nominal voltage (kV)")
    ax.set_ylabel("Load share")
    ax.set_title("N490 load share by connected bus voltage")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "N490_load_share_by_voltage.png", dpi=200)
    plt.close(fig)

    return plot_df, summary, regression_df


def build_substation_voltage_table(
    bus: pd.DataFrame,
    voltages_kv: list[float] = VOLTAGES_KV,
    coordinate_round_digits: int = 6,
) -> pd.DataFrame:
    """
    Construct a substation-level table from N490 buses.

    N490 does not explicitly store a substation identifier in the bus dataframe.
    This helper treats buses with the same rounded latitude and longitude as
    belonging to the same physical substation. It then creates one row per
    inferred substation and adds one boolean column per nominal voltage.

    Parameters
    ----------
    bus:
        N490 bus dataframe.
    voltages_kv:
        Nominal voltages to represent as boolean columns.
    coordinate_round_digits:
        Number of decimal places used when grouping latitude/longitude.
        Six decimal places is usually strict enough to avoid accidental merging
        while still avoiding floating-point representation noise.

    Returns
    -------
    pandas.DataFrame
        One row per inferred substation, with columns for representative
        location, bidding zone, country, voltage presence booleans, and count
        of hosted voltage levels.
    """

    bus = bus.copy()

    bus["lat_r"] = bus["lat"].round(coordinate_round_digits)
    bus["lon_r"] = bus["lon"].round(coordinate_round_digits)
    bus["substation_key"] = bus["lat_r"].astype(str) + "_" + bus["lon_r"].astype(str)

    substation_lookup = {
        key: idx
        for idx, key in enumerate(sorted(bus["substation_key"].unique()))
    }
    bus["substation_id"] = bus["substation_key"].map(substation_lookup)

    substations = (
        bus.groupby("substation_id")
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            bidz=("bidz", "first"),
            country=("country", "first"),
        )
        .reset_index()
    )

    for v in voltages_kv:
        col = f"{int(v)}_kV"
        buses_at_v = bus[bus["Vbase"] == v].groupby("substation_id").size().index
        substations[col] = substations["substation_id"].isin(buses_at_v)

    voltage_cols = [f"{int(v)}_kV" for v in voltages_kv]
    substations["n_voltage_levels"] = substations[voltage_cols].sum(axis=1)

    return substations


def analyze_voltage_proportions_by_bidding_zone(
    model: N490,
    output_dir: Path = N490_OUTPUT_DIR,
    voltages_kv: list[float] = VOLTAGES_KV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze which nominal voltages are present at substations in each bidding zone.

    This analysis first infers substations by grouping buses with identical
    rounded coordinates. For each inferred substation, the script records
    whether at least one bus exists at each nominal voltage in ``voltages_kv``.

    It then computes, for every bidding zone, the percentage of substations
    hosting each voltage. For example, if every inferred substation in SE3 has
    a 132 kV bus, the ``132_kV`` cell for SE3 will be 100.0.

    Saved outputs:
      - ``N490_substation_voltage_table.csv``
      - ``N490_substation_voltage_table.pkl``
      - ``N490_substation_counts_by_bidding_zone.csv``
      - ``N490_voltage_proportions_by_bidding_zone.csv``
      - ``N490_voltage_proportions_by_bidding_zone.pkl``
      - ``N490_voltage_proportions_by_bidding_zone_formatted.csv``
      - ``N490_substation_voltage_level_count_distribution.csv``
    """

    output_dir = ensure_output_dir(output_dir)

    bus = model.bus.copy()
    substations = build_substation_voltage_table(bus, voltages_kv=voltages_kv)

    zone_counts = (
        substations.groupby("bidz")
        .size()
        .reset_index(name="n_substations")
        .sort_values("bidz")
    )

    voltage_cols = [f"{int(v)}_kV" for v in voltages_kv]

    proportions = (
        substations.groupby("bidz")[voltage_cols]
        .mean()
        .mul(100.0)
        .reset_index()
        .sort_values("bidz")
    )

    proportions = proportions.merge(zone_counts, on="bidz", how="left")
    proportions = proportions[["bidz", "n_substations"] + voltage_cols]

    formatted = proportions.copy()
    for col in voltage_cols:
        formatted[col] = formatted[col].map(lambda x: f"{x:.1f}%")

    voltage_level_distribution = (
        substations["n_voltage_levels"]
        .value_counts()
        .sort_index()
        .rename_axis("n_voltage_levels")
        .reset_index(name="n_substations")
    )

    substations.to_csv(output_dir / "N490_substation_voltage_table.csv", index=False)
    substations.to_pickle(output_dir / "N490_substation_voltage_table.pkl")
    zone_counts.to_csv(output_dir / "N490_substation_counts_by_bidding_zone.csv", index=False)
    proportions.to_csv(output_dir / "N490_voltage_proportions_by_bidding_zone.csv", index=False)
    proportions.to_pickle(output_dir / "N490_voltage_proportions_by_bidding_zone.pkl")
    formatted.to_csv(
        output_dir / "N490_voltage_proportions_by_bidding_zone_formatted.csv",
        index=False,
    )
    voltage_level_distribution.to_csv(
        output_dir / "N490_substation_voltage_level_count_distribution.csv",
        index=False,
    )

    return substations, zone_counts, proportions


def main() -> None:
    """
    Run all three N490 voltage-statistics analyses and save outputs.

    The script prints compact summaries to the console and writes all reusable
    outputs to ``N490_OUTPUT_DIR``. It does not call ``plt.show()`` because the
    intended workflow is batch extraction of validation artifacts. Open the
    saved PNG files directly when you want to inspect the figures.
    """

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    output_dir = ensure_output_dir(N490_OUTPUT_DIR)

    print("Loading N490 model...")
    model = load_n490_model(year=2018)

    print("\n1. Generator capacity by connected bus voltage")
    gen_data, gen_summary, gen_regression = analyze_generator_capacity_by_voltage(
        model,
        output_dir=output_dir,
    )
    print(gen_summary.to_string(index=False))
    print(gen_regression.to_string(index=False))

    print("\n2. Load share by bus voltage")
    load_data, load_summary, load_regression = analyze_load_share_by_voltage(
        model,
        output_dir=output_dir,
        positive_load_only=True,
    )
    print(load_summary.to_string(index=False))
    print(load_regression.to_string(index=False))

    print("\n3. Voltage-level proportions by bidding zone")
    substations, zone_counts, voltage_proportions = analyze_voltage_proportions_by_bidding_zone(
        model,
        output_dir=output_dir,
        voltages_kv=VOLTAGES_KV,
    )
    print(zone_counts.to_string(index=False))
    print(voltage_proportions.to_string(index=False))

    print("\nSaved outputs to:")
    print(output_dir)


if __name__ == "__main__":
    main()
