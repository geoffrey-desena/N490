#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:56:43 2026

@author: geoffreydesena
"""

# Why are two 220 kV buses in DK2 not connected to any lines?

# -*- coding: utf-8 -*-
"""
Investigate N490 220 kV buses with no connected AC transmission lines.

The script:

- identifies 220 kV buses that do not appear in model.line,
- finds buses connected to them through model.trafo,
- prints details of the suspect and neighboring buses,
- prints generator summaries for the neighboring buses,
- plots suspect buses, transformer neighbors, and AC lines connected
  to those neighboring buses.

This is intended as a standalone diagnostic script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("n490_zero_degree_bus_diagnostic")

VOLTAGE_KV = 220

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_branch_endpoint_columns(
    table: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two bus endpoint columns in a branch-like table.
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
            bus0_col in table.columns
            and bus1_col in table.columns
        ):
            return bus0_col, bus1_col

    raise ValueError(
        "Could not identify branch endpoint columns.\n"
        f"Available columns:\n{table.columns.tolist()}"
    )


def identify_line_isolated_buses(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    voltage_kv: float,
) -> pd.DataFrame:
    """
    Identify buses at a given voltage that do not occur in model.line.
    """
    line_bus0, line_bus1 = find_branch_endpoint_columns(
        lines
    )

    bus_voltage = pd.to_numeric(
        buses["Vbase"],
        errors="coerce",
    )

    voltage_buses = buses.loc[
        np.isclose(
            bus_voltage,
            float(voltage_kv),
            equal_nan=False,
        )
    ].copy()

    # In N490, the bus DataFrame index is the bus ID.
    voltage_buses["bus_id"] = voltage_buses.index

    all_line_endpoints = pd.Index(
        pd.concat(
            [
                lines[line_bus0],
                lines[line_bus1],
            ],
            ignore_index=True,
        )
        .dropna()
        .unique()
    )

    isolated = voltage_buses.loc[
        ~voltage_buses["bus_id"].isin(
            all_line_endpoints
        )
    ].copy()

    return isolated


def find_transformer_neighbors(
    suspect_buses: pd.DataFrame,
    trafos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find transformer rows incident on suspect buses and return the
    neighboring bus IDs.
    """
    trafo_bus0, trafo_bus1 = find_branch_endpoint_columns(
        trafos
    )

    suspect_ids = set(
        suspect_buses["bus_id"].tolist()
    )

    mask = (
        trafos[trafo_bus0].isin(suspect_ids)
        | trafos[trafo_bus1].isin(suspect_ids)
    )

    suspect_trafos = trafos.loc[
        mask
    ].copy()

    rows = []

    for trafo_id, trafo in suspect_trafos.iterrows():

        bus0 = trafo[trafo_bus0]
        bus1 = trafo[trafo_bus1]

        if bus0 in suspect_ids:
            rows.append(
                {
                    "suspect_bus": bus0,
                    "neighbor_bus": bus1,
                    "transformer_id": trafo_id,
                }
            )

        if bus1 in suspect_ids:
            rows.append(
                {
                    "suspect_bus": bus1,
                    "neighbor_bus": bus0,
                    "transformer_id": trafo_id,
                }
            )

    neighbors = pd.DataFrame(rows)

    return suspect_trafos, neighbors


def find_incident_lines(
    bus_ids,
    lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return all AC lines incident on any bus in bus_ids.
    """
    line_bus0, line_bus1 = find_branch_endpoint_columns(
        lines
    )

    bus_ids = set(bus_ids)

    mask = (
        lines[line_bus0].isin(bus_ids)
        | lines[line_bus1].isin(bus_ids)
    )

    return lines.loc[
        mask
    ].copy()


def find_generators_at_buses(
    bus_ids,
    generators: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find generators connected to a set of buses.

    Tries common generator bus-column names.
    """
    candidate_columns = [
        "bus",
        "bus_id",
        "bus0",
    ]

    bus_column = None

    for column in candidate_columns:
        if column in generators.columns:
            bus_column = column
            break

    if bus_column is None:
        raise ValueError(
            "Could not identify generator bus column.\n"
            f"Generator columns:\n{generators.columns.tolist()}"
        )

    result = generators.loc[
        generators[bus_column].isin(
            bus_ids
        )
    ].copy()

    return result


def print_generator_summary(
    generators: pd.DataFrame,
    neighboring_bus_ids,
) -> None:
    """
    Print generator records and useful aggregate summaries.
    """
    print("\n")
    print("=" * 80)
    print("Generation connected to neighboring buses")
    print("=" * 80)

    if generators.empty:
        print(
            "No generators found at the neighboring buses."
        )
        return

    print(
        generators.to_string()
    )

    print("\nGenerator count:")
    print(
        len(generators)
    )

    # -------------------------------------------------------------
    # Per-bus counts
    # -------------------------------------------------------------
    bus_column = next(
        column
        for column in ["bus", "bus_id", "bus0"]
        if column in generators.columns
    )

    print("\nGenerator count by bus:")
    print(
        generators.groupby(
            bus_column
        ).size()
    )

    # -------------------------------------------------------------
    # Try to summarize installed capacity if available
    # -------------------------------------------------------------
    capacity_candidates = [
        "Pmax",
        "p_nom",
        "p_nom_opt",
        "capacity",
        "P",
    ]

    capacity_column = None

    for column in capacity_candidates:
        if column in generators.columns:
            capacity_column = column
            break

    if capacity_column is not None:

        capacity = pd.to_numeric(
            generators[capacity_column],
            errors="coerce",
        )

        print(
            f"\nTotal {capacity_column}: "
            f"{capacity.sum():.3f}"
        )

        print(
            f"\n{capacity_column} by bus:"
        )

        print(
            generators.assign(
                _capacity=capacity
            )
            .groupby(
                bus_column
            )["_capacity"]
            .sum()
        )

    # -------------------------------------------------------------
    # Try fuel/type grouping if available
    # -------------------------------------------------------------
    type_candidates = [
        "type",
        "carrier",
        "fuel",
        "technology",
    ]

    type_column = None

    for column in type_candidates:
        if column in generators.columns:
            type_column = column
            break

    if type_column is not None:

        print(
            f"\nGenerator count by {type_column}:"
        )

        print(
            generators.groupby(
                type_column
            ).size()
            .sort_values(
                ascending=False
            )
        )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_diagnostic_map(
    buses: pd.DataFrame,
    suspect_buses: pd.DataFrame,
    neighbors: pd.DataFrame,
    incident_lines: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot suspect buses, transformer neighbors, and lines incident on
    neighboring buses.
    """
    neighbor_ids = (
        neighbors["neighbor_bus"]
        .drop_duplicates()
        .tolist()
    )

    neighbor_buses = buses.loc[
        buses.index.isin(
            neighbor_ids
        )
    ].copy()

    # -------------------------------------------------------------
    # Determine bounds from all relevant buses
    # -------------------------------------------------------------
    relevant_buses = pd.concat(
        [
            suspect_buses,
            neighbor_buses,
        ],
        axis=0,
    )

    lon_min = relevant_buses["lon"].min()
    lon_max = relevant_buses["lon"].max()

    lat_min = relevant_buses["lat"].min()
    lat_max = relevant_buses["lat"].max()

    lon_pad = max(
        0.15,
        (lon_max - lon_min) * 0.50,
    )

    lat_pad = max(
        0.15,
        (lat_max - lat_min) * 0.50,
    )

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    # -------------------------------------------------------------
    # Background: all nearby N490 buses
    # -------------------------------------------------------------
    nearby_mask = (
        buses["lon"].between(
            lon_min - lon_pad,
            lon_max + lon_pad,
        )
        & buses["lat"].between(
            lat_min - lat_pad,
            lat_max + lat_pad,
        )
    )

    nearby_buses = buses.loc[
        nearby_mask
    ]

    ax.scatter(
        nearby_buses["lon"],
        nearby_buses["lat"],
        s=12,
        alpha=0.25,
        label="Nearby N490 buses",
        zorder=1,
    )

    # -------------------------------------------------------------
    # AC lines incident on transformer-neighbor buses
    # -------------------------------------------------------------
    for _, line in incident_lines.iterrows():

        lats = line.get(
            "lat"
        )

        lons = line.get(
            "lon"
        )

        if lats is None or lons is None:
            continue

        try:
            ax.plot(
                lons,
                lats,
                linewidth=1.4,
                alpha=0.75,
                zorder=2,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    # -------------------------------------------------------------
    # Transformer connections as straight dashed segments
    # -------------------------------------------------------------
    for _, relation in neighbors.iterrows():

        suspect_id = relation[
            "suspect_bus"
        ]

        neighbor_id = relation[
            "neighbor_bus"
        ]

        suspect = buses.loc[
            suspect_id
        ]

        neighbor = buses.loc[
            neighbor_id
        ]

        ax.plot(
            [
                suspect["lon"],
                neighbor["lon"],
            ],
            [
                suspect["lat"],
                neighbor["lat"],
            ],
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
            zorder=3,
        )

    # -------------------------------------------------------------
    # Neighboring buses
    # -------------------------------------------------------------
    ax.scatter(
        neighbor_buses["lon"],
        neighbor_buses["lat"],
        s=90,
        marker="o",
        label="Transformer-neighbor buses",
        zorder=4,
    )

    # -------------------------------------------------------------
    # Suspect buses
    # -------------------------------------------------------------
    ax.scatter(
        suspect_buses["lon"],
        suspect_buses["lat"],
        s=150,
        marker="x",
        linewidths=3,
        label="220 kV buses with no AC lines",
        zorder=5,
    )

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------
    for bus_id, bus in neighbor_buses.iterrows():

        label = (
            f"{bus_id}: "
            f"{bus.get('name', '')}"
        )

        ax.annotate(
            label,
            xy=(
                bus["lon"],
                bus["lat"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            zorder=6,
        )

    for _, bus in suspect_buses.iterrows():

        label = (
            f"{bus['bus_id']}: "
            f"{bus.get('name', '')}"
        )

        ax.annotate(
            label,
            xy=(
                bus["lon"],
                bus["lat"],
            ),
            xytext=(7, -12),
            textcoords="offset points",
            fontsize=10,
            zorder=6,
        )

    ax.set_xlim(
        lon_min - lon_pad,
        lon_max + lon_pad,
    )

    ax.set_ylim(
        lat_min - lat_pad,
        lat_max + lat_pad,
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.set_title(
        "N490 diagnostic: 220 kV transformer-only buses"
    )

    ax.legend()

    ax.grid(
        False
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"\nSaved map:\n  {output_path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    model = N490(
        year=2018
    )

    buses = model.bus.copy()
    lines = model.line.copy()
    trafos = model.trafo.copy()
    generators = model.gen.copy()

    # -------------------------------------------------------------
    # 1. Identify 220 kV buses with no model.line connection
    # -------------------------------------------------------------
    suspect_buses = identify_line_isolated_buses(
        buses=buses,
        lines=lines,
        voltage_kv=VOLTAGE_KV,
    )

    print("\n")
    print("=" * 80)
    print(
        f"{VOLTAGE_KV} kV buses with no model.line connection"
    )
    print("=" * 80)

    print(
        suspect_buses.to_string()
    )

    print(
        f"\nNumber of suspect buses: "
        f"{len(suspect_buses)}"
    )

    # -------------------------------------------------------------
    # 2. Find transformer connections
    # -------------------------------------------------------------
    suspect_trafos, neighbors = (
        find_transformer_neighbors(
            suspect_buses=suspect_buses,
            trafos=trafos,
        )
    )

    print("\n")
    print("=" * 80)
    print("Transformer connections")
    print("=" * 80)

    print(
        suspect_trafos.to_string()
    )

    print("\n")
    print("=" * 80)
    print("Suspect-to-neighbor relationships")
    print("=" * 80)

    print(
        neighbors.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # 3. Neighboring bus details
    # -------------------------------------------------------------
    neighbor_ids = (
        neighbors["neighbor_bus"]
        .drop_duplicates()
        .tolist()
    )

    neighboring_buses = buses.loc[
        buses.index.isin(
            neighbor_ids
        )
    ].copy()

    print("\n")
    print("=" * 80)
    print("Transformer-neighbor bus details")
    print("=" * 80)

    print(
        neighboring_buses.to_string()
    )

    # -------------------------------------------------------------
    # 4. Lines connected to neighboring buses
    # -------------------------------------------------------------
    incident_lines = find_incident_lines(
        bus_ids=neighbor_ids,
        lines=lines,
    )

    print("\n")
    print("=" * 80)
    print("AC lines connected to transformer-neighbor buses")
    print("=" * 80)

    if incident_lines.empty:
        print(
            "No AC lines found at neighboring buses."
        )
    else:
        print(
            incident_lines.to_string()
        )

    # -------------------------------------------------------------
    # 5. Generation at neighboring buses
    # -------------------------------------------------------------
    neighbor_generators = find_generators_at_buses(
        bus_ids=neighbor_ids,
        generators=generators,
    )

    print_generator_summary(
        generators=neighbor_generators,
        neighboring_bus_ids=neighbor_ids,
    )

    # -------------------------------------------------------------
    # 6. Map
    # -------------------------------------------------------------
    plot_diagnostic_map(
        buses=buses,
        suspect_buses=suspect_buses,
        neighbors=neighbors,
        incident_lines=incident_lines,
        output_path=(
            OUTPUT_DIR
            / "N490_220kv_transformer_only_bus_diagnostic.png"
        ),
    )

    # -------------------------------------------------------------
    # 7. Save tables
    # -------------------------------------------------------------
    suspect_buses.to_pickle(
        OUTPUT_DIR
        / "suspect_buses.pkl"
    )

    suspect_trafos.to_pickle(
        OUTPUT_DIR
        / "suspect_transformers.pkl"
    )

    neighbors.to_pickle(
        OUTPUT_DIR
        / "suspect_bus_neighbors.pkl"
    )

    neighboring_buses.to_pickle(
        OUTPUT_DIR
        / "neighboring_buses.pkl"
    )

    incident_lines.to_pickle(
        OUTPUT_DIR
        / "neighboring_bus_lines.pkl"
    )

    neighbor_generators.to_pickle(
        OUTPUT_DIR
        / "neighboring_bus_generators.pkl"
    )


if __name__ == "__main__":
    main()