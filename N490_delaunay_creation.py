# -*- coding: utf-8 -*-
"""
Build the raw geographic Delaunay triangulation for the Nordic490 network.

Purpose
-------
This script performs only the first two steps of the N490 Delaunay analysis:

1. Build a geographic node set from N490 bus coordinates.
2. Construct an unmodified Delaunay triangulation of those geographic nodes.

No geographic filtering is applied here. In particular, this script does not:
- remove water crossings,
- clip edges to Nordic land/area polygons,
- repair isolated nodes,
- construct an MST,
- construct Delaunay-2 / Delaunay-3 neighbor sets,
- compare the triangulation to actual N490 transmission lines.

The goal is to establish a clean and reproducible baseline before introducing
the geographic modifications needed for the Nordic region.

Outputs
-------
The script creates:

- ``n490_geographic_nodes.geojson``
    One point per unique N490 geographic substation location.

- ``n490_delaunay_raw.geojson``
    All edges in the unmodified Delaunay triangulation.

It also prints summary diagnostics and plots the raw triangulation.

Notes
-----
Multiple N490 buses can occupy the same geographic substation coordinates,
for example when a substation contains buses at several voltage levels. These
co-located buses are collapsed into a single geographic node for this analysis.
"""

from __future__ import annotations

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from shapely.geometry import LineString
import networkx as nx

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("n490_delaunay_analysis")

SOURCE_CRS = "EPSG:4326"

# Same projected CRS used in your 2025 exploratory analysis.
TARGET_CRS = "EPSG:3845"

LAND_AREA_PATH = Path(
    "/Users/geoffreydesena/Documents/N490/"
    "nordic_land_area.geojson"
)

VOLTAGE_LEVELS = [220, 300, 380]
DELAUNAY_K_VALUES = [2, 3, 4, 5]

# ---------------------------------------------------------------------
# Calculate a limit for the Delaunay connections
# ---------------------------------------------------------------------

def get_max_n490_line_length_km(
    lines: pd.DataFrame,
    voltage_kv: float,
    voltage_col: str = "Vbase",
) -> float:
    """
    Return the length of the longest N490 AC transmission line at one voltage.

    Parameters
    ----------
    lines:
        N490 line table. Expected to contain ``lat`` and ``lon`` columns,
        where each row stores the geographic path of one transmission line.
    voltage_kv:
        Nominal voltage level to analyze.
    voltage_col:
        Column containing the nominal line voltage.

    Returns
    -------
    float
        Maximum line length, in kilometers, among N490 lines at the requested
        voltage.

    Notes
    -----
    The Delaunay analysis is performed independently for each voltage level.
    Consequently, the maximum candidate-line length is also calculated
    independently for each voltage network.
    """
    if voltage_col not in lines.columns:
        raise ValueError(
            f"N490 line table does not contain voltage column "
            f"{voltage_col!r}. Available columns: {list(lines.columns)}"
        )

    line_voltage = pd.to_numeric(
        lines[voltage_col],
        errors="coerce",
    )

    voltage_lines = lines[
        np.isclose(
            line_voltage,
            float(voltage_kv),
            equal_nan=False,
        )
    ].copy()

    if voltage_lines.empty:
        raise ValueError(
            f"No N490 lines found at {voltage_kv:g} kV."
        )

    line_geoms = []

    for lats, lons in zip(
        voltage_lines["lat"],
        voltage_lines["lon"],
    ):
        if lats is None or lons is None:
            continue

        pts = [
            (lon, lat)
            for lat, lon in zip(lats, lons)
            if pd.notna(lat) and pd.notna(lon)
        ]

        if len(pts) < 2:
            continue

        line_geoms.append(LineString(pts))

    if not line_geoms:
        raise ValueError(
            f"Could not construct any valid {voltage_kv:g} kV "
            "N490 line geometries."
        )

    gdf_lines = gpd.GeoDataFrame(
        geometry=line_geoms,
        crs=SOURCE_CRS,
    ).to_crs(TARGET_CRS)

    return float(
        (gdf_lines.geometry.length / 1000.0).max()
    )


# ---------------------------------------------------------------------
# Step 1: Build geographic node set
# ---------------------------------------------------------------------

def build_voltage_nodes(
    buses: pd.DataFrame,
    voltage_kv: float,
    source_crs: str = SOURCE_CRS,
    target_crs: str = TARGET_CRS,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build the geographic node set for one N490 voltage network.

    Only buses operating at ``voltage_kv`` are included. Co-located buses at
    that same voltage are collapsed into one geographic node.

    Existing bus point geometries are used when available. If the bus table
    does not contain usable geometry, point geometry is constructed from the
    bus/substation longitude and latitude.

    Parameters
    ----------
    buses:
        N490 bus table. Must contain ``Vbase`` and either valid point geometry
        or ``lat``/``lon`` columns.
    voltage_kv:
        Voltage network to construct, for example 220, 300, or 380 kV.
    source_crs:
        Geographic CRS used for longitude/latitude coordinates.
    target_crs:
        Projected metric CRS used for Delaunay triangulation.

    Returns
    -------
    nodes_wgs84:
        Voltage-specific geographic nodes in ``source_crs``.
    nodes_projected:
        The same nodes projected to ``target_crs``.

    Notes
    -----
    ``node_id`` is deliberately reset to a contiguous zero-based sequence
    independently for every voltage level. All graph operations therefore
    operate only within that voltage network.
    """
    if "Vbase" not in buses.columns:
        raise ValueError(
            "N490 bus table must contain a 'Vbase' column."
        )

    buses_work = buses.copy()
    buses_work["bus_id"] = buses_work.index

    bus_voltage = pd.to_numeric(
        buses_work["Vbase"],
        errors="coerce",
    )

    buses_work = buses_work[
        np.isclose(
            bus_voltage,
            float(voltage_kv),
            equal_nan=False,
        )
    ].copy()

    if buses_work.empty:
        raise ValueError(
            f"No N490 buses found at {voltage_kv:g} kV."
        )

    # -------------------------------------------------------------
    # Establish point geometry.
    # -------------------------------------------------------------
    if (
        isinstance(buses_work, gpd.GeoDataFrame)
        and "geometry" in buses_work.columns
    ):
        bus_gdf = buses_work.copy()

        if bus_gdf.crs is None:
            bus_gdf = bus_gdf.set_crs(
                source_crs,
                allow_override=True,
            )
        else:
            bus_gdf = bus_gdf.to_crs(source_crs)

        valid_geometry = (
            bus_gdf.geometry.notna()
            & ~bus_gdf.geometry.is_empty
        )

        # Backfill missing geometry from lon/lat where possible.
        if not valid_geometry.all():
            if not {"lat", "lon"}.issubset(bus_gdf.columns):
                missing_count = int((~valid_geometry).sum())
                raise ValueError(
                    f"{missing_count} {voltage_kv:g} kV buses have no "
                    "usable geometry and lat/lon are unavailable."
                )

            fill_mask = (
                ~valid_geometry
                & bus_gdf["lat"].notna()
                & bus_gdf["lon"].notna()
            )

            bus_gdf.loc[fill_mask, "geometry"] = (
                gpd.points_from_xy(
                    bus_gdf.loc[fill_mask, "lon"],
                    bus_gdf.loc[fill_mask, "lat"],
                )
            )

        bus_gdf = bus_gdf[
            bus_gdf.geometry.notna()
            & ~bus_gdf.geometry.is_empty
        ].copy()

    else:
        required = {"lat", "lon"}
        missing = required - set(buses_work.columns)

        if missing:
            raise ValueError(
                f"{voltage_kv:g} kV buses have no geometry and the "
                f"table is missing coordinate columns: {sorted(missing)}"
            )

        buses_work = buses_work.dropna(
            subset=["lat", "lon"]
        ).copy()

        bus_gdf = gpd.GeoDataFrame(
            buses_work,
            geometry=gpd.points_from_xy(
                buses_work["lon"],
                buses_work["lat"],
            ),
            crs=source_crs,
        )

    if bus_gdf.empty:
        raise ValueError(
            f"No {voltage_kv:g} kV buses have usable coordinates."
        )

    # -------------------------------------------------------------
    # Work in source CRS so identical substation locations collapse.
    # -------------------------------------------------------------
    bus_gdf = bus_gdf.to_crs(source_crs)

    bus_gdf["_x"] = bus_gdf.geometry.x
    bus_gdf["_y"] = bus_gdf.geometry.y

    grouped_rows = []

    for (x, y), group in bus_gdf.groupby(
        ["_x", "_y"],
        sort=True,
    ):
        if "name" in group.columns:
            bus_names = tuple(
                group["name"].astype(str).tolist()
            )
        else:
            bus_names = tuple()

        grouped_rows.append(
            {
                "voltage_kv": float(voltage_kv),
                "lon": float(x),
                "lat": float(y),
                "n_buses_at_location": len(group),
                "bus_ids": tuple(group["bus_id"].tolist()),
                "bus_names": bus_names,
            }
        )

    nodes_df = pd.DataFrame(grouped_rows).reset_index(drop=True)

    # Local node IDs are unique only within this voltage network.
    nodes_df["node_id"] = np.arange(
        len(nodes_df),
        dtype=int,
    )

    nodes_wgs84 = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(
            nodes_df["lon"],
            nodes_df["lat"],
        ),
        crs=source_crs,
    )

    nodes_projected = nodes_wgs84.to_crs(target_crs)

    return nodes_wgs84, nodes_projected



# ---------------------------------------------------------------------
# Step 2: Construct raw Delaunay triangulation
# ---------------------------------------------------------------------

def build_raw_delaunay_edges(
    nodes_projected: gpd.GeoDataFrame,
    voltage_kv: float,
) -> gpd.GeoDataFrame:
    """
    Construct the raw Delaunay triangulation for one voltage network.

    Parameters
    ----------
    nodes_projected:
        Voltage-specific geographic nodes in the projected analysis CRS.
    voltage_kv:
        Nominal voltage of the network.

    Returns
    -------
    geopandas.GeoDataFrame
        Unique undirected Delaunay edges for this voltage network.
    """
    if len(nodes_projected) < 3:
        raise ValueError(
            f"At least three geographic nodes are required for the "
            f"{voltage_kv:g} kV Delaunay triangulation; found "
            f"{len(nodes_projected)}."
        )

    nodes_projected = (
        nodes_projected
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    expected_ids = np.arange(len(nodes_projected))

    if not np.array_equal(
        nodes_projected["node_id"].to_numpy(),
        expected_ids,
    ):
        raise ValueError(
            "node_id must be contiguous and zero-based within each "
            "voltage network."
        )

    coords = np.array(
        [
            (geom.x, geom.y)
            for geom in nodes_projected.geometry
        ],
        dtype=float,
    )

    triangulation = Delaunay(coords)

    edge_pairs = set()

    for simplex in triangulation.simplices:
        i, j, k = map(int, simplex)

        edge_pairs.add(tuple(sorted((i, j))))
        edge_pairs.add(tuple(sorted((j, k))))
        edge_pairs.add(tuple(sorted((i, k))))

    rows = []

    for u, v in sorted(edge_pairs):
        p_u = nodes_projected.geometry.iloc[u]
        p_v = nodes_projected.geometry.iloc[v]

        geometry = LineString([p_u, p_v])

        rows.append(
            {
                "voltage_kv": float(voltage_kv),
                "from_node": u,
                "to_node": v,
                "length_km": float(
                    geometry.length / 1000.0
                ),
                "edge_source": "delaunay",
                "edge_set": "Delaunay1",
                "geometry": geometry,
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=nodes_projected.crs,
    )


def filter_delaunay_edges_by_length(
    edges: gpd.GeoDataFrame,
    max_length_km: float,
) -> gpd.GeoDataFrame:
    """
    Remove Delaunay edges longer than the allowed maximum.

    Parameters
    ----------
    edges:
        Raw Delaunay edge table containing ``length_km``.
    max_length_km:
        Maximum allowed edge length in kilometers.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered Delaunay edge table.
    """
    if max_length_km <= 0:
        raise ValueError("max_length_km must be greater than zero.")

    out = edges[
        edges["length_km"] <= float(max_length_km)
    ].copy()

    return out.reset_index(drop=True)

def filter_delaunay_edges_by_land_area(
    edges: gpd.GeoDataFrame,
    land_area: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Remove Delaunay edges that depart from the Nordic land-area polygon.

    An edge is retained only when its entire straight-line geometry is covered
    by the land-area geometry. Edges that leave the polygon at any point are
    removed.

    Parameters
    ----------
    edges:
        Delaunay edges, normally after the maximum-length filter.
    land_area:
        Processed Nordic land-area polygon.

    Returns
    -------
    kept:
        Delaunay edges entirely covered by the land area.
    removed:
        Delaunay edges that depart from the land area.

    Notes
    -----
    ``covers`` is used rather than ``within`` so that an edge lying exactly on
    the polygon boundary is retained.
    """
    if edges.crs != land_area.crs:
        land_area = land_area.to_crs(edges.crs)

    # The land-area file should normally contain one combined geometry, but
    # union defensively in case it contains more than one feature.
    land_geom = land_area.geometry.union_all()

    inside_mask = edges.geometry.apply(
        land_geom.covers
    )

    kept = edges.loc[inside_mask].copy().reset_index(drop=True)
    removed = edges.loc[~inside_mask].copy().reset_index(drop=True)

    return kept, removed

def repair_disconnected_delaunay_nodes(
    nodes_projected: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    voltage_kv: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Reconnect disconnected components of one voltage-specific Delaunay graph.

    At each iteration, the shortest straight-line connection from any node
    outside the largest component to any node inside that component is added.
    Connectivity is then recalculated.

    Parameters
    ----------
    nodes_projected:
        Nodes belonging only to the requested voltage network.
    edges:
        Voltage-specific Delaunay-1 edges after length and land filtering.
    voltage_kv:
        Nominal network voltage.

    Returns
    -------
    repaired_edges:
        Filtered Delaunay graph plus connectivity-repair edges.
    repair_edges:
        Connectivity-repair edges only.
    """
    if nodes_projected.crs != edges.crs:
        raise ValueError("Node and edge CRS do not match.")

    node_geometry = (
        nodes_projected
        .set_index("node_id")
        .geometry
        .to_dict()
    )

    all_node_ids = set(
        int(x)
        for x in nodes_projected["node_id"]
    )

    current = edges.copy().reset_index(drop=True)
    repair_rows = []

    iteration = 0

    while True:
        graph = nx.Graph()
        graph.add_nodes_from(all_node_ids)

        if not current.empty:
            graph.add_edges_from(
                zip(
                    current["from_node"].astype(int),
                    current["to_node"].astype(int),
                )
            )

        components = list(
            nx.connected_components(graph)
        )

        if len(components) <= 1:
            break

        iteration += 1

        main_component = max(
            components,
            key=len,
        )

        disconnected_nodes = (
            all_node_ids - main_component
        )

        best = None

        for u in disconnected_nodes:
            p_u = node_geometry[u]

            for v in main_component:
                p_v = node_geometry[v]

                distance_m = p_u.distance(p_v)

                if (
                    best is None
                    or distance_m < best["distance_m"]
                ):
                    best = {
                        "from_node": int(u),
                        "to_node": int(v),
                        "distance_m": float(distance_m),
                    }

        if best is None:
            raise RuntimeError(
                f"Could not identify a repair edge for the "
                f"{voltage_kv:g} kV network."
            )

        u = best["from_node"]
        v = best["to_node"]

        geometry = LineString(
            [
                node_geometry[u],
                node_geometry[v],
            ]
        )

        repair_row = {
            "voltage_kv": float(voltage_kv),
            "from_node": u,
            "to_node": v,
            "length_km": float(
                geometry.length / 1000.0
            ),
            "repair_iteration": iteration,
            "edge_source": "connectivity_repair",
            "edge_set": "Delaunay1",
            "geometry": geometry,
        }

        repair_rows.append(repair_row)

        repair_gdf = gpd.GeoDataFrame(
            [repair_row],
            geometry="geometry",
            crs=edges.crs,
        )

        current = gpd.GeoDataFrame(
            pd.concat(
                [current, repair_gdf],
                ignore_index=True,
            ),
            geometry="geometry",
            crs=edges.crs,
        )

        print(
            f"{voltage_kv:g} kV repair {iteration}: "
            f"node {u} -> node {v}, "
            f"{geometry.length / 1000.0:.2f} km"
        )

    if repair_rows:
        repair_edges = gpd.GeoDataFrame(
            repair_rows,
            geometry="geometry",
            crs=edges.crs,
        )
    else:
        repair_edges = gpd.GeoDataFrame(
            columns=[
                "voltage_kv",
                "from_node",
                "to_node",
                "length_km",
                "repair_iteration",
                "edge_source",
                "edge_set",
                "geometry",
            ],
            geometry="geometry",
            crs=edges.crs,
        )

    print(f"\n{voltage_kv:g} kV connectivity repair")
    print("-" * 32)
    print(f"Repair edges added:             {len(repair_edges)}")
    print("Final connected components:     1")

    return current, repair_edges


def build_mst_from_delaunay(
    nodes_projected: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    voltage_kv: float,
) -> gpd.GeoDataFrame:
    """
    Build the MST for one voltage-specific modified Delaunay network.

    Parameters
    ----------
    nodes_projected:
        Voltage-specific geographic node table.
    edges:
        Connected voltage-specific Delaunay-1 graph.
    voltage_kv:
        Nominal voltage of the network.

    Returns
    -------
    geopandas.GeoDataFrame
        Minimum-spanning-tree edges.
    """
    if nodes_projected.crs != edges.crs:
        raise ValueError("Node and edge CRS do not match.")

    graph = nx.Graph()

    graph.add_nodes_from(
        nodes_projected["node_id"].astype(int)
    )

    for idx, row in edges.iterrows():
        graph.add_edge(
            int(row["from_node"]),
            int(row["to_node"]),
            weight=float(row["length_km"]),
            edge_index=idx,
        )

    if not nx.is_connected(graph):
        raise ValueError(
            f"Cannot construct the {voltage_kv:g} kV MST because "
            f"the graph has "
            f"{nx.number_connected_components(graph)} components."
        )

    mst_graph = nx.minimum_spanning_tree(
        graph,
        weight="weight",
        algorithm="kruskal",
    )

    mst_indices = [
        data["edge_index"]
        for _, _, data in mst_graph.edges(data=True)
    ]

    mst = edges.loc[mst_indices].copy().reset_index(drop=True)

    mst["voltage_kv"] = float(voltage_kv)
    mst["mst"] = True
    mst["edge_set"] = "MST"

    expected_edges = len(nodes_projected) - 1

    if len(mst) != expected_edges:
        raise RuntimeError(
            f"{voltage_kv:g} kV MST has {len(mst)} edges; "
            f"expected {expected_edges}."
        )

    return gpd.GeoDataFrame(
        mst,
        geometry="geometry",
        crs=edges.crs,
    )

def build_delaunay_k_neighbors(
    nodes_projected: gpd.GeoDataFrame,
    delaunay_edges: gpd.GeoDataFrame,
    k: int,
    voltage_kv: float,
) -> gpd.GeoDataFrame:
    """
    Build the Delaunay-k candidate set for one voltage network.

    Two nodes are Delaunay-k neighbors when their shortest-path separation in
    the modified Delaunay-1 graph at the same voltage is exactly ``k`` edges.

    Parameters
    ----------
    nodes_projected:
        Voltage-specific geographic node table.
    delaunay_edges:
        Repaired and filtered Delaunay-1 graph for this voltage.
    k:
        Delaunay graph distance. Must be >= 2.
    voltage_kv:
        Nominal network voltage.

    Returns
    -------
    geopandas.GeoDataFrame
        Direct candidate segments joining every unordered node pair whose
        Delaunay-1 graph distance is exactly ``k``.
    """
    if k < 2:
        raise ValueError("k must be >= 2.")

    if nodes_projected.crs != delaunay_edges.crs:
        raise ValueError(
            "Node and Delaunay edge CRS do not match."
        )

    graph = nx.Graph()

    graph.add_nodes_from(
        nodes_projected["node_id"].astype(int)
    )

    graph.add_edges_from(
        zip(
            delaunay_edges["from_node"].astype(int),
            delaunay_edges["to_node"].astype(int),
        )
    )

    if not nx.is_connected(graph):
        raise ValueError(
            f"The {voltage_kv:g} kV Delaunay-1 graph must be "
            "connected before Delaunay-k neighbors are calculated."
        )

    node_geometry = (
        nodes_projected
        .set_index("node_id")
        .geometry
        .to_dict()
    )

    candidate_pairs = set()

    for source in graph.nodes:
        distances = nx.single_source_shortest_path_length(
            graph,
            source,
            cutoff=k,
        )

        for target, distance in distances.items():
            if distance != k:
                continue

            if source >= target:
                continue

            candidate_pairs.add(
                (int(source), int(target))
            )

    rows = []

    for u, v in sorted(candidate_pairs):
        geometry = LineString(
            [
                node_geometry[u],
                node_geometry[v],
            ]
        )

        rows.append(
            {
                "voltage_kv": float(voltage_kv),
                "from_node": u,
                "to_node": v,
                "length_km": float(
                    geometry.length / 1000.0
                ),
                "delaunay_k": int(k),
                "edge_set": f"Delaunay{k}",
                "geometry": geometry,
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=delaunay_edges.crs,
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def print_summary(
    buses: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> None:
    """
    Print basic diagnostics for the node set and raw triangulation.
    """
    n_valid_buses = buses.dropna(subset=["lat", "lon"]).shape[0]

    print("\nN490 raw Delaunay analysis")
    print("--------------------------")
    print(f"N490 buses total:               {len(buses)}")
    print(f"N490 buses with coordinates:    {n_valid_buses}")
    print(f"Unique geographic nodes:        {len(nodes)}")
    print(f"Co-located buses collapsed:     {n_valid_buses - len(nodes)}")
    print(f"Raw Delaunay edges:             {len(edges)}")

    if not edges.empty:
        print()
        print("Raw Delaunay edge lengths (km)")
        print("------------------------------")
        print(edges["length_km"].describe().round(2).to_string())

    multi_bus = nodes[nodes["n_buses_at_location"] > 1]

    print()
    print(f"Locations containing >1 bus:    {len(multi_bus)}")

    if not multi_bus.empty:
        print(
            multi_bus["n_buses_at_location"]
            .value_counts()
            .sort_index()
            .rename_axis("buses_at_location")
            .reset_index(name="n_locations")
            .to_string(index=False)
        )


def plot_raw_delaunay(
    nodes_projected: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> None:
    """
    Plot the raw Delaunay triangulation over a basemap.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    edges.plot(
        ax=ax,
        linewidth=0.7,
        alpha=0.6,
        label="Raw Delaunay edges",
    )

    nodes_projected.plot(
        ax=ax,
        markersize=8,
        alpha=0.8,
        label="N490 geographic nodes",
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        crs=nodes_projected.crs,
    )

    ax.set_title("N490 Raw Geographic Delaunay Triangulation")
    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()

def load_land_area(
    path: Path = LAND_AREA_PATH,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Load the processed Nordic land-area polygon used to filter Delaunay edges.

    The land-area geometry has already undergone morphological closing so that
    small bays, straits, and gaps between nearby islands do not unnecessarily
    eliminate plausible transmission-line candidates.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Nordic land-area polygon: {path}"
        )

    land_area = gpd.read_file(path)

    if land_area.empty:
        raise ValueError(
            f"Nordic land-area file contains no geometry: {path}"
        )

    if land_area.crs is None:
        raise ValueError(
            "Nordic land-area layer has no CRS."
        )

    land_area = land_area[
        land_area.geometry.notna()
        & ~land_area.geometry.is_empty
    ].copy()

    if land_area.empty:
        raise ValueError(
            "Nordic land-area layer contains no usable geometries."
        )

    return land_area.to_crs(target_crs)

def print_filter_zone_summary(
    zones: gpd.GeoDataFrame,
) -> None:
    """
    Print basic information about the geographic filter polygon layer.
    """
    print("\nDelaunay geographic filter polygons")
    print("-----------------------------------")
    print(f"Rows:             {len(zones)}")
    print(f"CRS:              {zones.crs}")
    print(
        "Geometry types:   "
        + ", ".join(sorted(zones.geometry.geom_type.unique()))
    )

    descriptive_cols = [
        col
        for col in ["zone", "name", "country"]
        if col in zones.columns
    ]

    if descriptive_cols:
        print("\nPolygon records:")
        print(
            zones[descriptive_cols]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print(f"Columns:          {list(zones.columns)}")
        
def plot_delaunay_with_filter_zones(
    nodes_projected: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> None:
    """
    Plot the current Delaunay network together with the geographic polygons
    intended for later water-crossing filtering.

    This function does not remove any edges. It is purely diagnostic.

    Parameters
    ----------
    nodes_projected:
        Geographic N490 nodes in the projected analysis CRS.
    edges:
        Current Delaunay edge set, preferably after length filtering.
    zones:
        Geographic filter polygons in the same projected CRS.

    Returns
    -------
    None
    """
    if nodes_projected.crs != edges.crs:
        raise ValueError("Node and Delaunay edge CRS do not match.")

    if zones.crs != edges.crs:
        zones = zones.to_crs(edges.crs)

    fig, ax = plt.subplots(figsize=(14, 14))

    # Show the polygon footprint clearly but keep it transparent enough
    # that Delaunay edges remain visible.
    zones.plot(
        ax=ax,
        alpha=0.20,
        edgecolor="black",
        linewidth=0.8,
        label="Filter zones",
    )

    edges.plot(
        ax=ax,
        linewidth=0.8,
        alpha=0.8,
        label="Length-filtered Delaunay",
    )

    nodes_projected.plot(
        ax=ax,
        markersize=7,
        alpha=0.8,
        label="N490 geographic nodes",
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        crs=edges.crs,
    )

    ax.set_title(
        "N490 Delaunay Network and Candidate Geographic Filter Polygons"
    )
    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()
    
def plot_voltage_mst(
    nodes_projected: gpd.GeoDataFrame,
    delaunay_edges: gpd.GeoDataFrame,
    mst_edges: gpd.GeoDataFrame,
    voltage_kv: float,
) -> None:
    """
    Plot the minimum spanning tree on top of the modified Delaunay-1 graph
    for one voltage level.

    Parameters
    ----------
    nodes_projected:
        Geographic nodes belonging to this voltage network.
    delaunay_edges:
        Final modified Delaunay-1 graph after length filtering,
        land-area filtering, and connectivity repair.
    mst_edges:
        Minimum spanning tree derived from ``delaunay_edges``.
    voltage_kv:
        Nominal voltage of the network.

    Returns
    -------
    None
    """
    if nodes_projected.crs != delaunay_edges.crs:
        raise ValueError(
            "Node and Delaunay edge CRS do not match."
        )

    if mst_edges.crs != delaunay_edges.crs:
        raise ValueError(
            "MST and Delaunay edge CRS do not match."
        )

    fig, ax = plt.subplots(figsize=(14, 14))

    # -------------------------------------------------------------
    # Modified Delaunay-1 network
    # -------------------------------------------------------------
    delaunay_edges.plot(
        ax=ax,
        color="gray",
        linewidth=0.7,
        alpha=0.45,
        label="Modified Delaunay-1",
        zorder=2,
    )

    # -------------------------------------------------------------
    # Highlight connectivity-repair edges separately, if present.
    # -------------------------------------------------------------
    if "edge_source" in delaunay_edges.columns:
        repair_edges = delaunay_edges[
            delaunay_edges["edge_source"]
            == "connectivity_repair"
        ]

        if not repair_edges.empty:
            repair_edges.plot(
                ax=ax,
                color="orange",
                linewidth=2.0,
                alpha=0.95,
                label="Connectivity repair",
                zorder=3,
            )

    # -------------------------------------------------------------
    # MST
    # -------------------------------------------------------------
    mst_edges.plot(
        ax=ax,
        color="red",
        linewidth=1.5,
        alpha=0.9,
        label="Minimum spanning tree",
        zorder=4,
    )

    # -------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------
    nodes_projected.plot(
        ax=ax,
        color="black",
        markersize=10,
        alpha=0.8,
        label=f"{voltage_kv:g} kV nodes",
        zorder=5,
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        crs=nodes_projected.crs,
    )

    ax.set_title(
        f"N490 {voltage_kv:g} kV Delaunay Network and MST"
    )

    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_voltage_delaunay_k_sets(
    nodes_projected: gpd.GeoDataFrame,
    delaunay_edges: gpd.GeoDataFrame,
    delaunay_k_sets: dict[int, gpd.GeoDataFrame],
    voltage_kv: float,
) -> None:
    """
    Plot all filtered Delaunay-k candidate sets for one voltage network.

    The modified Delaunay-1 graph is plotted faintly in the background.
    Delaunay-2 and higher candidate sets are then overlaid to permit visual
    inspection of the voltage-specific length and land-area filtering.

    Parameters
    ----------
    nodes_projected:
        Geographic nodes belonging to this voltage network.
    delaunay_edges:
        Final modified Delaunay-1 graph.
    delaunay_k_sets:
        Dictionary of filtered Delaunay-k GeoDataFrames keyed by k.
        Typically contains k = 2, 3, 4, and 5.
    voltage_kv:
        Nominal voltage of the network.

    Returns
    -------
    None
    """
    if nodes_projected.crs != delaunay_edges.crs:
        raise ValueError(
            "Node and Delaunay edge CRS do not match."
        )

    if not delaunay_k_sets:
        raise ValueError(
            "delaunay_k_sets is empty."
        )

    fig, ax = plt.subplots(figsize=(16, 16))

    # -------------------------------------------------------------
    # Delaunay-1 background
    # -------------------------------------------------------------
    delaunay_edges.plot(
        ax=ax,
        color="lightgray",
        linewidth=0.5,
        alpha=0.35,
        label="Delaunay-1",
        zorder=1,
    )

    # -------------------------------------------------------------
    # Higher-order Delaunay sets
    # -------------------------------------------------------------
    styles = {
        2: {
            "color": "blue",
            "linewidth": 0.8,
            "alpha": 0.50,
        },
        3: {
            "color": "green",
            "linewidth": 0.8,
            "alpha": 0.45,
        },
        4: {
            "color": "orange",
            "linewidth": 0.8,
            "alpha": 0.40,
        },
        5: {
            "color": "red",
            "linewidth": 0.8,
            "alpha": 0.35,
        },
    }

    for k in sorted(delaunay_k_sets):

        k_edges = delaunay_k_sets[k]

        if k_edges.empty:
            continue

        style = styles.get(
            k,
            {
                "linewidth": 0.8,
                "alpha": 0.4,
            },
        )

        k_edges.plot(
            ax=ax,
            label=f"Delaunay-{k}",
            zorder=k + 1,
            **style,
        )

    # -------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------
    nodes_projected.plot(
        ax=ax,
        color="black",
        markersize=7,
        alpha=0.75,
        label=f"{voltage_kv:g} kV nodes",
        zorder=10,
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        crs=nodes_projected.crs,
    )

    ax.set_title(
        f"N490 {voltage_kv:g} kV Filtered Delaunay-2+ Candidate Sets"
    )

    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()
    

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

def save_voltage_outputs(
    voltage_kv: float,
    nodes_wgs84: gpd.GeoDataFrame,
    delaunay_edges: gpd.GeoDataFrame,
    mst_edges: gpd.GeoDataFrame,
    repair_edges: gpd.GeoDataFrame,
    delaunay_k_sets: dict[int, gpd.GeoDataFrame],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Save all retained Delaunay products for one voltage network.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    voltage_tag = str(int(voltage_kv))

    nodes_out = nodes_wgs84.copy()

    for col in ["bus_ids", "bus_names"]:
        if col in nodes_out.columns:
            nodes_out[col] = nodes_out[col].apply(
                lambda values: "; ".join(
                    str(value)
                    for value in values
                )
            )

    nodes_out.to_file(
        output_dir
        / f"n490_{voltage_tag}kv_nodes.geojson",
        driver="GeoJSON",
    )

    delaunay_edges.to_crs(SOURCE_CRS).to_file(
        output_dir
        / f"n490_{voltage_tag}kv_delaunay1.geojson",
        driver="GeoJSON",
    )

    mst_edges.to_crs(SOURCE_CRS).to_file(
        output_dir
        / f"n490_{voltage_tag}kv_mst.geojson",
        driver="GeoJSON",
    )

    if not repair_edges.empty:
        repair_edges.to_crs(SOURCE_CRS).to_file(
            output_dir
            / f"n490_{voltage_tag}kv_connectivity_repairs.geojson",
            driver="GeoJSON",
        )

    for k, k_edges in delaunay_k_sets.items():
        if k_edges.empty:
            continue

        k_edges.to_crs(SOURCE_CRS).to_file(
            output_dir
            / f"n490_{voltage_tag}kv_delaunay{k}.geojson",
            driver="GeoJSON",
        )
        


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Build voltage-specific N490 Delaunay networks.

    The 220, 300, and 380 kV systems are treated as independent geographic
    graphs. A substation participates in a voltage network only when N490
    contains a bus at that voltage at that location.

    For each voltage level:

    1. Select buses operating at that voltage.
    2. Construct one geographic node per physical location.
    3. Build the raw Delaunay-1 triangulation.
    4. Apply the voltage-specific maximum-line-length filter.
    5. Apply the Nordic land-area filter.
    6. Repair disconnected components.
    7. Construct the minimum spanning tree.
    8. Construct Delaunay-k candidate sets from the repaired Delaunay-1 graph.
    9. Apply the same length and land-area filters to each Delaunay-k set.
    10. Save all voltage-specific products.
    """
    model = N490(year=2018)

    buses = model.bus.copy()
    lines = model.line.copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    land_area = load_land_area()

    results = {}

    for voltage_kv in VOLTAGE_LEVELS:

        print("\n" + "=" * 72)
        print(f"N490 {voltage_kv:g} kV Delaunay analysis")
        print("=" * 72)

        # ---------------------------------------------------------
        # 1. Voltage-specific node set
        # ---------------------------------------------------------
        nodes_wgs84, nodes_projected = build_voltage_nodes(
            buses=buses,
            voltage_kv=voltage_kv,
        )

        print(
            f"Geographic nodes:               "
            f"{len(nodes_projected)}"
        )

        # ---------------------------------------------------------
        # 2. Raw Delaunay-1
        # ---------------------------------------------------------
        raw_edges = build_raw_delaunay_edges(
            nodes_projected=nodes_projected,
            voltage_kv=voltage_kv,
        )

        # ---------------------------------------------------------
        # 3. Voltage-specific maximum-length filter
        # ---------------------------------------------------------
        max_length_km = get_max_n490_line_length_km(
            lines=lines,
            voltage_kv=voltage_kv,
        )

        length_filtered_edges = (
            filter_delaunay_edges_by_length(
                edges=raw_edges,
                max_length_km=max_length_km,
            )
        )

        print("\nLength filtering")
        print("----------------")
        print(
            f"Longest actual line:            "
            f"{max_length_km:.2f} km"
        )
        print(
            f"Raw Delaunay edges:             "
            f"{len(raw_edges)}"
        )
        print(
            f"After length filtering:         "
            f"{len(length_filtered_edges)}"
        )

        # ---------------------------------------------------------
        # 4. Land-area filter
        # ---------------------------------------------------------
        (
            land_filtered_edges,
            water_removed_edges,
        ) = filter_delaunay_edges_by_land_area(
            edges=length_filtered_edges,
            land_area=land_area,
        )

        print("\nLand-area filtering")
        print("-------------------")
        print(
            f"Edges entering filter:          "
            f"{len(length_filtered_edges)}"
        )
        print(
            f"Edges retained:                 "
            f"{len(land_filtered_edges)}"
        )
        print(
            f"Edges removed:                  "
            f"{len(water_removed_edges)}"
        )

        # ---------------------------------------------------------
        # 5. Connectivity repair
        # ---------------------------------------------------------
        repaired_edges, repair_edges = (
            repair_disconnected_delaunay_nodes(
                nodes_projected=nodes_projected,
                edges=land_filtered_edges,
                voltage_kv=voltage_kv,
            )
        )

        # ---------------------------------------------------------
        # 6. Minimum spanning tree
        # ---------------------------------------------------------
        mst_edges = build_mst_from_delaunay(
            nodes_projected=nodes_projected,
            edges=repaired_edges,
            voltage_kv=voltage_kv,
        )

        print("\nMinimum spanning tree")
        print("---------------------")
        print(
            f"Nodes:                           "
            f"{len(nodes_projected)}"
        )
        print(
            f"Modified Delaunay edges:        "
            f"{len(repaired_edges)}"
        )
        print(
            f"MST edges:                      "
            f"{len(mst_edges)}"
        )
        print(
            f"MST total length:               "
            f"{mst_edges['length_km'].sum():.2f} km"
        )

        # ---------------------------------------------------------
        # 7. Delaunay-k sets
        # ---------------------------------------------------------
        delaunay_k_sets = {}

        for k in DELAUNAY_K_VALUES:

            k_edges_raw = build_delaunay_k_neighbors(
                nodes_projected=nodes_projected,
                delaunay_edges=repaired_edges,
                k=k,
                voltage_kv=voltage_kv,
            )

            k_edges_length = (
                filter_delaunay_edges_by_length(
                    edges=k_edges_raw,
                    max_length_km=max_length_km,
                )
            )

            (
                k_edges_filtered,
                k_edges_outside_land,
            ) = filter_delaunay_edges_by_land_area(
                edges=k_edges_length,
                land_area=land_area,
            )

            delaunay_k_sets[k] = (
                k_edges_filtered
            )

            print(
                f"\n{voltage_kv:g} kV Delaunay-{k}"
            )
            print("-" * 30)
            print(
                f"Raw candidates:                 "
                f"{len(k_edges_raw)}"
            )
            print(
                f"After length filter:            "
                f"{len(k_edges_length)}"
            )
            print(
                f"After land filter:              "
                f"{len(k_edges_filtered)}"
            )

            # Save rejected candidates for diagnostics.
            k_edges_too_long = k_edges_raw[
                k_edges_raw["length_km"]
                > max_length_km
            ].copy()

            voltage_tag = str(
                int(voltage_kv)
            )

            if not k_edges_too_long.empty:
                k_edges_too_long.to_crs(
                    SOURCE_CRS
                ).to_file(
                    OUTPUT_DIR
                    / (
                        f"n490_{voltage_tag}kv_"
                        f"delaunay{k}_removed_length.geojson"
                    ),
                    driver="GeoJSON",
                )

            if not k_edges_outside_land.empty:
                k_edges_outside_land.to_crs(
                    SOURCE_CRS
                ).to_file(
                    OUTPUT_DIR
                    / (
                        f"n490_{voltage_tag}kv_"
                        f"delaunay{k}_removed_land.geojson"
                    ),
                    driver="GeoJSON",
                )

        # ---------------------------------------------------------
        # 8. Save voltage-specific results
        # ---------------------------------------------------------
        save_voltage_outputs(
            voltage_kv=voltage_kv,
            nodes_wgs84=nodes_wgs84,
            delaunay_edges=repaired_edges,
            mst_edges=mst_edges,
            repair_edges=repair_edges,
            delaunay_k_sets=delaunay_k_sets,
        )
        
        # ---------------------------------------------------------
        # 9. Diagnostic plots
        # ---------------------------------------------------------
        plot_voltage_mst(
            nodes_projected=nodes_projected,
            delaunay_edges=repaired_edges,
            mst_edges=mst_edges,
            voltage_kv=voltage_kv,
        )
        
        plot_voltage_delaunay_k_sets(
            nodes_projected=nodes_projected,
            delaunay_edges=repaired_edges,
            delaunay_k_sets=delaunay_k_sets,
            voltage_kv=voltage_kv,
        )

        voltage_tag = str(int(voltage_kv))

        if not water_removed_edges.empty:
            water_removed_edges.to_crs(
                SOURCE_CRS
            ).to_file(
                OUTPUT_DIR
                / (
                    f"n490_{voltage_tag}kv_"
                    "delaunay1_removed_land.geojson"
                ),
                driver="GeoJSON",
            )

        # Keep objects available for later analysis.
        results[voltage_kv] = {
            "nodes_wgs84": nodes_wgs84,
            "nodes_projected": nodes_projected,
            "raw_edges": raw_edges,
            "repaired_edges": repaired_edges,
            "repair_edges": repair_edges,
            "mst_edges": mst_edges,
            "delaunay_k_sets": delaunay_k_sets,
            "max_length_km": max_length_km,
        }

    # -------------------------------------------------------------
    # Cross-voltage summary
    # -------------------------------------------------------------
    print("\n")
    print("=" * 72)
    print("Voltage-specific Delaunay summary")
    print("=" * 72)

    summary_rows = []

    for voltage_kv, result in results.items():
        summary_rows.append(
            {
                "voltage_kv": voltage_kv,
                "nodes": len(
                    result["nodes_projected"]
                ),
                "delaunay1_edges": len(
                    result["repaired_edges"]
                ),
                "mst_edges": len(
                    result["mst_edges"]
                ),
                "repair_edges": len(
                    result["repair_edges"]
                ),
                "max_line_length_km": (
                    result["max_length_km"]
                ),
                **{
                    f"delaunay{k}_edges": len(
                        result["delaunay_k_sets"][k]
                    )
                    for k in DELAUNAY_K_VALUES
                },
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(
        summary
        .round(2)
        .to_string(index=False)
    )

    summary.to_csv(
        OUTPUT_DIR
        / "n490_delaunay_summary_by_voltage.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
    