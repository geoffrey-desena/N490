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

DELAUNAY_K_VALUES = [2, 3, 4, 5]

# ---------------------------------------------------------------------
# Calculate a limit for the Delaunay connections
# ---------------------------------------------------------------------

def get_max_n490_line_length_km(
    lines: pd.DataFrame,
) -> float:
    """
    Return the length of the longest N490 AC transmission line.

    Parameters
    ----------
    lines:
        N490 line table. Expected to contain ``lat`` and ``lon`` columns,
        where each row stores the geographic path of one transmission line.

    Returns
    -------
    float
        Maximum N490 line length in kilometers.

    Notes
    -----
    Line length is calculated from the supplied N490 line geometry after
    projection to the same metric CRS used for the Delaunay analysis.

    This value is used only as a provisional upper bound for plausible
    Delaunay candidate lengths.
    """
    line_geoms = []

    for lats, lons in zip(lines["lat"], lines["lon"]):
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
        raise ValueError("Could not construct any valid N490 line geometries.")

    gdf_lines = gpd.GeoDataFrame(
        geometry=line_geoms,
        crs=SOURCE_CRS,
    ).to_crs(TARGET_CRS)

    lengths_km = gdf_lines.geometry.length / 1000.0

    return float(lengths_km.max())


# ---------------------------------------------------------------------
# Step 1: Build geographic node set
# ---------------------------------------------------------------------

def build_geographic_nodes(
    buses: pd.DataFrame,
    source_crs: str = SOURCE_CRS,
    target_crs: str = TARGET_CRS,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build one geographic node per unique N490 bus coordinate.

    Parameters
    ----------
    buses:
        N490 bus table. Must contain ``lat`` and ``lon`` columns.
    source_crs:
        CRS of the latitude/longitude coordinates.
    target_crs:
        Metric CRS used for triangulation and distance calculations.

    Returns
    -------
    nodes_wgs84:
        Geographic nodes in the source CRS.
    nodes_projected:
        The same nodes projected to ``target_crs``.

    Notes
    -----
    N490 can contain multiple buses at the same physical location because a
    substation may contain multiple voltage levels. Those co-located buses are
    collapsed into one geographic node.

    Each output row contains:
    - ``node_id``: stable zero-based geographic-node index
    - ``lat``
    - ``lon``
    - ``n_buses_at_location``
    - ``bus_ids``: tuple of N490 bus IDs at that location
    - ``bus_names``: tuple of N490 bus names at that location
    - ``voltages_kv``: tuple of nominal voltages present at that location
    - ``geometry``
    """
    required = {"lat", "lon"}
    missing = required - set(buses.columns)

    if missing:
        raise ValueError(
            f"N490 bus table is missing required columns: {sorted(missing)}"
        )

    buses_work = buses.copy()

    # Preserve the original N490 bus identifier.
    buses_work["bus_id"] = buses_work.index

    # Remove records without usable coordinates.
    buses_work = buses_work.dropna(subset=["lat", "lon"]).copy()

    if buses_work.empty:
        raise ValueError("No N490 buses have valid latitude/longitude coordinates.")

    # Group buses located at the same physical coordinates.
    grouped_rows = []

    for (lat, lon), group in buses_work.groupby(["lat", "lon"], sort=True):
        bus_ids = tuple(group["bus_id"].tolist())

        if "name" in group.columns:
            bus_names = tuple(group["name"].astype(str).tolist())
        else:
            bus_names = tuple()

        if "Vbase" in group.columns:
            voltages = tuple(
                sorted(
                    pd.to_numeric(group["Vbase"], errors="coerce")
                    .dropna()
                    .astype(float)
                    .unique()
                    .tolist()
                )
            )
        else:
            voltages = tuple()

        grouped_rows.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "n_buses_at_location": len(group),
                "bus_ids": bus_ids,
                "bus_names": bus_names,
                "voltages_kv": voltages,
            }
        )

    nodes_df = pd.DataFrame(grouped_rows).reset_index(drop=True)
    nodes_df["node_id"] = nodes_df.index.astype(int)

    nodes_wgs84 = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["lon"], nodes_df["lat"]),
        crs=source_crs,
    )

    nodes_projected = nodes_wgs84.to_crs(target_crs)

    return nodes_wgs84, nodes_projected


# ---------------------------------------------------------------------
# Step 2: Construct raw Delaunay triangulation
# ---------------------------------------------------------------------

def build_raw_delaunay_edges(
    nodes_projected: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Construct the unmodified Delaunay triangulation.

    Parameters
    ----------
    nodes_projected:
        Geographic nodes in a projected CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per unique undirected Delaunay edge with columns:
        - ``from_node``
        - ``to_node``
        - ``length_km``
        - ``geometry``

    Raises
    ------
    ValueError
        If fewer than three geographic nodes are available.
    """
    if len(nodes_projected) < 3:
        raise ValueError(
            "At least three unique geographic nodes are required "
            "for Delaunay triangulation."
        )

    coords = np.array(
        [(geom.x, geom.y) for geom in nodes_projected.geometry],
        dtype=float,
    )

    triangulation = Delaunay(coords)

    edge_pairs: set[tuple[int, int]] = set()

    for simplex in triangulation.simplices:
        i, j, k = map(int, simplex)

        edge_pairs.add(tuple(sorted((i, j))))
        edge_pairs.add(tuple(sorted((j, k))))
        edge_pairs.add(tuple(sorted((i, k))))

    rows = []

    for from_node, to_node in sorted(edge_pairs):
        p_from = nodes_projected.geometry.iloc[from_node]
        p_to = nodes_projected.geometry.iloc[to_node]

        geom = LineString([p_from, p_to])

        rows.append(
            {
                "from_node": int(from_node),
                "to_node": int(to_node),
                "length_km": float(geom.length / 1000.0),
                "geometry": geom,
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
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Iteratively reconnect disconnected geographic nodes to the main network.

    At each iteration:

    1. Build the graph from the current Delaunay edges.
    2. Identify the largest connected component as the main network.
    3. Consider every node outside the main network.
    4. Find the shortest straight-line connection between any disconnected
       node and any node currently in the main network.
    5. Add only that single edge.
    6. Recompute connectivity and repeat until all nodes are connected.

    Adding one edge per iteration allows nearby disconnected substations to
    join the growing main component successively rather than forcing every
    disconnected node to independently connect to a distant original node.

    Parameters
    ----------
    nodes_projected:
        Geographic node table in a projected metric CRS. ``node_id`` must
        correspond to the endpoint IDs stored in ``edges``.
    edges:
        Current Delaunay edge table after length and land-area filtering.

    Returns
    -------
    repaired_edges:
        Original edge set plus all connectivity-repair edges.
    repair_edges:
        Only the newly added repair edges, in the order they were added.
    """
    if nodes_projected.crs != edges.crs:
        raise ValueError("Node and edge CRS do not match.")

    current = edges.copy().reset_index(drop=True)
    repair_rows = []

    all_node_ids = set(nodes_projected["node_id"].astype(int))

    iteration = 0

    while True:
        iteration += 1

        # ---------------------------------------------------------
        # Build current graph, explicitly including isolated nodes.
        # ---------------------------------------------------------
        graph = nx.Graph()
        graph.add_nodes_from(all_node_ids)

        graph.add_edges_from(
            zip(
                current["from_node"].astype(int),
                current["to_node"].astype(int),
            )
        )

        components = list(nx.connected_components(graph))

        if len(components) <= 1:
            break

        # Largest component is treated as the main network.
        main_component = max(components, key=len)

        disconnected_nodes = all_node_ids - main_component

        # ---------------------------------------------------------
        # Find globally shortest disconnected -> main connection.
        # ---------------------------------------------------------
        best = None

        for u in disconnected_nodes:
            p_u = nodes_projected.geometry.iloc[u]

            for v in main_component:
                p_v = nodes_projected.geometry.iloc[v]

                distance_m = p_u.distance(p_v)

                if best is None or distance_m < best["distance_m"]:
                    best = {
                        "from_node": int(u),
                        "to_node": int(v),
                        "distance_m": float(distance_m),
                    }

        if best is None:
            raise RuntimeError(
                "Could not identify a repair edge for disconnected nodes."
            )

        u = best["from_node"]
        v = best["to_node"]

        geometry = LineString(
            [
                nodes_projected.geometry.iloc[u],
                nodes_projected.geometry.iloc[v],
            ]
        )

        repair_row = {
            "from_node": u,
            "to_node": v,
            "length_km": geometry.length / 1000.0,
            "repair_iteration": iteration,
            "edge_source": "connectivity_repair",
            "geometry": geometry,
        }

        repair_rows.append(repair_row)

        current = pd.concat(
            [
                current,
                gpd.GeoDataFrame(
                    [repair_row],
                    geometry="geometry",
                    crs=edges.crs,
                ),
            ],
            ignore_index=True,
        )

        print(
            f"Repair {iteration}: "
            f"node {u} -> node {v}, "
            f"{geometry.length / 1000.0:.2f} km"
        )

    repair_edges = gpd.GeoDataFrame(
        repair_rows,
        geometry="geometry",
        crs=edges.crs,
    )

    print("\nConnectivity repair")
    print("-------------------")
    print(f"Repair edges added:             {len(repair_edges)}")
    print(f"Final connected components:     1")

    return (
        gpd.GeoDataFrame(current, geometry="geometry", crs=edges.crs),
        repair_edges,
    )

def build_mst_from_delaunay(
    nodes_projected: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Build the minimum spanning tree from the modified Delaunay edge set.

    The input edge set should already have undergone:
    1. maximum-length filtering,
    2. land-area filtering, and
    3. connectivity repair.

    Edge length is used as the MST weight.

    Parameters
    ----------
    nodes_projected:
        Geographic node table in the projected analysis CRS.
    edges:
        Final connected Delaunay edge set.

    Returns
    -------
    geopandas.GeoDataFrame
        Minimum-spanning-tree edges with the original edge attributes
        preserved where possible.
    """
    if nodes_projected.crs != edges.crs:
        raise ValueError("Node and edge CRS do not match.")

    graph = nx.Graph()

    graph.add_nodes_from(
        nodes_projected["node_id"].astype(int)
    )

    for idx, row in edges.iterrows():
        u = int(row["from_node"])
        v = int(row["to_node"])

        graph.add_edge(
            u,
            v,
            weight=float(row["length_km"]),
            edge_index=idx,
        )

    if not nx.is_connected(graph):
        components = nx.number_connected_components(graph)

        raise ValueError(
            "Cannot construct a single MST because the modified Delaunay "
            f"graph has {components} connected components."
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

    mst["mst"] = True
    mst["edge_set"] = "MST"

    # Basic theoretical check: a connected MST must have n - 1 edges.
    expected_edges = len(nodes_projected) - 1

    if len(mst) != expected_edges:
        raise RuntimeError(
            f"MST has {len(mst)} edges; expected {expected_edges} "
            f"for {len(nodes_projected)} nodes."
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
) -> gpd.GeoDataFrame:
    """
    Build the Delaunay-k neighbor candidate set.

    Two nodes are Delaunay-k neighbors when their shortest-path distance in
    the modified Delaunay graph is exactly ``k`` edges.

    Parameters
    ----------
    nodes_projected:
        Geographic node table in the projected analysis CRS.
    delaunay_edges:
        Final modified Delaunay-1 graph after length filtering, land-area
        filtering, and connectivity repair.
    k:
        Desired graph-neighbor distance. Must be >= 2.

    Returns
    -------
    geopandas.GeoDataFrame
        One straight-line candidate edge for every unordered pair of nodes
        whose shortest-path distance in the Delaunay graph is exactly ``k``.

        Columns include:
        - ``from_node``
        - ``to_node``
        - ``length_km``
        - ``delaunay_k``
        - ``edge_set``
        - ``geometry``

    Notes
    -----
    The graph distance is calculated from the modified Delaunay-1 network.
    The resulting Delaunay-k candidate itself is represented by the direct
    straight-line segment between the two nodes.
    """
    if k < 2:
        raise ValueError("k must be >= 2.")

    if nodes_projected.crs != delaunay_edges.crs:
        raise ValueError("Node and Delaunay edge CRS do not match.")

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
            "Modified Delaunay graph must be connected before "
            "Delaunay-k neighbors can be calculated."
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

            candidate_pairs.add((int(source), int(target)))

    rows = []

    for u, v in sorted(candidate_pairs):
        p_u = nodes_projected.geometry.iloc[u]
        p_v = nodes_projected.geometry.iloc[v]

        geometry = LineString([p_u, p_v])

        rows.append(
            {
                "from_node": u,
                "to_node": v,
                "length_km": float(geometry.length / 1000.0),
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
    
def plot_mst(
    nodes_projected: gpd.GeoDataFrame,
    delaunay_edges: gpd.GeoDataFrame,
    mst_edges: gpd.GeoDataFrame,
) -> None:
    """
    Plot the modified Delaunay graph with its minimum spanning tree.
    """
    fig, ax = plt.subplots(figsize=(14, 14))

    delaunay_edges.plot(
        ax=ax,
        linewidth=0.5,
        alpha=0.25,
        label="Modified Delaunay",
    )

    mst_edges.plot(
        ax=ax,
        linewidth=1.5,
        alpha=0.9,
        label="MST",
    )

    nodes_projected.plot(
        ax=ax,
        markersize=6,
        alpha=0.8,
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        crs=nodes_projected.crs,
    )

    ax.set_title("N490 Minimum Spanning Tree")
    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()
    
def plot_delaunay_d2_d3(
    nodes_projected: gpd.GeoDataFrame,
    delaunay_edges: gpd.GeoDataFrame,
    delaunay_k_sets: dict[int, gpd.GeoDataFrame],
) -> None:
    """
    Plot the filtered Delaunay-2 and Delaunay-3 candidate sets.

    The modified Delaunay-1 graph is shown faintly in the background for
    reference.

    Parameters
    ----------
    nodes_projected:
        Geographic nodes in the projected analysis CRS.
    delaunay_edges:
        Final modified Delaunay-1 graph.
    delaunay_k_sets:
        Dictionary of filtered Delaunay-k GeoDataFrames keyed by k.
        Must contain entries for k=2 and k=3.
    """
    if 2 not in delaunay_k_sets:
        raise ValueError("delaunay_k_sets does not contain Delaunay-2.")

    if 3 not in delaunay_k_sets:
        raise ValueError("delaunay_k_sets does not contain Delaunay-3.")

    d2 = delaunay_k_sets[2]
    d3 = delaunay_k_sets[3]

    fig, ax = plt.subplots(figsize=(14, 14))

    # Delaunay-1 background.
    delaunay_edges.plot(
        ax=ax,
        color="gray",
        linewidth=0.4,
        alpha=0.20,
        label="Delaunay-1",
    )

    # Delaunay-2.
    d2.plot(
        ax=ax,
        color="blue",
        linewidth=0.8,
        alpha=0.55,
        label="Delaunay-2",
    )

    # Delaunay-3.
    d3.plot(
        ax=ax,
        color="red",
        linewidth=0.8,
        alpha=0.45,
        label="Delaunay-3",
    )

    nodes_projected.plot(
        ax=ax,
        color="black",
        markersize=5,
        alpha=0.7,
        label="N490 substations",
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        crs=nodes_projected.crs,
    )

    ax.set_title(
        "N490 Filtered Delaunay-2 and Delaunay-3 Candidate Sets"
    )

    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

def save_outputs(
    nodes_wgs84: gpd.GeoDataFrame,
    edges_projected: gpd.GeoDataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Save geographic nodes and raw Delaunay edges as GeoJSON.

    Tuple-valued metadata fields are converted to strings because GeoJSON
    cannot reliably store Python tuples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_out = nodes_wgs84.copy()

    for col in ["bus_ids", "bus_names", "voltages_kv"]:
        nodes_out[col] = nodes_out[col].apply(
            lambda values: "; ".join(str(value) for value in values)
        )

    nodes_out.to_file(
        output_dir / "n490_geographic_nodes.geojson",
        driver="GeoJSON",
    )

    edges_wgs84 = edges_projected.to_crs(SOURCE_CRS)

    edges_wgs84.to_file(
        output_dir / "n490_delaunay_land_filtered.geojson",
        driver="GeoJSON",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the initial N490 geographic Delaunay analysis.

    Steps
    -----
    1. Build unique geographic nodes from N490 buses.
    2. Construct the raw Delaunay triangulation.
    3. Remove Delaunay edges longer than the longest actual N490 line.
    4. Load and visually inspect the geographic polygons intended for
       later water-crossing filtering.
    """
    model = N490(year=2018)

    buses = model.bus.copy()
    lines = model.line.copy()

    # -------------------------------------------------------------
    # 1. Geographic nodes
    # -------------------------------------------------------------
    nodes_wgs84, nodes_projected = build_geographic_nodes(buses)

    # -------------------------------------------------------------
    # 2. Raw Delaunay
    # -------------------------------------------------------------
    raw_edges = build_raw_delaunay_edges(nodes_projected)

    # -------------------------------------------------------------
    # 3. Length filter
    # -------------------------------------------------------------
    max_n490_length_km = get_max_n490_line_length_km(lines)

    filtered_edges = filter_delaunay_edges_by_length(
        edges=raw_edges,
        max_length_km=max_n490_length_km,
    )

    print("\nLength filtering")
    print("----------------")
    print(f"Longest actual N490 line:       {max_n490_length_km:.2f} km")
    print(f"Raw Delaunay edges:             {len(raw_edges)}")
    print(f"Edges after length filtering:   {len(filtered_edges)}")
    print(f"Edges removed for length:       {len(raw_edges) - len(filtered_edges)}")

    if len(raw_edges):
        removed_share = (
            100.0
            * (len(raw_edges) - len(filtered_edges))
            / len(raw_edges)
        )
        print(f"Share removed:                  {removed_share:.2f}%")

    # -------------------------------------------------------------
    # 4. Geographic land-area filter
    # -------------------------------------------------------------
    land_area = load_land_area()

    land_filtered_edges, water_removed_edges = (
        filter_delaunay_edges_by_land_area(
            edges=filtered_edges,
            land_area=land_area,
        )
    )

    print("\nLand-area filtering")
    print("-------------------")
    print(f"Edges entering filter:          {len(filtered_edges)}")
    print(f"Edges retained:                 {len(land_filtered_edges)}")
    print(f"Edges removed:                  {len(water_removed_edges)}")

    if len(filtered_edges):
        removed_share = (
            100.0
            * len(water_removed_edges)
            / len(filtered_edges)
        )
        print(f"Share removed:                  {removed_share:.2f}%")
        
    # -------------------------------------------------------------
    # 5. Connectivity repair
    # -------------------------------------------------------------
    repaired_edges, repair_edges = repair_disconnected_delaunay_nodes(
        nodes_projected=nodes_projected,
        edges=land_filtered_edges,
    )

    print("\nRepair edge details")
    print("-------------------")

    if repair_edges.empty:
        print("No connectivity repairs required.")
    else:
        print(
            repair_edges[
                [
                    "repair_iteration",
                    "from_node",
                    "to_node",
                    "length_km",
                ]
            ]
            .round({"length_km": 2})
            .to_string(index=False)
        )
        
    # -------------------------------------------------------------
    # 6. Minimum spanning tree
    # -------------------------------------------------------------
    mst_edges = build_mst_from_delaunay(
        nodes_projected=nodes_projected,
        edges=repaired_edges,
    )

    print("\nMinimum spanning tree")
    print("---------------------")
    print(f"Geographic nodes:               {len(nodes_projected)}")
    print(f"Modified Delaunay edges:        {len(repaired_edges)}")
    print(f"MST edges:                      {len(mst_edges)}")
    print(
        f"MST total length:               "
        f"{mst_edges['length_km'].sum():.2f} km"
    )

    # -------------------------------------------------------------
    # 7. Delaunay-k neighbor candidate sets
    # -------------------------------------------------------------
    delaunay_k_sets = {}

    for k in DELAUNAY_K_VALUES:

        # Build all node pairs whose shortest-path distance in the
        # modified Delaunay-1 graph is exactly k.
        k_edges_raw = build_delaunay_k_neighbors(
            nodes_projected=nodes_projected,
            delaunay_edges=repaired_edges,
            k=k,
        )

        # ---------------------------------------------------------
        # Apply the same maximum-length rule used for Delaunay-1.
        # ---------------------------------------------------------
        k_edges_length = filter_delaunay_edges_by_length(
            edges=k_edges_raw,
            max_length_km=max_n490_length_km,
        )

        # ---------------------------------------------------------
        # Require the direct candidate segment to remain entirely
        # within the processed Nordic land area.
        # ---------------------------------------------------------
        k_edges_filtered, k_edges_outside_land = (
            filter_delaunay_edges_by_land_area(
                edges=k_edges_length,
                land_area=land_area,
            )
        )

        delaunay_k_sets[k] = k_edges_filtered

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------
        print(f"\nDelaunay-{k}")
        print("-" * (9 + len(str(k))))
        print(f"Raw candidate edges:            {len(k_edges_raw)}")
        print(f"After length filter:            {len(k_edges_length)}")
        print(f"After land-area filter:         {len(k_edges_filtered)}")

        print(
            f"Removed for length:             "
            f"{len(k_edges_raw) - len(k_edges_length)}"
        )

        print(
            f"Removed for land area:          "
            f"{len(k_edges_outside_land)}"
        )

        if not k_edges_filtered.empty:
            print(
                f"Mean retained length:           "
                f"{k_edges_filtered['length_km'].mean():.2f} km"
            )
            print(
                f"Maximum retained length:        "
                f"{k_edges_filtered['length_km'].max():.2f} km"
            )

        # ---------------------------------------------------------
        # Save retained candidate set
        # ---------------------------------------------------------
        k_edges_filtered.to_crs(SOURCE_CRS).to_file(
            OUTPUT_DIR / f"n490_delaunay{k}.geojson",
            driver="GeoJSON",
        )
        
        # Candidates rejected only because they exceed the length limit.
        k_edges_too_long = k_edges_raw[
            k_edges_raw["length_km"] > max_n490_length_km
        ].copy()
        
        if not k_edges_too_long.empty:
            k_edges_too_long.to_crs(SOURCE_CRS).to_file(
                OUTPUT_DIR / f"n490_delaunay{k}_removed_length.geojson",
                driver="GeoJSON",
            )
        
        if not k_edges_outside_land.empty:
            k_edges_outside_land.to_crs(SOURCE_CRS).to_file(
                OUTPUT_DIR / f"n490_delaunay{k}_removed_land_area.geojson",
                driver="GeoJSON",
            )
            
    # -------------------------------------------------------------
    # General diagnostics
    # -------------------------------------------------------------
    print_summary(
        buses=buses,
        nodes=nodes_wgs84,
        edges=repaired_edges,
    )

    save_outputs(
        nodes_wgs84=nodes_wgs84,
        edges_projected=repaired_edges,
    )

    plot_raw_delaunay(
        nodes_projected=nodes_projected,
        edges=repaired_edges,
    )

    plot_delaunay_with_filter_zones(
        nodes_projected=nodes_projected,
        edges=repaired_edges,
        zones=land_area,
    )
    
    plot_mst(
        nodes_projected=nodes_projected,
        delaunay_edges=repaired_edges,
        mst_edges=mst_edges,
    )

    plot_delaunay_d2_d3(
        nodes_projected=nodes_projected,
        delaunay_edges=repaired_edges,
        delaunay_k_sets=delaunay_k_sets,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    water_removed_edges.to_crs(SOURCE_CRS).to_file(
        OUTPUT_DIR / "n490_delaunay_removed_outside_land_area.geojson",
        driver="GeoJSON",
    )
    
    if not repair_edges.empty:
        repair_edges.to_crs(SOURCE_CRS).to_file(
            OUTPUT_DIR / "n490_delaunay_connectivity_repairs.geojson",
            driver="GeoJSON",
        )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mst_edges.to_crs(SOURCE_CRS).to_file(
        OUTPUT_DIR / "n490_delaunay_mst.geojson",
        driver="GeoJSON",
    )
    
    k_edges_filtered.to_crs(SOURCE_CRS).to_file(
        OUTPUT_DIR / f"n490_delaunay{k}.geojson",
        driver="GeoJSON",
    )


if __name__ == "__main__":
    main()
    