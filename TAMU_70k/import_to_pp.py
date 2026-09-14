# -*- coding: utf-8 -*-

"""
import_to_pp.py

Convert the Texas A&M ACTIVSg70k MATPOWER case to a pandapower network
and print basic network diagnostics.

Expected directory structure
----------------------------
/Users/geoffreydesena/Documents/N490/TAMU_70k/
    import_to_pp.py
    case_ACTIVSg70k.m

Outputs
-------
    case_ACTIVSg70k_pp.p
        Native pandapower pickle representation.

    case_ACTIVSg70k_pp.json
        Pandapower JSON representation.

Requirements
------------
    pip install pandapower matpowercaseframes

Notes
-----
The source case is a MATPOWER version-2 case. pandapower's from_mpc()
converter reads the MATPOWER data and converts branches into appropriate
pandapower elements such as lines, transformers, or impedance elements.

The ACTIVSg70k network is a North American system, so a nominal system
frequency of 60 Hz is used.
"""

from pathlib import Path

import pandas as pd
import pandapower as pp
from pandapower.converter.matpower import from_mpc


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

MATPOWER_FILE = SCRIPT_DIR / "case_ACTIVSg70k.m"

OUTPUT_PICKLE = SCRIPT_DIR / "case_ACTIVSg70k_pp.p"
OUTPUT_JSON = SCRIPT_DIR / "case_ACTIVSg70k_pp.json"

SYSTEM_FREQUENCY_HZ = 60.0


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    """Print a formatted diagnostic section header."""
    width = 90
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_element_count(net, element):
    """Print the number of elements in a pandapower element table."""
    if element in net:
        print(f"{element:<20s}: {len(net[element]):>10,d}")


def summarize_voltage_levels(net):
    """Print number and percentage of buses at each nominal voltage level."""

    print_header("BUS VOLTAGE LEVELS")

    if len(net.bus) == 0:
        print("No buses found.")
        return

    voltage_summary = (
        net.bus.groupby("vn_kv", dropna=False)
        .size()
        .rename("n_buses")
        .reset_index()
        .sort_values("vn_kv", ascending=False)
    )

    voltage_summary["percent"] = (
        100.0 * voltage_summary["n_buses"] / len(net.bus)
    )

    print(
        voltage_summary.to_string(
            index=False,
            formatters={
                "vn_kv": lambda x: f"{x:,.3f}",
                "n_buses": lambda x: f"{x:,d}",
                "percent": lambda x: f"{x:6.2f}%",
            },
        )
    )


def summarize_generators(net):
    """Print counts for the different pandapower generation elements."""

    print_header("GENERATION")

    n_gen = len(net.gen) if "gen" in net else 0
    n_sgen = len(net.sgen) if "sgen" in net else 0
    n_ext_grid = len(net.ext_grid) if "ext_grid" in net else 0

    print(f"{'gen':<30s}: {n_gen:>10,d}")
    print(f"{'sgen':<30s}: {n_sgen:>10,d}")
    print(f"{'ext_grid':<30s}: {n_ext_grid:>10,d}")
    print("-" * 43)

    # ext_grid is normally the MATPOWER reference/slack generator.
    total_sources = n_gen + n_sgen + n_ext_grid

    print(f"{'Total generation-source elements':<30s}: {total_sources:>10,d}")

    if n_gen:
        p_gen = net.gen["p_mw"].sum()
        print(f"\nScheduled gen P        : {p_gen:,.2f} MW")

    if n_sgen:
        p_sgen = net.sgen["p_mw"].sum()
        print(f"Scheduled sgen P       : {p_sgen:,.2f} MW")


def summarize_branches(net):
    """Print counts of AC branch-like pandapower elements."""

    print_header("AC NETWORK ELEMENTS")

    elements = [
        "line",
        "trafo",
        "trafo3w",
        "impedance",
        "switch",
    ]

    for element in elements:
        print_element_count(net, element)

    n_lines = len(net.line) if "line" in net else 0
    n_trafo = len(net.trafo) if "trafo" in net else 0
    n_trafo3w = len(net.trafo3w) if "trafo3w" in net else 0
    n_impedance = len(net.impedance) if "impedance" in net else 0

    total = n_lines + n_trafo + n_trafo3w + n_impedance

    print("-" * 43)
    print(f"{'Total branch-like elements':<20s}: {total:>10,d}")


def summarize_loads(net):
    """Print basic load information."""

    print_header("LOADS")

    n_load = len(net.load) if "load" in net else 0

    print(f"{'Load elements':<30s}: {n_load:>10,d}")

    if n_load:
        print(f"{'Total P load [MW]':<30s}: {net.load.p_mw.sum():>10,.2f}")
        print(f"{'Total Q load [Mvar]':<30s}: {net.load.q_mvar.sum():>10,.2f}")


def summarize_shunts(net):
    """Print shunt counts and installed reactive-power parameters."""

    print_header("SHUNTS")

    n_shunt = len(net.shunt) if "shunt" in net else 0

    print(f"{'Shunt elements':<30s}: {n_shunt:>10,d}")

    if n_shunt:
        print(
            f"{'Sum shunt P [MW]':<30s}: "
            f"{net.shunt.p_mw.sum():>10,.2f}"
        )
        print(
            f"{'Sum shunt Q [Mvar]':<30s}: "
            f"{net.shunt.q_mvar.sum():>10,.2f}"
        )


def summarize_network(net):
    """Print overall pandapower network diagnostics."""

    print_header("PANDAPOWER NETWORK SUMMARY")

    print(f"pandapower version : {pp.__version__}")
    print(f"Network frequency  : {net.f_hz:.1f} Hz")
    print(f"System base power  : {net.sn_mva:.1f} MVA")

    print("\nElement counts:")

    for element in [
        "bus",
        "load",
        "gen",
        "sgen",
        "ext_grid",
        "line",
        "trafo",
        "trafo3w",
        "impedance",
        "shunt",
        "switch",
    ]:
        print_element_count(net, element)

    summarize_voltage_levels(net)
    summarize_generators(net)
    summarize_loads(net)
    summarize_branches(net)
    summarize_shunts(net)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header("ACTIVSg70k MATPOWER -> PANDAPOWER CONVERSION")

    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Input file       : {MATPOWER_FILE}")
    print(f"Frequency        : {SYSTEM_FREQUENCY_HZ:.1f} Hz")

    # -------------------------------------------------------------------------
    # Check input
    # -------------------------------------------------------------------------

    if not MATPOWER_FILE.exists():
        raise FileNotFoundError(
            "\nMATPOWER case was not found.\n"
            f"Expected file:\n    {MATPOWER_FILE}\n"
        )

    # -------------------------------------------------------------------------
    # Convert MATPOWER -> pandapower
    # -------------------------------------------------------------------------

    print("\nLoading MATPOWER case and converting to pandapower...")
    print("For a 70,000-bus network this may require substantial memory.")

    net = from_mpc(
        str(MATPOWER_FILE),
        f_hz=SYSTEM_FREQUENCY_HZ,
        validate_conversion=False,
    )

    print("Conversion completed successfully.")

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    summarize_network(net)

    # -------------------------------------------------------------------------
    # Save pandapower network
    # -------------------------------------------------------------------------

    print_header("SAVING PANDAPOWER NETWORK")

    pp.to_pickle(net, str(OUTPUT_PICKLE))
    print(f"Pickle saved:\n    {OUTPUT_PICKLE}")

    pp.to_json(net, str(OUTPUT_JSON))
    print(f"\nJSON saved:\n    {OUTPUT_JSON}")

    # -------------------------------------------------------------------------
    # Final verification
    # -------------------------------------------------------------------------

    print_header("SAVE VERIFICATION")

    # Reload the pickle to make sure the serialized pandapower case is valid.
    net_check = pp.from_pickle(str(OUTPUT_PICKLE))

    print("Successfully reloaded saved pandapower network.")
    print(f"Buses        : {len(net_check.bus):,}")
    print(f"Lines        : {len(net_check.line):,}")
    print(f"Transformers : {len(net_check.trafo):,}")
    print(f"Generators   : {len(net_check.gen):,}")
    print(f"External grids: {len(net_check.ext_grid):,}")

    print_header("DONE")
    print("ACTIVSg70k has been converted to pandapower.")


if __name__ == "__main__":
    main()