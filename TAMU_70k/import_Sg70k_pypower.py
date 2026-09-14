# -*- coding: utf-8 -*-

"""
convert_ACTIVSg70k_to_pypower.py

Convert the Texas A&M ACTIVSg70k MATPOWER .m case into a native
PYPOWER Python case file.

Expected directory
------------------
/Users/geoffreydesena/Documents/N490/TAMU_70k/

    case_ACTIVSg70k.m
    convert_ACTIVSg70k_to_pypower.py

Output
------
    case_ACTIVSg70k_ppc.py

Requirements
------------
    pip install PYPOWER matpowercaseframes

Approach
--------
The MATPOWER .m file is parsed with matpowercaseframes. The standard
MATPOWER matrices are then copied directly into a PYPOWER case dict:

    baseMVA
    bus
    gen
    branch
    gencost

This avoids converting branches into pandapower line / transformer /
impedance element types.

The resulting dictionary is saved using PYPOWER's savecase(), producing
a native PYPOWER Python case file that can subsequently be loaded with
PYPOWER loadcase().
"""

from pathlib import Path

import numpy as np

from matpowercaseframes import CaseFrames
from pypower.loadcase import loadcase
from pypower.savecase import savecase


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

MATPOWER_FILE = SCRIPT_DIR / "case_ACTIVSg70k.m"
OUTPUT_FILE = SCRIPT_DIR / "case_ACTIVSg70k_ppc.mat"


# =============================================================================
# FORMATTING
# =============================================================================

def header(title):
    width = 100
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# =============================================================================
# HELPERS
# =============================================================================

def dataframe_to_array(df, name):
    """
    Convert a CaseFrames pandas DataFrame to a floating-point NumPy array.
    """

    if df is None:
        raise ValueError(f"Required MATPOWER table '{name}' was not found.")

    arr = df.to_numpy(dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} did not convert to a two-dimensional array."
        )

    return arr


def summarize_ppc(ppc):
    """Print a compact structural and operating-point summary."""

    header("PYPOWER CASE SUMMARY")

    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]

    print(f"baseMVA    : {ppc['baseMVA']:,.1f}")
    print(f"Buses      : {bus.shape[0]:,}")
    print(f"Generators : {gen.shape[0]:,}")
    print(f"Branches   : {branch.shape[0]:,}")

    if "gencost" in ppc:
        print(f"Gen costs  : {ppc['gencost'].shape[0]:,}")

    # -------------------------------------------------------------------------
    # Bus types
    # MATPOWER/PYPOWER:
    #   1 = PQ
    #   2 = PV
    #   3 = reference
    #   4 = isolated
    # -------------------------------------------------------------------------

    bus_type = bus[:, 1].astype(int)

    print("\nBus types:")
    print(f"  PQ        : {(bus_type == 1).sum():,}")
    print(f"  PV        : {(bus_type == 2).sum():,}")
    print(f"  Reference : {(bus_type == 3).sum():,}")
    print(f"  Isolated  : {(bus_type == 4).sum():,}")

    # -------------------------------------------------------------------------
    # Voltage levels
    # bus column 10 in MATPOWER = Python index 9
    # -------------------------------------------------------------------------

    base_kv = bus[:, 9]

    unique_kv, counts = np.unique(base_kv, return_counts=True)

    print("\nBus voltage levels:")

    for kv, count in zip(unique_kv[::-1], counts[::-1]):
        pct = 100.0 * count / len(bus)

        print(
            f"  {kv:8.3f} kV : "
            f"{count:>8,d} buses "
            f"({pct:6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Demand
    # bus:
    # Pd = col 3 -> index 2
    # Qd = col 4 -> index 3
    # -------------------------------------------------------------------------

    p_load = bus[:, 2].sum()
    q_load = bus[:, 3].sum()

    print("\nDemand:")
    print(f"  P load    : {p_load:,.2f} MW")
    print(f"  Q load    : {q_load:,.2f} Mvar")

    # -------------------------------------------------------------------------
    # Generation
    # gen:
    # Pg = col 2 -> index 1
    # Qg = col 3 -> index 2
    # status = col 8 -> index 7
    # -------------------------------------------------------------------------

    gen_status = gen[:, 7] > 0

    online_gen = gen[gen_status]

    p_gen = online_gen[:, 1].sum()
    q_gen = online_gen[:, 2].sum()

    print("\nGeneration:")
    print(f"  Online generators : {len(online_gen):,}")
    print(f"  P generation      : {p_gen:,.2f} MW")
    print(f"  Q generation      : {q_gen:,.2f} Mvar")

    # -------------------------------------------------------------------------
    # Imported solved bus state
    # Vm = col 8 -> index 7
    # Va = col 9 -> index 8
    # -------------------------------------------------------------------------

    vm = bus[:, 7]
    va = bus[:, 8]

    print("\nImported MATPOWER bus state:")
    print(f"  Vm min    : {vm.min():.6f} p.u.")
    print(f"  Vm mean   : {vm.mean():.6f} p.u.")
    print(f"  Vm max    : {vm.max():.6f} p.u.")

    print(f"  Va min    : {va.min():.6f} deg")
    print(f"  Va max    : {va.max():.6f} deg")

    # -------------------------------------------------------------------------
    # Branch status
    # branch status = col 11 -> index 10
    # -------------------------------------------------------------------------

    branch_status = branch[:, 10] > 0

    print("\nBranches:")
    print(f"  In service     : {branch_status.sum():,}")
    print(f"  Out of service : {(~branch_status).sum():,}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    header("ACTIVSg70k MATPOWER -> PYPOWER CONVERSION")
    
    SCRIPT_DIR = Path(__file__).resolve().parent

    MATPOWER_FILE = SCRIPT_DIR / "case_ACTIVSg70k.m"
    OUTPUT_FILE = SCRIPT_DIR / "case_ACTIVSg70k_ppc.mat"

    print(f"Input  : {MATPOWER_FILE}")
    print(f"Output : {OUTPUT_FILE}")

    if not MATPOWER_FILE.exists():
        raise FileNotFoundError(
            f"MATPOWER case not found:\n{MATPOWER_FILE}"
        )

    # -------------------------------------------------------------------------
    # Read MATPOWER .m file
    # -------------------------------------------------------------------------

    header("READING MATPOWER CASE")

    print("Parsing MATPOWER .m file with matpowercaseframes...")

    cf = CaseFrames(str(MATPOWER_FILE))

    print("MATPOWER case parsed successfully.")

    # -------------------------------------------------------------------------
    # Inspect what was read
    # -------------------------------------------------------------------------

    print("\nAvailable CaseFrames attributes:")

    for name in [
        "baseMVA",
        "bus",
        "gen",
        "branch",
        "gencost",
    ]:
        value = getattr(cf, name, None)

        if value is None:
            print(f"  {name:<10s}: missing")
        elif hasattr(value, "shape"):
            print(f"  {name:<10s}: shape {value.shape}")
        else:
            print(f"  {name:<10s}: {value}")

    # -------------------------------------------------------------------------
    # Construct PYPOWER dictionary
    # -------------------------------------------------------------------------

    header("BUILDING PYPOWER CASE")

    ppc = {
        "version": "2",
        "baseMVA": float(cf.baseMVA),
        "bus": dataframe_to_array(cf.bus, "bus"),
        "gen": dataframe_to_array(cf.gen, "gen"),
        "branch": dataframe_to_array(cf.branch, "branch"),
    }

    # gencost is optional for ordinary power flow, but retain it if present.
    if getattr(cf, "gencost", None) is not None:
        ppc["gencost"] = dataframe_to_array(
            cf.gencost,
            "gencost",
        )

    summarize_ppc(ppc)

    # =============================================================================
    # SAVE AS PYPOWER/MATPOWER MAT FILE
    # =============================================================================
    
    header("SAVING PYPOWER CASE")
    
    OUTPUT_FILE = SCRIPT_DIR / "case_ACTIVSg70k_ppc.mat"
    
    savecase(str(OUTPUT_FILE), ppc)
    
    print(f"Saved PYPOWER case:\n  {OUTPUT_FILE}")
    
    
    # =============================================================================
    # RELOAD VERIFICATION
    # =============================================================================
    
    header("VERIFYING SAVED CASE")
    
    ppc_check = loadcase(str(OUTPUT_FILE))
    
    if not isinstance(ppc_check, dict):
        raise RuntimeError(
            f"PYPOWER loadcase() failed. Returned: {ppc_check}"
        )
    
    print("PYPOWER successfully reloaded the saved case.")
    
    print(f"Buses      : {ppc_check['bus'].shape[0]:,}")
    print(f"Generators : {ppc_check['gen'].shape[0]:,}")
    print(f"Branches   : {ppc_check['branch'].shape[0]:,}")
    print(f"Gen costs  : {ppc_check['gencost'].shape[0]:,}")
    
    
    # =============================================================================
    # ROUND-TRIP CHECK
    # =============================================================================
    
    header("ROUND-TRIP COMPARISON")
    
    for key in ["bus", "gen", "branch", "gencost"]:
    
        a = ppc[key]
        b = ppc_check[key]
    
        same_shape = a.shape == b.shape
    
        if same_shape:
            max_diff = np.max(np.abs(a - b))
            same_values = np.allclose(
                a,
                b,
                rtol=0.0,
                atol=1e-10,
                equal_nan=True,
            )
        else:
            max_diff = np.nan
            same_values = False
    
        print(
            f"{key:<10s}: "
            f"shape match = {same_shape}, "
            f"value match = {same_values}, "
            f"max abs diff = {max_diff}"
        )


if __name__ == "__main__":
    main()