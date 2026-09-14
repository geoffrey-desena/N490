"""
run_ACTIVSg70k_pypower.py

Reload the converted ACTIVSg70k PYPOWER case, run an AC Newton-Raphson
power flow, and compare the resulting solution against the solved state
supplied in the original MATPOWER case.

Expected directory
------------------
/Users/geoffreydesena/Documents/N490/TAMU_70k/

    case_ACTIVSg70k_ppc.mat
    run_ACTIVSg70k_pypower.py

The saved PYPOWER case retains the original MATPOWER matrices exactly,
including solved quantities:

Bus matrix
----------
VM          column 8
VA          column 9

Generator matrix
----------------
PG          column 2
QG          column 3

Branch matrix
-------------
PF          column 14
QF          column 15
PT          column 16
QT          column 17

The power flow is initialized from the supplied VM and VA values already
contained in the bus matrix.
"""

from pathlib import Path
import time

import numpy as np

from pypower.loadcase import loadcase
from pypower.runpf import runpf
from pypower.ppoption import ppoption

from pypower.idx_bus import (
    BUS_I, BUS_TYPE, PD, QD, VM, VA, BASE_KV,
    PQ, PV, REF, NONE,
)
from pypower.idx_gen import (
    GEN_BUS, PG, QG, QMAX, QMIN, GEN_STATUS,
)
from pypower.idx_brch import (
    BR_STATUS, PF, QF, PT, QT,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

CASE_FILE = SCRIPT_DIR / "case_ACTIVSg70k_ppc.mat"

# Standard Newton-Raphson AC power flow.
PF_ALG = 1

# Do not enforce generator Q limits on this first test. We want to determine
# whether PYPOWER reproduces the supplied solved MATPOWER operating point
# before allowing the solver to alter PV/PQ classifications.
ENFORCE_Q_LIMITS = True

# Suppress PYPOWER's enormous normal output for a 70,000-bus case.
VERBOSE = 1
OUT_ALL = 0


# =============================================================================
# FORMATTING
# =============================================================================

def header(title):
    width = 100
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def fmt(x, decimals=6):
    return f"{x:,.{decimals}f}"


# =============================================================================
# ANGLE COMPARISON
# =============================================================================

def wrapped_angle_difference(a, b):
    """
    Return angular difference a - b wrapped to [-180, 180) degrees.

    This avoids treating, for example, +179 and -179 degrees as differing
    by 358 degrees.
    """

    return (a - b + 180.0) % 360.0 - 180.0


# =============================================================================
# MAIN
# =============================================================================

def main():

    header("ACTIVSg70k PYPOWER AC POWER FLOW")

    print(f"Case file: {CASE_FILE}")

    if not CASE_FILE.exists():
        raise FileNotFoundError(
            f"PYPOWER case not found:\n{CASE_FILE}"
        )

    # =========================================================================
    # LOAD CASE
    # =========================================================================

    header("LOADING PYPOWER CASE")

    ppc = loadcase(str(CASE_FILE))
    
    # Normalize baseMVA loaded from .mat
    ppc["baseMVA"] = float(np.asarray(ppc["baseMVA"]).squeeze())

    if not isinstance(ppc, dict):
        raise RuntimeError(
            f"loadcase() failed. Returned: {ppc}"
        )

    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]

    print(f"Buses      : {bus.shape[0]:,}")
    print(f"Generators : {gen.shape[0]:,}")
    print(f"Branches   : {branch.shape[0]:,}")
    base_mva = float(np.asarray(ppc["baseMVA"]).squeeze())
    print(f"baseMVA    : {base_mva:,.1f}")

    # =========================================================================
    # SAVE ORIGINAL SOLVED STATE
    # =========================================================================

    header("SUPPLIED MATPOWER OPERATING POINT")

    # Make copies because runpf() will modify the case matrices.
    vm_original = bus[:, VM].copy()
    va_original = bus[:, VA].copy()

    pg_original = gen[:, PG].copy()
    qg_original = gen[:, QG].copy()

    pf_original = branch[:, PF].copy()
    qf_original = branch[:, QF].copy()
    pt_original = branch[:, PT].copy()
    qt_original = branch[:, QT].copy()

    print("Bus voltage:")
    print(f"  Vm min    : {vm_original.min():.6f} p.u.")
    print(f"  Vm mean   : {vm_original.mean():.6f} p.u.")
    print(f"  Vm median : {np.median(vm_original):.6f} p.u.")
    print(f"  Vm max    : {vm_original.max():.6f} p.u.")

    print("\nBus angles:")
    print(f"  Va min    : {va_original.min():.6f} deg")
    print(f"  Va max    : {va_original.max():.6f} deg")

    # -------------------------------------------------------------------------
    # Bus types
    # -------------------------------------------------------------------------

    bus_type = bus[:, BUS_TYPE].astype(int)

    print("\nBus types:")
    print(f"  PQ        : {(bus_type == PQ).sum():,}")
    print(f"  PV        : {(bus_type == PV).sum():,}")
    print(f"  Reference : {(bus_type == REF).sum():,}")
    print(f"  Isolated  : {(bus_type == NONE).sum():,}")

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print("\nLoad:")
    print(f"  P         : {bus[:, PD].sum():,.2f} MW")
    print(f"  Q         : {bus[:, QD].sum():,.2f} Mvar")

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    online = gen[:, GEN_STATUS] > 0

    print("\nOnline generation:")
    print(f"  Units     : {online.sum():,}")
    print(f"  P         : {pg_original[online].sum():,.2f} MW")
    print(f"  Q         : {qg_original[online].sum():,.2f} Mvar")

    # =========================================================================
    # PYPOWER OPTIONS
    # =========================================================================

    header("POWER FLOW SETTINGS")

    ppopt = ppoption(
        PF_ALG=PF_ALG,
        ENFORCE_Q_LIMS=ENFORCE_Q_LIMITS,
        PF_TOL=1e-8,
        PF_MAX_IT=50,
        VERBOSE=VERBOSE,
        OUT_ALL=OUT_ALL,
    )

    print("Algorithm             : Newton-Raphson")
    print("Initial Vm/Va         : supplied MATPOWER solved state")
    print(f"Enforce Q limits      : {ENFORCE_Q_LIMITS}")
    print("Tolerance             : 1e-8 p.u.")
    print("Maximum NR iterations : 50")
    print()
    print(
        "Running power flow. For a 70,000-bus case this may take "
        "some time."
    )

    # =========================================================================
    # RUN POWER FLOW
    # =========================================================================

    header("RUNNING AC POWER FLOW")

    start = time.perf_counter()

    results, success = runpf(
        ppc,
        ppopt,
    )

    elapsed = time.perf_counter() - start

    print()
    print(f"Elapsed time : {elapsed:,.2f} seconds")
    print(f"Success      : {bool(success)}")

    if not success:

        header("POWER FLOW FAILED")

        print(
            "PYPOWER did not converge from the supplied MATPOWER "
            "operating point."
        )

        return

    # =========================================================================
    # EXTRACT RESULTS
    # =========================================================================

    bus_result = results["bus"]
    gen_result = results["gen"]
    branch_result = results["branch"]

    vm_result = bus_result[:, VM]
    va_result = bus_result[:, VA]

    pg_result = gen_result[:, PG]
    qg_result = gen_result[:, QG]

    pf_result = branch_result[:, PF]
    qf_result = branch_result[:, QF]
    pt_result = branch_result[:, PT]
    qt_result = branch_result[:, QT]

    # =========================================================================
    # BUS SOLUTION COMPARISON
    # =========================================================================

    header("BUS SOLUTION COMPARISON")

    vm_diff = vm_result - vm_original

    va_diff = wrapped_angle_difference(
        va_result,
        va_original,
    )

    print("Voltage magnitude difference:")
    print(
        f"  Mean absolute : "
        f"{np.mean(np.abs(vm_diff)):.10f} p.u."
    )
    print(
        f"  Median absolute: "
        f"{np.median(np.abs(vm_diff)):.10f} p.u."
    )
    print(
        f"  Maximum absolute: "
        f"{np.max(np.abs(vm_diff)):.10f} p.u."
    )

    print("\nVoltage angle difference (wrapped):")
    print(
        f"  Mean absolute : "
        f"{np.mean(np.abs(va_diff)):.8f} deg"
    )
    print(
        f"  Median absolute: "
        f"{np.median(np.abs(va_diff)):.8f} deg"
    )
    print(
        f"  Maximum absolute: "
        f"{np.max(np.abs(va_diff)):.8f} deg"
    )

    print("\nSolved voltage range:")
    print(
        f"  Vm: "
        f"{vm_result.min():.6f} - "
        f"{vm_result.max():.6f} p.u."
    )

    # =========================================================================
    # GENERATOR COMPARISON
    # =========================================================================

    header("GENERATOR SOLUTION COMPARISON")

    pg_diff = pg_result - pg_original
    qg_diff = qg_result - qg_original

    print("Active power Pg:")
    print(
        f"  Mean absolute difference : "
        f"{np.mean(np.abs(pg_diff[online])):.8f} MW"
    )
    print(
        f"  Maximum absolute difference: "
        f"{np.max(np.abs(pg_diff[online])):.8f} MW"
    )

    print("\nReactive power Qg:")
    print(
        f"  Mean absolute difference : "
        f"{np.mean(np.abs(qg_diff[online])):.8f} Mvar"
    )
    print(
        f"  Maximum absolute difference: "
        f"{np.max(np.abs(qg_diff[online])):.8f} Mvar"
    )

    print("\nSolved online generation:")
    print(
        f"  P total : "
        f"{pg_result[online].sum():,.2f} MW"
    )
    print(
        f"  Q total : "
        f"{qg_result[online].sum():,.2f} Mvar"
    )

    # -------------------------------------------------------------------------
    # Q-limit check
    # -------------------------------------------------------------------------

    qmax = gen[:, QMAX]
    qmin = gen[:, QMIN]

    tol = 1e-5

    above_qmax = (
        online &
        (qg_result > qmax + tol)
    )

    below_qmin = (
        online &
        (qg_result < qmin - tol)
    )

    print("\nGenerator Q limits:")
    print(
        f"  Above Qmax : "
        f"{above_qmax.sum():,}"
    )
    print(
        f"  Below Qmin : "
        f"{below_qmin.sum():,}"
    )
    print(
        f"  Any violation: "
        f"{(above_qmax | below_qmin).sum():,}"
    )

    # =========================================================================
    # BRANCH FLOW COMPARISON
    # =========================================================================

    header("BRANCH FLOW COMPARISON")

    in_service_branch = branch[:, BR_STATUS] > 0

    def report_branch_difference(name, result, original, unit):

        diff = result - original
        diff = diff[in_service_branch]

        print(
            f"{name:<3s}: "
            f"mean abs = {np.mean(np.abs(diff)):12.8f} {unit}, "
            f"max abs = {np.max(np.abs(diff)):12.8f} {unit}"
        )

    report_branch_difference(
        "Pf",
        pf_result,
        pf_original,
        "MW",
    )

    report_branch_difference(
        "Qf",
        qf_result,
        qf_original,
        "Mvar",
    )

    report_branch_difference(
        "Pt",
        pt_result,
        pt_original,
        "MW",
    )

    report_branch_difference(
        "Qt",
        qt_result,
        qt_original,
        "Mvar",
    )

    # =========================================================================
    # SYSTEM ACTIVE-POWER BALANCE
    # =========================================================================

    header("SYSTEM ACTIVE-POWER BALANCE")

    p_load = bus_result[:, PD].sum()
    p_gen = pg_result[online].sum()

    p_losses = (
        pf_result[in_service_branch]
        + pt_result[in_service_branch]
    ).sum()

    print(f"Total generation : {p_gen:,.2f} MW")
    print(f"Total load       : {p_load:,.2f} MW")
    print(f"Branch P losses  : {p_losses:,.2f} MW")
    print(
        f"Gen - load       : "
        f"{p_gen - p_load:,.2f} MW"
    )

    # =========================================================================
    # FINAL ASSESSMENT
    # =========================================================================

    header("FINAL ASSESSMENT")

    vm_mae = np.mean(np.abs(vm_diff))
    va_mae = np.mean(np.abs(va_diff))

    print("PYPOWER converged successfully.")
    print()
    print(f"Mean |delta Vm| : {vm_mae:.10f} p.u.")
    print(f"Mean |delta Va| : {va_mae:.8f} deg")

    if vm_mae < 1e-5 and va_mae < 1e-3:

        print(
            "\nThe PYPOWER solution essentially reproduces the "
            "supplied MATPOWER operating point."
        )

    else:

        print(
            "\nPYPOWER converged, but the resulting state differs "
            "noticeably from the supplied MATPOWER operating point."
        )


if __name__ == "__main__":
    main()