"""tangle -- photograph two cables and get an integer with a proof.

`|lk| >= 1` certifies that the two cables cannot be pulled apart while their four ends
stay where they are.  `lk = 0` certifies nothing and says so.  Where the photograph cannot
read a crossing, the achievable set of linking numbers is an interval computable in O(k)
with no enumeration, and the tool either certifies over the whole interval or refuses and
names a crossing to re-shoot.

This module exports the certified layer only -- the part that imports no pixels.  The
imaging layer (trace, crossings, render, overlay, cli) is imported from its own modules.
"""

from .alexander import (
    K_MAX,
    DeterminantRefused,
    build_shadow,
    det_parity_ok,
    det_values,
    determinant,
    is_connected,
)
from .certify import (
    BANNED,
    CERTIFIED,
    CONVENTION,
    EXIT,
    LINKED,
    NOT_CERTIFIED,
    REFUSED,
    SEPARABLE,
    TAU,
    Interval,
    Verdict,
    achievable,
    brute_force_interval,
    certify,
    intersect,
    lk_interval,
    next_crossing,
    over_everywhere,
    parity_ok,
    r_min,
)
from .diagram import SIN_MIN, Branch, Cable, Crossing, Diagram

__version__ = "0.1.0"

__all__ = [
    "BANNED",
    "Branch",
    "CERTIFIED",
    "CONVENTION",
    "Cable",
    "Crossing",
    "DeterminantRefused",
    "Diagram",
    "EXIT",
    "Interval",
    "K_MAX",
    "LINKED",
    "NOT_CERTIFIED",
    "REFUSED",
    "SEPARABLE",
    "SIN_MIN",
    "TAU",
    "Verdict",
    "achievable",
    "brute_force_interval",
    "build_shadow",
    "certify",
    "det_parity_ok",
    "det_values",
    "determinant",
    "intersect",
    "is_connected",
    "lk_interval",
    "next_crossing",
    "over_everywhere",
    "parity_ok",
    "r_min",
    "__version__",
]
