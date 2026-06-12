"""demsym — exact twisted symmetries of stim detector error models.

Given a stim circuit (or hand-built noise model): test candidate detector
permutations for exact twisted symmetry of the DEM, solving the label
action (A, X) over GF(2) — including actions that MIX logical observables
(A != I) and syndrome-dependent corrections (X != 0) — and wrap any
PyTorch backbone into an exactly equivariant decoder.

    import demsym
    dem  = demsym.Dem.from_circuit(circuit)
    perm = demsym.perm_from_coords(circuit, my_coord_map)
    elem = demsym.solve_action(dem, perm, name="C3")   # Element or None
    group = demsym.close([elem], dem)
    model = demsym.nn.TwistedEquivariant(backbone, group)

`None` from solve_action is a real answer: the permutation is not a
twisted symmetry of that DEM. See the README for what this machinery found
on the surface code (obstruction) vs the honeycomb Floquet code (exact
symmetry, with A mixing the logical qubits).
"""
from .dem import Dem
from .element import Element, apply_A, identity
from .gf2 import gl2_matrices, solve_gf2
from .perms import perm_from_coords
from .solve import close, solve_action

from . import nn  # noqa: E402  (torch imported lazily inside)

__version__ = "0.1.0"
__all__ = ["Dem", "Element", "apply_A", "identity", "gl2_matrices",
           "solve_gf2", "perm_from_coords", "solve_action", "close", "nn",
           "__version__"]
