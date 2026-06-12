"""Candidate detector permutations from stim detector coordinates."""
from __future__ import annotations

import numpy as np


def perm_from_coords(circuit, fn, decimals: int = 6) -> np.ndarray:
    """Detector permutation induced by a map on detector coordinates.

    fn receives each detector's coordinate tuple (as floats, from
    DETECTOR(...) annotations) and returns the image coordinate tuple.
    Every detector must carry coordinates, and fn must permute the
    coordinate set exactly; raises ValueError otherwise (which is itself
    informative: the map is not a symmetry of the detector layout).

    Tip for periodic layouts: fold coordinates modulo the lattice size
    inside fn.
    """
    coords = circuit.get_detector_coordinates()
    nd = circuit.num_detectors
    if len(coords) != nd or any(len(c) == 0 for c in coords.values()):
        raise ValueError("every detector needs a DETECTOR(...) coordinate "
                         "annotation to use perm_from_coords")

    def key(c):
        return tuple(round(float(x), decimals) for x in c)

    index = {}
    for d, c in coords.items():
        kk = key(c)
        if kk in index:
            raise ValueError(f"two detectors share coordinates {kk}; "
                             "disambiguate (e.g. add a layer coordinate)")
        index[kk] = d
    perm = np.zeros(nd, dtype=np.int64)
    for d, c in coords.items():
        kk = key(fn(tuple(float(x) for x in c)))
        if kk not in index:
            raise ValueError(
                f"image of detector {d} at {key(c)} -> {kk} is not a "
                "detector coordinate; the map does not permute the layout")
        perm[d] = index[kk]
    if len(set(perm.tolist())) != nd:
        raise ValueError("coordinate map is not a bijection on detectors")
    return perm
