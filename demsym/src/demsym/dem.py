"""Detector-error-model extraction and grouping.

The unit demsym works on is the multiset view of a DEM: error mechanisms
grouped by their detector set, each group holding the multiset of
(observable-mask, probability) pairs. Symmetry is then a statement about
this object — see `solve.solve_action`.

Grouping by detector set (rather than pairing individual mechanisms) is
load-bearing: stim merges error mechanisms that hit identical (detectors,
observables), so per-mechanism pairing across a candidate symmetry breaks on
merge collisions while the per-group multiset comparison does not.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Dem:
    """Grouped detector error model.

    groups: detector-set -> sorted tuple of (obs_mask, prob) pairs, where
    obs_mask packs observable indices as bits (bit i = observable i flipped).
    """
    num_detectors: int
    num_observables: int
    groups: dict[frozenset[int], tuple[tuple[int, float], ...]] = \
        field(default_factory=dict)

    @property
    def num_mechanisms(self) -> int:
        return sum(len(g) for g in self.groups.values())

    @classmethod
    def from_circuit(cls, circuit, prob_decimals: int = 9) -> "Dem":
        """Extract from a stim.Circuit (via its detector error model)."""
        dem = circuit.detector_error_model(flatten_loops=True)
        return cls.from_stim_dem(dem, prob_decimals=prob_decimals)

    @classmethod
    def from_stim_dem(cls, dem, prob_decimals: int = 9) -> "Dem":
        """Extract from a stim.DetectorErrorModel."""
        mechs = []
        off = 0
        for inst in dem.flattened():
            if inst.type == "shift_detectors":
                t0 = inst.targets_copy()[0]
                off += t0.val if hasattr(t0, "val") else int(t0)
            elif inst.type == "error":
                p = inst.args_copy()[0]
                dets, ob = [], 0
                for t in inst.targets_copy():
                    if t.is_relative_detector_id():
                        dets.append(t.val + off)
                    elif t.is_logical_observable_id():
                        ob ^= 1 << t.val
                mechs.append((dets, ob, p))
        return cls.from_mechanisms(mechs, dem.num_detectors,
                                   dem.num_observables,
                                   prob_decimals=prob_decimals)

    @classmethod
    def from_mechanisms(cls, mechanisms: Iterable[tuple], num_detectors: int,
                        num_observables: int,
                        prob_decimals: int = 9) -> "Dem":
        """Build from explicit (detector_ids, obs_mask, probability) triples.

        Use this for code-capacity / hand-built noise models that never pass
        through stim.
        """
        by_v = defaultdict(list)
        for dets, ob, p in mechanisms:
            by_v[frozenset(int(d) for d in dets)].append(
                (int(ob), round(float(p), prob_decimals)))
        groups = {v: tuple(sorted(g)) for v, g in by_v.items()}
        return cls(int(num_detectors), int(num_observables), groups)
