"""
EQUIVARIANT NEURAL DECODERS for the surface code -- experiment #1 of APPLICATIONS_2026.
2026-06-10. Does exact code-symmetry weight-tying make an ML decoder more
SAMPLE-EFFICIENT at matched parameters? (The project's equivariance law -- pays iff
symmetry exact + data scarce -- applied to QEC, where symmetry is exact BY CONSTRUCTION
and real-device syndrome data is the scarce resource.)

PREDICTIONS (logged pre-run): (1) C2-invariant >= plain at scarce n, gap shrinking
with n; (2) both approach the MWPM baseline at large n (beating MWPM is NOT claimed --
that needs correlated/real-device noise, cf. AlphaQubit); (3) effect modest (group
order is only 2).

Setup: stim surface_code:rotated_memory_z, d in {3,5}, rounds=d, circuit-level
depolarizing p=0.008. Label = logical observable flip. Arms (matched params):
  plain : MLP(ndet-128-128-1)
  c2inv : SAME MLP, group-averaged over the exact C2 detector permutation
          (coords (x,y,t) -> (X-x, Y-y, t); asserted to be a bijective involution)
  tconv : time-translation weight-tying (Conv1d over rounds) ~matched params
Baseline: pymatching MWPM from the exact detector error model.
Output: runs/equiv_decoder.{log,json}
"""
import json
import time

import numpy as np
import stim
import pymatching
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
OUT = {"predictions": ["c2inv>=plain at scarce n", "approach MWPM", "modest (|G|=2)"]}
NTRAINS = [250, 1000, 4000, 16000, 64000]
NTEST = 200_000
SEEDS = [0, 1, 2, 3, 4]
P = 0.008


def log(m):
    print(m, flush=True)


def dem_errors(c):
    """Error mechanisms of the exact DEM as (frozenset(dets), obs_bit, prob)."""
    dem = c.detector_error_model(flatten_loops=True)
    errs, off = [], 0
    for inst in dem:
        if inst.type == "shift_detectors":
            t0 = inst.targets_copy()[0]
            off += t0.val if hasattr(t0, "val") else int(t0)
        elif inst.type == "error":
            p = inst.args_copy()[0]
            dets, obs = [], 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val + off)
                elif t.is_logical_observable_id():
                    obs ^= 1
            errs.append((frozenset(dets), obs, round(p, 9)))
    return errs


def solve_gf2(A, b):
    """Solve A chi = b over GF(2); returns chi or None."""
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    piv = []
    r = 0
    for col in range(n):
        rows = np.nonzero(A[r:, col])[0]
        if len(rows) == 0:
            continue
        A[[r, r + rows[0]]] = A[[r + rows[0], r]]
        b[[r, r + rows[0]]] = b[[r + rows[0], r]]
        for rr in range(m):
            if rr != r and A[rr, col]:
                A[rr] ^= A[r]
                b[rr] ^= b[r]
        piv.append(col)
        r += 1
        if r == m:
            break
    if np.any(b[r:]):
        return None
    chi = np.zeros(n, dtype=np.int64)
    for k, col in enumerate(piv):
        chi[col] = b[k]
    return chi


def verified_symmetry_group(c):
    """Spatial transforms forming a TWISTED symmetry of the decoding problem:
    (dets, obs) -> (P dets, obs XOR chi.P dets). The lattice permutation P must map
    the DEM's mechanism multiset onto itself (ignoring obs); the observable twist
    chi is then solved exactly over GF(2) from the obs-bit differences (chi=0 =
    plain invariance). Lesson: the symmetry maps the logical representative to
    representative x stabilizers, whose values are linear in the detector record."""
    coords = c.get_detector_coordinates()
    nd = c.num_detectors
    X = max(v[0] for v in coords.values())
    Y = max(v[1] for v in coords.values())
    lookup = {tuple(np.round(coords[i], 4)): i for i in range(nd)}
    cands = {"mirror_x": lambda x, y, t: (X - x, y, t),
             "mirror_y": lambda x, y, t: (x, Y - y, t),
             "C2": lambda x, y, t: (X - x, Y - y, t)}
    errs = dem_errors(c)
    by_v = {}
    for v, o, p in errs:
        by_v.setdefault(v, []).append((o, p))
    good = []
    for name, f in cands.items():
        perm = np.full(nd, -1, dtype=int)
        ok = True
        for i in range(nd):
            j = lookup.get(tuple(np.round(f(*coords[i]), 4)))
            if j is None:
                ok = False
                break
            perm[i] = j
        if not ok or len(set(perm)) != nd:
            print(f"    {name}: no detector bijection", flush=True)
            continue
        # mechanism-set invariance (ignoring obs) + collect twist equations
        rows, tbits = [], []
        ok = True
        for v, o, p in errs:
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = by_v.get(v2)
            if tgt is None or not any(abs(pp - p) < 1e-9 for (_, pp) in tgt):
                ok = False
                break
            cand_o = [oo for (oo, pp) in tgt if abs(pp - p) < 1e-9]
            if len(set(cand_o)) == 1:  # unambiguous obs at the image -> an equation
                row = np.zeros(nd, dtype=np.int64)
                for d in v2:
                    row[d] = 1
                rows.append(row)
                tbits.append(cand_o[0] ^ o)
        if not ok:
            print(f"    {name}: mechanism multiset NOT preserved -> rejected", flush=True)
            continue
        chi = solve_gf2(np.array(rows), np.array(tbits))
        if chi is None:
            print(f"    {name}: mechanisms map but obs twist NOT linear -> rejected", flush=True)
            continue
        # full exact verification with the solved twist
        full = sorted(((frozenset(int(perm[d]) for d in v),
                        o ^ (int(sum(chi[int(perm[d])] for d in v)) % 2), p)
                       for v, o, p in errs), key=str)
        if full == sorted(errs, key=str):
            kind = "plain" if not chi.any() else f"TWISTED (|chi|={int(chi.sum())})"
            print(f"    {name}: VERIFIED {kind}", flush=True)
            good.append((name, perm, chi))
        else:
            print(f"    {name}: twist solution failed full verification -> rejected", flush=True)
    return good


def make_code(d):
    """Try (circuit, noise-profile) variants until one has a VERIFIED label-preserving
    symmetry. Lessons encoded: (1) rotated_memory_z's C2 twists the observable;
    (2) CIRCUIT-LEVEL noise breaks spatial symmetry via the extraction schedule's CNOT
    ordering (hook errors) -> phenomenological noise (data depol + measurement flips)
    is the honest v1 regime; schedule-aware group actions = the research frontier."""
    profiles = {
        "circuit-level": dict(after_clifford_depolarization=P,
                              before_measure_flip_probability=P,
                              after_reset_flip_probability=P,
                              before_round_data_depolarization=P),
        "phenomenological": dict(before_measure_flip_probability=P,
                                 before_round_data_depolarization=P),
    }
    for kind in ("surface_code:unrotated_memory_z", "surface_code:rotated_memory_z"):
        for pname, kw in profiles.items():
            c = stim.Circuit.generated(kind, distance=d, rounds=d, **kw)
            print(f"  [{kind} | {pname}]", flush=True)
            good = verified_symmetry_group(c)
            if good:
                name, perm, chi = good[0]
                assert (perm[perm] == np.arange(c.num_detectors)).all()
                return c, perm, chi, f"{kind}|{pname}|{name}"
    raise AssertionError("no verified symmetry in any (circuit, noise) variant")


def sample(c, n, seed):
    s = c.compile_detector_sampler(seed=seed)
    dets, obs = s.sample(n, separate_observables=True)
    return dets.astype(np.float32), obs[:, 0].astype(np.float32)


class MLP(nn.Module):
    def __init__(self, nd, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


class TwistedInv(nn.Module):
    """Exactly (twisted-)invariant model at IDENTICAL params to MLP:
    f(x) = 1/2 [ MLP(x) + s(x) MLP(Px) ],  s(x) = (-1)^{chi . Px}.
    The transformed record Px is a valid sample whose label is obs XOR chi.Px, so the
    sign converts the second branch's logit back to the original label's frame."""
    def __init__(self, nd, perm, chi, h=128):
        super().__init__()
        self.mlp = MLP(nd, h)
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("chi", torch.tensor(chi, dtype=torch.float32))

    def forward(self, x):
        x2 = x[:, self.perm]
        s = 1.0 - 2.0 * torch.remainder(x2 @ self.chi, 2.0)
        return 0.5 * (self.mlp(x) + s * self.mlp(x2))


class TConv(nn.Module):
    """Time-translation weight-tying: Conv1d over rounds, then pooled head."""
    def __init__(self, order, nt, ns, h=128):
        super().__init__()
        self.register_buffer("order", torch.tensor(order, dtype=torch.long))
        self.nt, self.ns = nt, ns
        self.conv = nn.Sequential(nn.Conv1d(ns, h, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        z = x[:, self.order].reshape(-1, self.nt, self.ns).transpose(1, 2)
        z = self.conv(z).mean(dim=2)
        return self.head(z)[:, 0]


def train_eval(model, Xtr, ytr, Xte, yte, steps=4000, bs=256, lr=1e-3):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=DEV)
    ytr_t = torch.tensor(ytr, device=DEV)
    n = len(ytr)
    for s in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=DEV)
        opt.zero_grad()
        loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward()
        opt.step()
    model.eval()
    accs = []
    with torch.no_grad():
        for i in range(0, len(yte), 50000):
            xb = torch.tensor(Xte[i:i + 50000], device=DEV)
            pred = (model(xb) > 0).float().cpu().numpy()
            accs.append(pred == yte[i:i + 50000])
    return float(np.mean(np.concatenate(accs)))


for d in (3, 5):
    log(f"\n== surface d={d}: candidate symmetry verification ==")
    try:
        c, perm, chi, symname = make_code(d)
    except AssertionError as e:
        log(f"  SKIPPED d={d}: {e} (twist must exist by the stabilizer-telescoping "
            f"argument -> solver-robustness TODO, not physics)")
        OUT[f"d{d}"] = {"skipped": str(e)}
        continue
    nd = c.num_detectors
    log(f"  using {symname} ({nd} detectors, fixed pts "
        f"{int(np.sum(perm == np.arange(nd)))})")
    Xte, yte = sample(c, NTEST, seed=999 + d)
    base = max(yte.mean(), 1 - yte.mean())
    m = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True))
    mw = m.decode_batch(Xte.astype(bool))[:, 0]
    mwpm_acc = float(np.mean(mw == yte))
    log(f"  test LER base={1-base:.4f}; MWPM acc={mwpm_acc:.4f}")
    key = f"d{d}"
    OUT[key] = {"mwpm": mwpm_acc, "majority": float(base), "ndet": nd,
                "symmetry": symname, "sweep": {}}
    pc = sum(p.numel() for p in MLP(nd).parameters())
    log(f"  params (identical plain vs inv): {pc}")
    for n in NTRAINS:
        accs = {"plain": [], "inv": []}
        for seed in SEEDS:
            torch.manual_seed(seed)
            Xtr, ytr = sample(c, n, seed=10 * seed + d)
            accs["plain"].append(train_eval(MLP(nd), Xtr, ytr, Xte, yte))
            torch.manual_seed(seed)
            accs["inv"].append(train_eval(TwistedInv(nd, perm, chi), Xtr, ytr, Xte, yte))
        row = {k: (float(np.mean(v)), float(np.std(v))) for k, v in accs.items()}
        OUT[key]["sweep"][n] = row
        log(f"  n={n:6d}: plain {row['plain'][0]:.4f}±{row['plain'][1]:.4f}  "
            f"inv {row['inv'][0]:.4f}±{row['inv'][1]:.4f}   [MWPM {mwpm_acc:.4f}]")

OUT["runtime_s"] = time.time() - T0
with open("runs/equiv_decoder.json", "w") as f:
    json.dump(OUT, f, indent=1)
log(f"\nDONE in {OUT['runtime_s']:.0f}s -> runs/equiv_decoder.json")
