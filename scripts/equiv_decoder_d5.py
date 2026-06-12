"""
EQUIVARIANT DECODER -- d=5/7 scaling fix (2026-06-11). Follow-up to equiv_decoder.py.

d=5 failure diagnosis: the v1 chi-solver paired mechanisms by (det-set, prob) with a
1e-9 tolerance PER MECHANISM; at d=5 stim's DEM merging creates probability
collisions between DISTINCT mechanisms, producing wrong "unambiguous" equations, so
the solved chi failed full verification.

Fix (group-based equations, no pairing): group DEM mechanisms by det-set v. For the
permutation P, the image group at v2=Pv must satisfy multiset equality
  M(v2) == {(o XOR c, p) : (o,p) in M(v)}   with c = chi . v2 in {0,1}.
Each group independently forces c=0, c=1, leaves it free, or rejects the symmetry.
The forced equations feed GF(2); the FULL verification gate from v1 (group form)
still decides. Same twisted-invariant model, same matched-params protocol.

Output: runs/equiv_decoder_d5.{log,json}
"""
import json
import time
from collections import defaultdict

import numpy as np
import stim
import pymatching
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
OUT = {"predictions": [
    "group-based chi verifies at d=5 (v1 failure was solver pairing, not physics)",
    "inv >= plain at scarce n with the gap largest at small n (as at d=3)",
    "both approach MWPM from below; beating MWPM NOT claimed"]}
NTRAINS = [250, 1000, 4000, 16000, 64000]
NTEST = 200_000
SEEDS = [0, 1, 2, 3, 4]
P = 0.008


def log(m):
    print(m, flush=True)


def dem_errors(c):
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


def group_multiset(group):
    """Canonical multiset of (obs, prob) pairs."""
    return tuple(sorted(group))


def flipped(group):
    return tuple(sorted((o ^ 1, p) for (o, p) in group))


def verified_symmetry_group(c):
    """Group-based twisted-symmetry verification (v2 of the chi solver)."""
    coords = c.get_detector_coordinates()
    nd = c.num_detectors
    X = max(v[0] for v in coords.values())
    Y = max(v[1] for v in coords.values())
    lookup = {tuple(np.round(coords[i], 4)): i for i in range(nd)}
    cands = {"mirror_x": lambda x, y, t: (X - x, y, t),
             "mirror_y": lambda x, y, t: (x, Y - y, t),
             "C2": lambda x, y, t: (X - x, Y - y, t)}
    errs = dem_errors(c)
    by_v = defaultdict(list)
    for v, o, p in errs:
        by_v[v].append((o, p))
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
            log(f"    {name}: no detector bijection")
            continue
        rows, tbits = [], []
        n_free = n_forced = 0
        ok = True
        for v, group in by_v.items():
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = by_v.get(v2)
            if tgt is None:
                log(f"    {name}: image det-set missing -> rejected")
                ok = False
                break
            m_t = group_multiset(tgt)
            c0 = group_multiset(group) == m_t       # c = chi.v2 = 0 consistent
            c1 = flipped(group) == m_t              # c = 1 consistent
            if not c0 and not c1:
                log(f"    {name}: group multiset mismatch -> rejected")
                ok = False
                break
            if c0 and c1:
                n_free += 1
                continue
            n_forced += 1
            row = np.zeros(nd, dtype=np.int64)
            for d in v2:
                row[d] = 1
            rows.append(row)
            tbits.append(0 if c0 else 1)
        if not ok:
            continue
        chi = solve_gf2(np.array(rows), np.array(tbits))
        if chi is None:
            log(f"    {name}: forced equations inconsistent -> rejected")
            continue
        # FULL verification, group form: every det-set group must map exactly.
        ok = True
        for v, group in by_v.items():
            v2 = frozenset(int(perm[d]) for d in v)
            cbit = int(sum(chi[d] for d in v2)) % 2
            want = flipped(group) if cbit else group_multiset(group)
            if group_multiset(by_v[v2]) != want:
                ok = False
                break
        if ok:
            kind = "plain" if not chi.any() else f"TWISTED (|chi|={int(chi.sum())})"
            log(f"    {name}: VERIFIED {kind} (forced {n_forced}, free {n_free} groups)")
            good.append((name, perm, chi))
        else:
            log(f"    {name}: chi failed full group verification -> rejected")
    return good


def make_code(d):
    kw = dict(before_measure_flip_probability=P, before_round_data_depolarization=P)
    kind = "surface_code:unrotated_memory_z"
    c = stim.Circuit.generated(kind, distance=d, rounds=d, **kw)
    log(f"  [{kind} | phenomenological]")
    good = verified_symmetry_group(c)
    if not good:
        raise AssertionError(f"no verified symmetry at d={d}")
    name, perm, chi = good[0]
    assert (perm[perm] == np.arange(c.num_detectors)).all()
    return c, perm, chi, f"{kind}|phenomenological|{name}"


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
    def __init__(self, nd, perm, chi, h=128):
        super().__init__()
        self.mlp = MLP(nd, h)
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("chi", torch.tensor(chi, dtype=torch.float32))

    def forward(self, x):
        x2 = x[:, self.perm]
        s = 1.0 - 2.0 * torch.remainder(x2 @ self.chi, 2.0)
        return 0.5 * (self.mlp(x) + s * self.mlp(x2))


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


for d in (5, 7):
    log(f"\n== surface d={d}: group-based symmetry verification ==")
    try:
        c, perm, chi, symname = make_code(d)
    except AssertionError as e:
        log(f"  SKIPPED d={d}: {e}")
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
    pc = sum(p.numel() for p in MLP(nd).parameters())
    OUT[key] = {"mwpm": mwpm_acc, "majority": float(base), "ndet": nd,
                "symmetry": symname, "params": pc, "sweep": {}}
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
with open("runs/equiv_decoder_d5.json", "w") as f:
    json.dump(OUT, f, indent=1)
log(f"\nDONE in {OUT['runtime_s']:.0f}s -> runs/equiv_decoder_d5.json")
