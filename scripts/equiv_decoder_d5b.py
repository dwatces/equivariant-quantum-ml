"""
EQUIVARIANT DECODER d=5 add-on (2026-06-11): exercise the TWISTED branch in training
and test whether MORE symmetry helps at matched params.

The main d=5 run auto-picked mirror_x, which verified PLAIN (chi=0) -- so the twist
machinery was verified but not trained through. Arms here (same matched-param MLP):
  c2tw   : TwistedInv with the C2 element (|chi|=120) -- the genuinely twisted branch
  klein4 : average over the full verified Klein group {id, mirror_x, mirror_y, C2}
           with each element's verified chi sign -- order-4 symmetrization
Compare against the completed run's plain/mirror_x numbers (runs/equiv_decoder_d5.json).
Output: runs/equiv_decoder_d5b.{log,json}
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
NTRAINS = [250, 1000, 4000, 16000, 64000]
NTEST = 200_000
SEEDS = [0, 1, 2, 3, 4]
P = 0.008
D = 5
OUT = {"predictions": [
    "c2tw (twisted) performs like mirror_x plain-inv (both exact order-2 symmetries)",
    "klein4 >= order-2 arms at scarce n (more symmetry, same params)"]}


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


def verified_elements(c):
    """All verified (name, perm, chi) via the group-based solver."""
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

    def ms(g):
        return tuple(sorted(g))

    def fl(g):
        return tuple(sorted((o ^ 1, p) for (o, p) in g))

    good = []
    for name, f in cands.items():
        perm = np.array([lookup[tuple(np.round(f(*coords[i]), 4))] for i in range(nd)])
        rows, tbits = [], []
        ok = True
        for v, group in by_v.items():
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = by_v.get(v2)
            if tgt is None:
                ok = False
                break
            c0 = ms(group) == ms(tgt)
            c1 = fl(group) == ms(tgt)
            if not c0 and not c1:
                ok = False
                break
            if c0 and c1:
                continue
            row = np.zeros(nd, dtype=np.int64)
            for d in v2:
                row[d] = 1
            rows.append(row)
            tbits.append(0 if c0 else 1)
        if not ok:
            continue
        chi = solve_gf2(np.array(rows), np.array(tbits))
        if chi is None:
            continue
        full_ok = all(
            ms(by_v[frozenset(int(perm[d]) for d in v)]) ==
            (fl(g) if int(sum(chi[int(perm[d])] for d in v)) % 2 else ms(g))
            for v, g in by_v.items())
        if full_ok:
            log(f"    {name}: VERIFIED |chi|={int(chi.sum())}")
            good.append((name, perm, chi))
    return good


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


class GroupInv(nn.Module):
    """f(x) = (1/|G|) sum_g s_g(x) MLP(P_g x), s_g(x) = (-1)^{chi_g . P_g x}.
    elements = list of (perm, chi); identity is included explicitly."""
    def __init__(self, nd, elements, h=128):
        super().__init__()
        self.mlp = MLP(nd, h)
        self.perms = nn.ParameterList()
        perms, chis = [], []
        for perm, chi in elements:
            perms.append(torch.tensor(perm, dtype=torch.long))
            chis.append(torch.tensor(chi, dtype=torch.float32))
        self.register_buffer("P", torch.stack(perms))     # (|G|, nd)
        self.register_buffer("C", torch.stack(chis))      # (|G|, nd)

    def forward(self, x):
        outs = []
        for k in range(self.P.shape[0]):
            xk = x[:, self.P[k]]
            s = 1.0 - 2.0 * torch.remainder(xk @ self.C[k], 2.0)
            outs.append(s * self.mlp(xk))
        return torch.stack(outs).mean(dim=0)


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


c = stim.Circuit.generated("surface_code:unrotated_memory_z", distance=D, rounds=D,
                           before_measure_flip_probability=P,
                           before_round_data_depolarization=P)
log(f"== d={D} add-on: twisted C2 + Klein-4 arms ==")
elems = verified_elements(c)
names = [n for n, _, _ in elems]
assert set(names) == {"mirror_x", "mirror_y", "C2"}, names
nd = c.num_detectors
by = {n: (p, ch) for n, p, ch in elems}
ident = (np.arange(nd), np.zeros(nd, dtype=np.int64))
c2_elems = [ident, by["C2"]]
klein_elems = [ident, by["mirror_x"], by["mirror_y"], by["C2"]]

# sanity: branch consistency of the Klein average on random data
with torch.no_grad():
    probe = GroupInv(nd, klein_elems).to(DEV)
    xb = torch.randint(0, 2, (64, nd), device=DEV).float()
    perm, chi = by["C2"]
    xg = xb[:, torch.tensor(perm, device=DEV)]
    sg = 1.0 - 2.0 * torch.remainder(xg @ torch.tensor(chi, dtype=torch.float32, device=DEV), 2.0)
    delta = (probe(xb) - sg * probe(xg)).abs().max().item()
log(f"  Klein-average C2-invariance check on random data: max|f(x)-s f(Px)| = {delta:.2e}")
OUT["klein_invariance_check"] = delta

Xte, yte = sample(c, NTEST, seed=999 + D)
m = pymatching.Matching.from_detector_error_model(
    c.detector_error_model(decompose_errors=True))
mwpm_acc = float(np.mean(m.decode_batch(Xte.astype(bool))[:, 0] == yte))
log(f"  MWPM acc={mwpm_acc:.4f}; params/branch={sum(p.numel() for p in MLP(nd).parameters())}")
OUT["mwpm"] = mwpm_acc
OUT["sweep"] = {}
for n in NTRAINS:
    accs = {"c2tw": [], "klein4": []}
    for seed in SEEDS:
        torch.manual_seed(seed)
        Xtr, ytr = sample(c, n, seed=10 * seed + D)
        accs["c2tw"].append(train_eval(GroupInv(nd, c2_elems), Xtr, ytr, Xte, yte))
        torch.manual_seed(seed)
        accs["klein4"].append(train_eval(GroupInv(nd, klein_elems), Xtr, ytr, Xte, yte))
    row = {k: (float(np.mean(v)), float(np.std(v))) for k, v in accs.items()}
    OUT["sweep"][n] = row
    log(f"  n={n:6d}: c2tw {row['c2tw'][0]:.4f}±{row['c2tw'][1]:.4f}  "
        f"klein4 {row['klein4'][0]:.4f}±{row['klein4'][1]:.4f}")

OUT["runtime_s"] = time.time() - T0
with open("runs/equiv_decoder_d5b.json", "w") as f:
    json.dump(OUT, f, indent=1)
log(f"\nDONE in {OUT['runtime_s']:.0f}s -> runs/equiv_decoder_d5b.json")
