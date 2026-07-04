"""hexlife experiment battery — square vs hex substrate, paired design.

  e1  kernel + wavefront anisotropy calibration (the discretization physics)
  e2  paired soup sweep over (mu, sigma): outcome fractions + anisotropy
  e3  rotation robustness of discovered travelers (survival vs seed angle)
  e4  blind gpt-5-mini judging of rendered finals (labels stripped)
  all e1+e2 (+e3 if travelers found)

Usage: .venv-gpu/bin/python3 hexlife/experiments.py e1|e2|e3|e4|all
Outputs: runs/hexlife_*.json, figures/hexlife_*.png, gallery PNGs.
Honesty: paired params + identical index-space soups across substrates;
Wilson CIs on fractions; nothing summarized before runs complete.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from lenia import (Lenia, classify, device, front_isotropy,  # noqa: E402
                   lattice_harmonics, physical_bump, render, soup)

ROOT = Path(__file__).parent.parent
RUNS, FIGS = ROOT / "runs", ROOT / "figures"
DEV = device()

N, R, STEPS = 128, 10.0, 500
MUS = torch.linspace(0.08, 0.42, 24)
SIGMAS = torch.logspace(math.log10(0.008), math.log10(0.10), 24)


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


# ------------------------------- E1 ----------------------------------------

def e1():
    """Discretization anisotropy, measured lattice-natively (no raster in
    the measurement path): (a) harmonic content of the sampled kernel,
    (b) sector-isotropy of a dynamically expanding front from a PHYSICAL
    Gaussian seed."""
    out = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, name in enumerate(("square", "hex")):
        sim = Lenia(name, n=N, R=R, dev=DEV)
        # (a) kernel harmonics around its own centre (0,0)
        kh = lattice_harmonics(torch.fft.fftshift(sim.kernel).cpu(), sim,
                               center=torch.tensor(
                                   [N * sim.basis[0, 0] / 2
                                    + N * sim.basis[1, 0] / 2,
                                    N * sim.basis[1, 1] / 2]))
        # note: fftshift moves kernel centre to mid-domain for measurement
        # (b) expanding front from a physically circular seed
        A = physical_bump(sim, sigma_r=3.0)
        centre = sim.centroid(A)[0].cpu()
        A, _, _ = sim.run(A, [0.15], [0.05], 25, record_every=25)
        iso, rbar = front_isotropy(A[0].cpu(), sim, centre)
        fh = lattice_harmonics(A[0].cpu(), sim, center=centre)
        out[name] = {"kernel": kh, "front": {"sector_std_over_mean": iso,
                                             **fh}}
        kimg = render(torch.fft.fftshift(sim.kernel).cpu(), sim, px=256)
        fimg = render(A[0], sim, px=256)
        axes[col][0].imshow(kimg, cmap="magma")
        axes[col][0].set_title(f"{name} kernel  c4={kh['c4']:.4f} "
                               f"c6={kh['c6']:.4f}")
        axes[col][1].imshow(fimg, cmap="viridis")
        axes[col][1].set_title(f"{name} front  iso={iso:.4f}")
        axes[col][2].plot(rbar, ".-")
        axes[col][2].set_title("front mean radius per sector")
        axes[col][2].set_ylim(0, max(rbar) * 1.3)
    for ax in axes.flat[:4]:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("E1 — discretization anisotropy (lattice-native metrics)")
    fig.tight_layout()
    fig.savefig(FIGS / "hexlife_e1_isotropy.png", dpi=110)
    (RUNS / "hexlife_e1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ------------------------------- E2 ----------------------------------------

def sweep_one(name: str, seed: int):
    sim = Lenia(name, n=N, R=R, dev=DEV)
    mus = MUS.repeat_interleave(len(SIGMAS))
    sigs = SIGMAS.repeat(len(MUS))
    B = len(mus)
    A = soup(B, N, seed=seed, dev=DEV)
    A, mh, ch = sim.run(A, mus, sigs, STEPS, record_every=20)
    drift = sim.drift(ch)
    outcomes = classify(mh, drift)
    # anisotropy of structured finals (lattice-native, no raster)
    anis = []
    for b in range(B):
        if outcomes[b] in ("alive", "traveler", "chaotic") \
                and 0.005 < float(mh[b, -1]) < 0.55:
            h = lattice_harmonics(A[b].cpu(), sim)
            anis.append({"i": b, "outcome": outcomes[b], **h})
    return sim, A, mh, drift, outcomes, anis, mus, sigs


def e2(seeds=(11, 12)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = {}
    galleries = {}
    for name in ("square", "hex"):
        agg = {"outcomes": [], "anis": [], "params": None}
        for seed in seeds:
            sim, A, mh, drift, outcomes, anis, mus, sigs = sweep_one(name, seed)
            agg["outcomes"] += outcomes
            agg["anis"] += anis
            agg["params"] = [mus.tolist(), sigs.tolist()]
            if name not in galleries:            # keep finals of seed 1
                galleries[name] = (sim, A.clone(), outcomes, mh.clone())
        n_tot = len(agg["outcomes"])
        stats = {}
        for cls in ("dead", "full", "alive", "traveler", "chaotic"):
            k = agg["outcomes"].count(cls)
            stats[cls] = {"k": k, "n": n_tot, "rate_ci": wilson(k, n_tot)}
        interesting = sum(agg["outcomes"].count(c)
                          for c in ("alive", "traveler", "chaotic"))
        stats["interesting"] = {"k": interesting, "n": n_tot,
                                "rate_ci": wilson(interesting, n_tot)}
        if agg["anis"]:
            c4 = [a["c4"] for a in agg["anis"]]
            c6 = [a["c6"] for a in agg["anis"]]
            stats["anisotropy"] = {
                "n": len(c4),
                "c4_mean": sum(c4) / len(c4),
                "c6_mean": sum(c6) / len(c6)}
        result[name] = stats
        print(f"{name}: " + ", ".join(
            f"{c}={stats[c]['k']}" for c in
            ("dead", "full", "alive", "traveler", "chaotic")))
    (RUNS / "hexlife_e2_sweep.json").write_text(json.dumps(result, indent=1))

    # outcome maps + gallery
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = {"dead": 0, "full": 1, "chaotic": 2, "alive": 3, "traveler": 4}
    for ax, name in zip(axes, ("square", "hex")):
        sim, A, outcomes, mh = galleries[name]
        grid = torch.tensor([order[o] for o in outcomes[:len(MUS) * len(SIGMAS)]]
                            ).reshape(len(MUS), len(SIGMAS))
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=4, aspect="auto",
                       origin="lower",
                       extent=[float(SIGMAS[0]), float(SIGMAS[-1]),
                               float(MUS[0]), float(MUS[-1])])
        ax.set_xscale("log")
        ax.set_xlabel("sigma"), ax.set_ylabel("mu")
        ax.set_title(f"{name} outcomes (0 dead → 4 traveler)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIGS / "hexlife_e2_map.png", dpi=110)

    for name in ("square", "hex"):
        sim, A, outcomes, mh = galleries[name]
        keep = [b for b, o in enumerate(outcomes)
                if o in ("traveler", "alive", "chaotic")][:24]
        if not keep:
            continue
        cols = 6
        rows = (len(keep) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows))
        for ax, b in zip(axs.flat, keep):
            ax.imshow(render(A[b], sim, px=160), cmap="viridis")
            ax.set_title(f"{outcomes[b]} μ={float(MUS[b // len(SIGMAS)]):.2f} "
                         f"σ={float(SIGMAS[b % len(SIGMAS)]):.3f}", fontsize=6)
        for ax in axs.flat:
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"{name} — structured finals (seed {11})")
        fig.tight_layout()
        fig.savefig(FIGS / f"hexlife_e2_gallery_{name}.png", dpi=110)
    print("E2 done → runs/hexlife_e2_sweep.json, figures/hexlife_e2_*.png")


# ------------------------------- E3 ----------------------------------------

def e3(angles=12, steps=300):
    """Rotation robustness: take each substrate's own best travelers from a
    fresh sweep, re-seed rotated by k*(360/angles)°, measure survival."""
    import torch.nn.functional as Fn
    res = {}
    for name in ("square", "hex"):
        sim, A, mh, drift, outcomes, anis, mus, sigs = sweep_one(name, seed=11)
        trav = [b for b, o in enumerate(outcomes) if o == "traveler"]
        trav = sorted(trav, key=lambda b: -float(drift.norm(dim=-1)[b]))[:6]
        if not trav:
            res[name] = {"n_travelers": 0}
            continue
        surv = []
        for b in trav:
            img = render(A[b], sim, px=N * 2)         # fine raster of final
            for k in range(angles):
                th = 2 * math.pi * k / angles
                ct, st = math.cos(th), math.sin(th)
                rot = torch.tensor([[ct, -st, 0.0], [st, ct, 0.0]])
                g = Fn.affine_grid(rot[None], (1, 1, N * 2, N * 2),
                                   align_corners=False)
                rimg = Fn.grid_sample(img[None, None].float(), g,
                                      align_corners=False)[0, 0]
                # sample back onto the lattice (positions → raster pixels)
                from lenia import cell_positions
                pos = cell_positions(N, sim.basis)
                Lx = float(N * sim.basis[0, 0])
                Ly = float(N * sim.basis[1, 1])
                px_i = (pos[..., 0] / Lx * (N * 2 - 1)).long().clamp(0, N * 2 - 1)
                px_j = (pos[..., 1] / Ly * (N * 2 - 1)).long().clamp(0, N * 2 - 1)
                seed_state = rimg[px_j, px_i][None].to(DEV)
                mu = float(mus[b])
                sg = float(sigs[b])
                A2, mh2, ch2 = sim.run(seed_state, [mu], [sg], steps,
                                       record_every=20)
                d2 = sim.drift(ch2)
                cls = classify(mh2, d2)[0]
                surv.append({"b": b, "angle": round(math.degrees(th)),
                             "outcome": cls})
        ok = sum(1 for s in surv if s["outcome"] in ("traveler", "alive"))
        res[name] = {"n_travelers": len(trav), "trials": len(surv),
                     "survived": ok, "rate_ci": wilson(ok, len(surv)),
                     "detail": surv}
        print(f"{name}: {len(trav)} travelers, rotation survival "
              f"{ok}/{len(surv)}")
    (RUNS / "hexlife_e3_rotation.json").write_text(json.dumps(res, indent=1))
    print("E3 done → runs/hexlife_e3_rotation.json")


# ------------------------------- E4 ----------------------------------------

def e4(per_substrate=16):
    """Blind LLM judging: render structured finals from both substrates,
    strip labels, shuffle, ask gpt-5-mini (vision) to score 0-10."""
    import random
    import urllib.request
    sys.path.insert(0, str(ROOT / "consulting" / "aivis"))
    from aivis import azure_env, API_VERSION  # reuse auth
    cfg = json.loads((ROOT / "consulting/aivis/rrea.json").read_text())
    az = azure_env(cfg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = []
    for name in ("square", "hex"):
        sim, A, mh, drift, outcomes, anis, mus, sigs = sweep_one(name, seed=13)
        keep = [b for b, o in enumerate(outcomes)
                if o in ("traveler", "alive", "chaotic")][:per_substrate]
        for b in keep:
            img = render(A[b], sim, px=192).numpy()
            fig = plt.figure(figsize=(2.6, 2.6), dpi=100)
            plt.imshow(img, cmap="viridis")
            plt.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            items.append({"substrate": name,
                          "b64": base64.b64encode(buf.getvalue()).decode()})
    random.Random(7).shuffle(items)

    def judge(b64: str) -> str:
        body = json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text":
                 "This is the final state of a 2D continuous cellular "
                 "automaton. Rate 0-10 how much it shows coherent, lifelike "
                 "self-organized structure (distinct organisms, cells, "
                 "waves with form) versus noise, smear, or uniformity. "
                 "Reply with ONLY the number."},
                {"type": "image_url", "image_url":
                 {"url": f"data:image/png;base64,{b64}"}}]}],
            "max_completion_tokens": 400}).encode()
        req = urllib.request.Request(
            f"{az['endpoint']}/openai/deployments/{az['deployment']}"
            f"/chat/completions?api-version={API_VERSION}",
            data=body, method="POST",
            headers={"Content-Type": "application/json", "api-key": az["key"]})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    # smoke test first
    try:
        t = judge(items[0]["b64"])
        print("vision smoke test reply:", t[:40])
    except Exception as exc:
        print(f"VISION UNSUPPORTED on this deployment — skipping E4: {exc}")
        return
    scores = {"square": [], "hex": []}
    for it in items:
        try:
            txt = judge(it["b64"]).strip()
            val = float("".join(ch for ch in txt.split()[0]
                                if ch.isdigit() or ch == "."))
            scores[it["substrate"]].append(val)
        except Exception as exc:
            print("  judge error:", str(exc)[:80])
    out = {k: {"n": len(v), "mean": sum(v) / len(v) if v else None,
               "scores": v} for k, v in scores.items()}
    (RUNS / "hexlife_e4_judge.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                      for k, v in out.items()}, indent=1))


def sweep_at_R(name: str, Rv: float, seed: int, steps: int = 400):
    sim = Lenia(name, n=N, R=Rv, dev=DEV)
    mus = MUS.repeat_interleave(len(SIGMAS))
    sigs = SIGMAS.repeat(len(MUS))
    A = soup(len(mus), N, seed=seed, dev=DEV)
    A, mh, ch = sim.run(A, mus, sigs, steps, record_every=20)
    drift = sim.drift(ch)
    outcomes = classify(mh, drift)
    return sim, outcomes


def e5(R_list=(2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0), seeds=(11, 12)):
    """THE R-FLOOR EXPERIMENT. E1 showed lattice imprint ∝ 1/R (washed out
    at R=10, where E2 finds substrates identical). Question: as the kernel
    shrinks toward the lattice scale — where square Lenia classically
    degrades, the reason Lenia uses big kernels at all — does the hexagonal
    lattice keep supporting coherent travelers to a lower R? If yes, hex
    lowers the resolution floor of Lenia-class life: same phenomena, fewer
    cells. Paired params + identical soups; front harmonics per R for the
    mechanism curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {"R": list(R_list)}
    for name in ("square", "hex"):
        rows = []
        for Rv in R_list:
            k = n_tot = 0
            for seed in seeds:
                sim, outcomes = sweep_at_R(name, Rv, seed)
                k += sum(1 for o in outcomes if o in ("traveler", "alive"))
                n_tot += len(outcomes)
            # mechanism probe: front harmonic content at this R
            sim = Lenia(name, n=N, R=Rv, dev=DEV)
            A = physical_bump(sim, sigma_r=max(2.0, Rv / 3))
            centre = sim.centroid(A)[0].cpu()
            A, _, _ = sim.run(A, [0.15], [0.05], 25, record_every=25)
            iso, _ = front_isotropy(A[0].cpu(), sim, centre)
            h = lattice_harmonics(A[0].cpu(), sim, center=centre)
            rows.append({"R": Rv, "alive_k": k, "n": n_tot,
                         "rate_ci": wilson(k, n_tot),
                         "front_iso": iso, **h})
            print(f"  {name} R={Rv}: alive {k}/{n_tot} "
                  f"({wilson(k, n_tot)[0]:.3f}), front_iso {iso:.4f}")
        res[name] = rows
    (RUNS / "hexlife_e5_rfloor.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, colr in (("square", "#c0392b"), ("hex", "#1b6d8c")):
        rr = [r["R"] for r in res[name]]
        p = [r["rate_ci"][0] for r in res[name]]
        lo = [r["rate_ci"][1] for r in res[name]]
        hi = [r["rate_ci"][2] for r in res[name]]
        axes[0].plot(rr, p, "o-", color=colr, label=name)
        axes[0].fill_between(rr, lo, hi, color=colr, alpha=0.18)
        axes[1].plot(rr, [r["front_iso"] for r in res[name]], "o-",
                     color=colr, label=name)
    axes[0].set_xlabel("kernel radius R (cells)")
    axes[0].set_ylabel("fraction of param space alive")
    axes[0].set_title("Where does life stop? (paired sweeps, Wilson 95% CI)")
    axes[1].set_xlabel("kernel radius R (cells)")
    axes[1].set_ylabel("front anisotropy (sector std/mean)")
    axes[1].set_title("Mechanism: discretization anisotropy vs R")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "hexlife_e5_rfloor.png", dpi=110)
    print("E5 done → runs/hexlife_e5_rfloor.json, figures/hexlife_e5_rfloor.png")


def spot_order(A_row: torch.Tensor, sim, thresh_frac: float = 0.4):
    """Detect spots (connected components above threshold) and measure
    bond-orientational order of their centroids: ψ4 and ψ6 (soft-matter
    standard), via Delaunay on a 3×3 periodic replication (central copy's
    bonds only). Also per-spot shape harmonics (c4, c6 around each spot's
    own centroid). Returns None if too few spots."""
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import Delaunay
    from lenia import cell_positions, min_image

    arr = A_row.cpu().numpy()
    mx = float(arr.max())
    if mx < 0.05:
        return None
    # connectivity must match the LATTICE's physical adjacency: index-space
    # 4-connectivity is correct on square but fragments hex blobs along the
    # missing axial diagonal (hex neighbors include (i+1,j-1) and (i-1,j+1))
    if sim.substrate == "hex":
        structure = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    lab, nl = ndimage.label(arr > thresh_frac * mx, structure=structure)
    if nl < 25:
        return None
    n = sim.n
    B = sim.basis.numpy()
    coms = ndimage.center_of_mass(arr, lab, range(1, nl + 1))   # (j, i)
    pts = np.array([[i * B[0, 0] + j * B[1, 0],
                     i * B[0, 1] + j * B[1, 1]] for j, i in coms])
    # periodic replication
    T1 = n * B[0]
    T2 = n * B[1]
    reps, owner = [], []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            reps.append(pts + a * T1 + b * T2)
            owner += [(a == 0 and b == 0)] * len(pts)
    P = np.concatenate(reps)
    owner = np.array(owner)
    tri = Delaunay(P)
    # bonds incident to central-copy points
    from collections import defaultdict
    nbrs = defaultdict(set)
    for simplex in tri.simplices:
        for u in simplex:
            for v in simplex:
                if u != v:
                    nbrs[u].add(v)
    psi4, psi6 = [], []
    central_idx = np.where(owner)[0]
    for u in central_idx:
        vs = nbrs[u]
        if len(vs) < 3:
            continue
        d = P[list(vs)] - P[u]
        # Delaunay on periodic copies can include long spurious bonds; keep
        # bonds shorter than 2× the median
        L = np.linalg.norm(d, axis=1)
        d = d[L < 2.0 * np.median(L)]
        th = np.arctan2(d[:, 1], d[:, 0])
        psi4.append(abs(np.exp(1j * 4 * th).mean()))
        psi6.append(abs(np.exp(1j * 6 * th).mean()))
    # per-spot shape harmonics
    pos = cell_positions(n, sim.basis)
    c4s, c6s = [], []
    for k in range(1, nl + 1):
        m = lab == k
        if m.sum() < 6:
            continue
        w = torch.from_numpy(arr * m).float()
        j, i = ndimage.center_of_mass(arr, lab, [k])[0]
        centre = torch.tensor([i * B[0, 0] + j * B[1, 0],
                               i * B[0, 1] + j * B[1, 1]]).float()
        d = min_image(pos - centre.view(1, 1, 2), sim)
        r = d.norm(dim=-1)
        th = torch.atan2(d[..., 1], d[..., 0])
        ww = w * (r > 0.3)
        tot = ww.sum().clamp(min=1e-9)
        c4s.append(float(torch.abs((ww * torch.exp(1j * 4 * th)).sum()) / tot))
        c6s.append(float(torch.abs((ww * torch.exp(1j * 6 * th)).sum()) / tot))
    if not psi4 or not c4s:
        return None
    return {"n_spots": nl,
            "psi4": sum(psi4) / len(psi4), "psi6": sum(psi6) / len(psi6),
            "shape_c4": sum(c4s) / len(c4s), "shape_c6": sum(c6s) / len(c6s)}


def e6(Rv: float = 10.0, seed: int = 11, max_patterns: int = 80):
    """THE SPOT-ORDER EXPERIMENT. E5 says the substrate doesn't change
    WHETHER structure emerges. The galleries say it changes WHAT KIND:
    square substrates grow square/cross spots in grid-locked packings, hex
    substrates rounder spots in triangular packings — a ~1e-3 kernel
    anisotropy amplified into visible order. Quantify: ψ4/ψ6 bond order of
    spot packings + per-spot shape harmonics, PAIRED on the same (mu,sigma)
    alive in both substrates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {}
    for name in ("square", "hex"):
        sim, outcomes = sweep_at_R(name, Rv, seed)
        # need the final states again — rerun with states kept
        simx = Lenia(name, n=N, R=Rv, dev=DEV)
        mus = MUS.repeat_interleave(len(SIGMAS))
        sigs = SIGMAS.repeat(len(MUS))
        A = soup(len(mus), N, seed=seed, dev=DEV)
        A, mh, ch = simx.run(A, mus, sigs, 400, record_every=100)
        alive = [b for b, o in enumerate(outcomes)
                 if o in ("traveler", "alive")]
        data[name] = (simx, A, set(alive))
    both = sorted(data["square"][2] & data["hex"][2])[:max_patterns]
    print(f"paired alive params: {len(both)}")
    rows = []
    for b in both:
        row = {"b": b}
        ok = True
        for name in ("square", "hex"):
            simx, A, _ = data[name]
            so = spot_order(A[b], simx)
            if so is None:
                ok = False
                break
            row[name] = so
        if ok:
            rows.append(row)
    if not rows:
        print("E6: no measurable paired spot phases — nothing to report")
        return
    import statistics as st

    def agg(name, key):
        v = [r[name][key] for r in rows]
        return {"mean": st.mean(v),
                "sd": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}

    # paired differences with sign test
    def paired(key):
        d = [r["hex"][key] - r["square"][key] for r in rows]
        pos = sum(1 for x in d if x > 0)
        return {"mean_diff_hex_minus_square": st.mean(d),
                "hex_greater_k": pos, "n": len(d),
                "sign_ci": wilson(pos, len(d))}

    out = {"R": Rv, "n_paired": len(rows),
           "square": {k: agg("square", k)
                      for k in ("psi4", "psi6", "shape_c4", "shape_c6")},
           "hex": {k: agg("hex", k)
                   for k in ("psi4", "psi6", "shape_c4", "shape_c6")},
           "paired": {k: paired(k)
                      for k in ("psi4", "psi6", "shape_c4", "shape_c6")}}
    (RUNS / "hexlife_e6_spotorder.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_paired", "square", "hex")},
                     indent=1))
    print("paired sign tests:", json.dumps(out["paired"], indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in ((axes[0], "psi6", "bond order ψ6 (hexatic)"),
                           (axes[1], "psi4", "bond order ψ4 (tetratic)")):
        sq = [r["square"][key] for r in rows]
        hx = [r["hex"][key] for r in rows]
        ax.scatter(sq, hx, s=14, alpha=0.6, color="#1b6d8c")
        lim = [min(sq + hx) * 0.95, max(sq + hx) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(f"square substrate {key}")
        ax.set_ylabel(f"hex substrate {key}")
        ax.set_title(title + " — paired params")
    fig.tight_layout()
    fig.savefig(FIGS / "hexlife_e6_spotorder.png", dpi=110)
    print("E6 done → runs/hexlife_e6_spotorder.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("e1", "all"):
        e1()
    if cmd in ("e2", "all"):
        e2()
    if cmd == "e3":
        e3()
    if cmd == "e4":
        e4()
    if cmd == "e5":
        e5()
    if cmd == "e6":
        import sys as _s
        rv = float(_s.argv[2]) if len(_s.argv) > 2 else 10.0
        e6(Rv=rv)
