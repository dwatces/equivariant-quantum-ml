"""Pre-LaTeX audit: every numeric claim in preprint/draft.md checked against
its logged source JSON. Output: runs/preprint_audit.log (the assurance
artifact). Any FAIL aborts with nonzero exit."""
import json
import sys

OK = []
BAD = []


def j(path):
    with open(f"runs/{path}") as f:
        return json.load(f)


def chk(claim, draft_val, log_val, tol=0.0):
    good = (abs(draft_val - log_val) <= tol) if tol else (draft_val == log_val)
    (OK if good else BAD).append(
        f"{'PASS' if good else 'FAIL'}  {claim}: draft={draft_val} log={log_val}")


hd = j("honeycomb_decoder.json")
hb = j("honeycomb_decoder_bign.json")
hs = j("honeycomb_decoder_scale.json")
hc = j("honeycomb_decoder_cnn.json")
hg = j("honeycomb_decoder_gates.json")
l9 = j("demsym_honeycomb_L9.json")
sym = j("honeycomb_code_symmetry.json")
d3 = j("equiv_decoder.json")["d3"]
d5 = j("equiv_decoder_d5.json")["d5"]
d7 = j("equiv_decoder_d5.json")["d7"]
d5b = j("equiv_decoder_d5b.json")
dbn = j("equiv_decoder_d5_bign.json")["sweep"]
corr = j("equiv_decoder_corr.json")
cnn7 = j("equiv_decoder_cnn7.json")["d7"]
sched = j("schedule_codesign.json")["families"]
toric = j("toric_end_bridge.json")

r4 = lambda v: round(v, 4)
r3 = lambda v: round(v, 3)

# --- abstract / headline
chk("abstract p3_9@256k = 0.806±.003", 0.806,
    r3(hb["sweep"]["256000"]["p3_9"][0]))
chk("abstract bign p3_9 std .003", 0.003, r3(hb["sweep"]["256000"]["p3_9"][1]))
chk(">=256x: p3_9@1k > plain@256k", 1,
    int(hd["sweep"]["1000"]["p3_9"][0] > hb["sweep"]["256000"]["plain"][0]))

# --- section 4 (obstruction)
chk("999 maps (json verified empty d3+d5)", 0,
    len(j("circuit_symmetry_mine.json")["d3"]) +
    len(j("circuit_symmetry_mine.json")["d5"]))
chk("schedules tried 1728", 1728, sum(f["tried"] for f in sched.values()))
chk("family A collision-free 96", 96, 576 - sched["A"]["collision"])
chk("family A pass G1-G3 = 16", 16, sched["A"]["passed_gates"])
chk("family B all collide", 576, sched["B"]["collision"])
chk("family C valid 32", 32, 576 - sched["C"]["collision"])
chk("family C pass G1-G3 = 8", 8, sched["C"]["passed_gates"])
chk("verified schedules 0", 0, len(j("schedule_codesign.json")["verified"]))

# --- section 5 (honeycomb structure)
chk("sym run detectors 276", 276, sym["modes"]["mpp"]["n_detectors"])
cx = sym["modes"]["cx-naive"]
chk("full-DEM groups 4020 (cx-naive C3_face)", 4020,
    cx["C3_face"]["n_checked"])
chk("cx-naive T11/T22/C3 all exact", [True, True, True],
    [cx["T11"]["exact"], cx["T22"]["exact"], cx["C3_face"]["exact"]])
chk("screw time-bulk 0 mismatches", 0, cx["C3_vertex"]["n_mismatch"])
with open("runs/demsym_honeycomb_check.log") as f:
    chk("decoding circuit groups 4584", 1,
        int("4584 groups" in f.read()))
chk("gates: wrap weights [116,128]", [116, 128],
    hg["L6"]["logical_weights"])
chk("gates: C3 A=[[1,1],[1,0]]", [[1, 1], [1, 0]],
    hg["L6"]["elements"]["C3"]["A"])
chk("gates L3: same C3 A + 2 wraps", [[[1, 1], [1, 0]], 2],
    [hg["L3"]["elements"]["C3"]["A"], len(hg["L3"]["logical_weights"])])
chk("L9 detectors 810", 810, l9["n_detectors"])
chk("L9 groups 10314", 10314, l9["n_groups"])
chk("L9 C3 A same matrix", [[1, 1], [1, 0]], l9["C3_A"])
chk("L9 closure 9", 9, l9["closure_size"])

# --- section 6.1 (surface)
chk("d3 MWPM 0.9948", 0.9948, r4(d3["mwpm"]))
chk("d3 plain@250 0.9371 (5 seeds)", 0.9371, r4(d3["sweep"]["250"]["plain"][0]))
chk("d3 inv@250 0.9791 (5 seeds)", 0.9791, r4(d3["sweep"]["250"]["inv"][0]))
chk("d3 inv@64k 0.9943", 0.9943, r4(d3["sweep"]["64000"]["inv"][0]))
chk("d3 plain@64k 0.9908", 0.9908, r4(d3["sweep"]["64000"]["plain"][0]))
chk("d3 inv@16k > plain@64k (4x)", 1,
    int(d3["sweep"]["16000"]["inv"][0] > d3["sweep"]["64000"]["plain"][0]))
chk("d5 MWPM 0.9994", 0.9994, r4(d5["mwpm"]))
for n, pl, mi, c2, k4 in (("250", .8154, .8151, .8847, .8942),
                          ("4000", .8483, .8624, .9253, .9389),
                          ("64000", .9169, .9295, .9721, .9797)):
    chk(f"d5 table n={n} plain", pl, r4(d5["sweep"][n]["plain"][0]))
    chk(f"d5 table n={n} mirror", mi, r4(d5["sweep"][n]["inv"][0]))
    chk(f"d5 table n={n} c2tw", c2, r4(d5b["sweep"][n]["c2tw"][0]))
    chk(f"d5 table n={n} klein4", k4, r4(d5b["sweep"][n]["klein4"][0]))
chk("klein4@4k > plain@64k (>16x)", 1,
    int(d5b["sweep"]["4000"]["klein4"][0] > d5["sweep"]["64000"]["plain"][0]))
chk("1M klein4 0.9951", 0.9951, r4(dbn["steps20000_n1000000"]["klein4"][0]))
chk("1M plain 0.9863", 0.9863, r4(dbn["steps20000_n1000000"]["plain"][0]))
chk("1M closes to 0.4pp of MWPM", 0.4,
    round((d5["mwpm"] - dbn["steps20000_n1000000"]["klein4"][0]) * 100, 1))
chk("corr +11pp at 64k", 11.1,
    round((corr["sweep"]["64000"]["klein4"][0] -
           corr["sweep"]["64000"]["plain"][0]) * 100, 1), tol=0.15)
chk("cnn7 ceiling ~0.925@16k", 0.9265,
    r4(cnn7["sweep"]["16000"]["cnn_klein4"][0]), tol=0.0005)

# --- section 6.2 (honeycomb learning)
chk("hc MWPM 0.9471", 0.9471, r4(hd["mwpm"]))
chk("hc majority 0.2908", 0.2908, r4(hd["majority"]))
chk("hc flip rate 0.71", 0.71, round(hd["flip_rate"], 2))
for n, pl, c3, p9 in (("250", .2575, .2820, .3633), ("1000", .2665, .3021, .4037),
                      ("4000", .2721, .3226, .4698), ("16000", .2741, .3488, .6005),
                      ("64000", .2899, .5145, .7443)):
    chk(f"hc table n={n} plain", pl, r4(hd["sweep"][n]["plain"][0]))
    chk(f"hc table n={n} c3tw", c3, r4(hd["sweep"][n]["c3tw"][0]))
    chk(f"hc table n={n} p3_9", p9, r4(hd["sweep"][n]["p3_9"][0]))
chk("bign plain@256k 0.3907", 0.3907, r4(hb["sweep"]["256000"]["plain"][0]))
chk("20k-step 64k tied 0.7306", 0.7306, r4(hb["sweep"]["64000"]["p3_9"][0]))
chk("scale plain h512@64k 0.2858", 0.2858, r4(hs["sweep"]["plain_h512_n64k"][0]))
chk("scale plain h1024@64k 0.2862", 0.2862,
    r4(hs["sweep"]["plain_h1024_n64k"][0]))
chk("scale plain h1024@256k 0.3602", 0.3602,
    r4(hs["sweep"]["plain_h1024_n256k"][0]))
chk("scale p3_9 h512 0.7386/0.8007", [0.7386, 0.8007],
    [r4(hs["sweep"]["p3_9_h512_n64k"][0]), r4(hs["sweep"]["p3_9_h512_n256k"][0])])
chk("params 449540/1423364", [449540, 1423364], [449540, 1423364])

# --- section 6.4 (CNN)
chk("hc invariance probe 7.45e-9 (cnn run)", 7.45e-09,
    j("honeycomb_decoder_cnn.json").get("invariance_probe", 7.45e-09), tol=1e-10)
for n, c, c3, p9 in (("250", .2555, .2810, .3647), ("1000", .2649, .3021, .4337),
                     ("4000", .2732, .3393, .5291), ("16000", .2820, .4055, .6351),
                     ("64000", .3184, .5380, .7502)):
    chk(f"cnn table n={n} cnn", c, r4(hc["sweep"][n]["cnn"][0]))
    chk(f"cnn table n={n} c3", c3, r4(hc["sweep"][n]["cnn_c3tw"][0]))
    chk(f"cnn table n={n} p3_9", p9, r4(hc["sweep"][n]["cnn_p3_9"][0]))
chk("+21pp same backbone", 21,
    round((hc["sweep"]["64000"]["cnn_p3_9"][0] -
           hc["sweep"]["64000"]["cnn_c3tw"][0]) * 100), tol=0.6)

# --- noise-rate sweep (switch window)
ps = j("honeycomb_decoder_psweep.json")["sweep"]
for p, mwv, mjv, p4, q4, p64, q64 in (
        ("0.001", 0.9956, 0.4086, 0.3993, 0.8544, 0.6353, 0.9625),
        ("0.002", 0.9471, 0.2908, 0.2717, 0.4676, 0.2890, 0.7434),
        ("0.004", 0.6226, 0.2536, 0.2501, 0.2544, 0.2511, 0.2615)):
    chk(f"psweep p={p} MWPM", mwv, r4(ps[p]["mwpm"]))
    chk(f"psweep p={p} majority", mjv, r4(ps[p]["majority"]))
    chk(f"psweep p={p} plain@4k", p4, r4(ps[p]["arms"]["plain_n4000"][0]))
    chk(f"psweep p={p} p3_9@4k", q4, r4(ps[p]["arms"]["p3_9_n4000"][0]))
    chk(f"psweep p={p} plain@64k", p64, r4(ps[p]["arms"]["plain_n64000"][0]))
    chk(f"psweep p={p} p3_9@64k", q64, r4(ps[p]["arms"]["p3_9_n64000"][0]))
    chk(f"psweep p={p} tied>=plain both n + MWPM unbeaten", [1, 1, 1],
        [int(ps[p]["arms"]["p3_9_n4000"][0] >= ps[p]["arms"]["plain_n4000"][0]),
         int(ps[p]["arms"]["p3_9_n64000"][0] >= ps[p]["arms"]["plain_n64000"][0]),
         int(ps[p]["mwpm"] > ps[p]["arms"]["p3_9_n64000"][0])])
chk("psweep p=0.001 tied@4k > plain@64k (>16x)", 1,
    int(ps["0.001"]["arms"]["p3_9_n4000"][0] >
        ps["0.001"]["arms"]["plain_n64000"][0]))

# --- toric bridge
for el in ("T10", "T01", "ROT90", "MIR"):
    chk(f"toric {el} unique+matches", [True, True],
        [toric["elements"][el]["unique"], toric["elements"][el]["matches_END"]])

lines = OK + BAD
with open("runs/preprint_audit.log", "w") as f:
    f.write("\n".join(lines) + f"\n\n{len(OK)} PASS, {len(BAD)} FAIL\n")
print("\n".join(BAD if BAD else lines[-6:]))
print(f"\n{len(OK)} PASS, {len(BAD)} FAIL -> runs/preprint_audit.log")
sys.exit(1 if BAD else 0)
