"""
Compare GA, ACO and Hill Climbing on all 50 testcases using best hyperparameters.
Results saved to result.txt (human-readable table + CSV section for plotting).

Usage:
    python test_algorithm.py               # 9s/algo, all cores
    python test_algorithm.py --time 5      # 5s per algo
    python test_algorithm.py --cores 8     # 8 parallel workers
    python test_algorithm.py --out my.txt  # custom output file
"""

import os
import sys
import time
import random
import argparse
import multiprocessing
from datetime import datetime

from solver_ga import run_ga, evaluate as ga_evaluate, make_decoder as ga_decoder
from solver_aco import run_aco
from finetune_hillclimb import solve as hc_solve

# ───────────────── Best hyperparameters from finetuning ─────────────────────
# HC  : finetune_result.txt  rank-1  (11 wins / 50)
HC_BEST = {"p_relocate": 0.80, "p_swap": 0.10}

# GA  : update dict below after running:  python finetune_ga_aco.py --algo ga
GA_BEST = {"pop_size": 60, "cx_rate": 0.6, "mut_rate": 0.3, "tourn_size": 5}

# ACO : update dict below after running:  python finetune_ga_aco.py --algo aco
ACO_BEST = {"n_ants": 5, "alpha": 2.0, "beta": 3.0, "rho": 0.5,
            "Q": 1000.0, "ls_steps": 300}

# ─────────────────────────────────────────────────────────────────────────────
SEED = 12345
ALGOS = ["GA", "ACO", "HC"]
ROOT = os.path.dirname(os.path.abspath(__file__))
TESTCASE_DIR = os.path.join(ROOT, "testcase")


# ─────────────────────────── Parser ─────────────────────────────────────────

def parse_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        tok = fh.read().split()
    N = int(tok[1]); nd = int(tok[3]); idx = 4
    t = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(nd):
        a, b, d = int(tok[idx]), int(tok[idx + 1]), int(tok[idx + 2])
        t[a][b] = d; idx += 3
    trailer_loc, trailer_time = int(tok[idx + 1]), int(tok[idx + 2]); idx += 3
    m = int(tok[idx + 1]); idx += 2
    truck_depot = {}
    for _ in range(m):
        truck_depot[int(tok[idx])] = int(tok[idx + 1]); idx += 2
    reqs = {}
    while idx < len(tok) and tok[idx] == "REQ":
        rid = int(tok[idx + 1]); size = int(tok[idx + 2])
        reqs[rid] = dict(
            q=(2 if size == 40 else 1),
            a=int(tok[idx + 3]), pa=tok[idx + 4], pdur=int(tok[idx + 5]),
            b=int(tok[idx + 6]), da=tok[idx + 7], ddur=int(tok[idx + 8])
        )
        idx += 9
    return dict(N=N, t=t, trailer_loc=trailer_loc, trailer_time=trailer_time,
                m=m, truck_depot=truck_depot, reqs=reqs)


# ─────────────────── Worker: 1 task = 1 testcase ────────────────────────────

def run_one(args):
    """Run GA, ACO, HC sequentially on one testcase. Return (name, results)."""
    path, time_limit = args
    name = os.path.basename(path)
    data = parse_file(path)
    n_req = len(data["reqs"])
    res = {}

    # ── GA ──────────────────────────────────────────────────────────────────
    random.seed(SEED)
    t0 = time.time()
    assign = run_ga(data, GA_BEST, time_limit, t0)
    decode = ga_decoder(data)
    f1, f2 = ga_evaluate(assign, decode, data["truck_depot"])
    res["GA"] = (f1, f2, time.time() - t0)

    # ── ACO ─────────────────────────────────────────────────────────────────
    random.seed(SEED)
    t0 = time.time()
    assign, score = run_aco(data, ACO_BEST, time_limit, t0)
    if score is None:
        f1, f2 = 10**9, 10**9
    else:
        f1, f2 = score
    res["ACO"] = (f1, f2, time.time() - t0)

    # ── HC ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    f1, f2 = hc_solve(data, HC_BEST["p_relocate"], HC_BEST["p_swap"],
                      time_limit, SEED)
    res["HC"] = (f1, f2, time.time() - t0)

    best_score = min((res[a][0], res[a][1]) for a in ALGOS)
    winners = [a for a in ALGOS if (res[a][0], res[a][1]) == best_score]
    print(f"  [{name}] n_req={n_req:>2}  "
          f"GA=({res['GA'][0]},{res['GA'][1]})  "
          f"ACO=({res['ACO'][0]},{res['ACO'][1]})  "
          f"HC=({res['HC'][0]},{res['HC'][1]})  "
          f"-> best={'+'.join(winners)}", flush=True)
    return name, res, n_req


# ───────────────────────── Format & save ────────────────────────────────────

def natural_key(name):
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def format_results(all_raw, time_limit):
    # all_raw: list of (name, res_dict, n_req)
    all_raw = sorted(all_raw, key=lambda x: natural_key(x[0]))

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    sep = "=" * 100
    lines.append(sep)
    lines.append("ALGORITHM COMPARISON: GA vs ACO vs Hill Climbing")
    lines.append(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Time/algo: {time_limit}s    Seed: {SEED}")
    lines.append(f"GA  : pop_size={GA_BEST['pop_size']}  cx_rate={GA_BEST['cx_rate']}  "
                 f"mut_rate={GA_BEST['mut_rate']}  tourn_size={GA_BEST['tourn_size']}")
    lines.append(f"ACO : n_ants={ACO_BEST['n_ants']}  alpha={ACO_BEST['alpha']}  "
                 f"beta={ACO_BEST['beta']}  rho={ACO_BEST['rho']}  "
                 f"ls_steps={ACO_BEST['ls_steps']}")
    lines.append(f"HC  : p_relocate={HC_BEST['p_relocate']}  "
                 f"p_swap={HC_BEST['p_swap']}  "
                 f"p_reorder={round(1-HC_BEST['p_relocate']-HC_BEST['p_swap'], 2)}")
    lines.append(sep)

    # ── Per-testcase table ───────────────────────────────────────────────────
    col_w = 100
    hdr = (f"{'Testcase':<13} {'nReq':>4} | "
           f"{'GA_F1':>7} {'GA_F2':>8} {'GA_t':>5} | "
           f"{'ACO_F1':>7} {'ACO_F2':>8} {'ACO_t':>5} | "
           f"{'HC_F1':>7} {'HC_F2':>8} {'HC_t':>5} | "
           f"{'BestF1':>7} {'BestF2':>8}  Winner")
    lines.append(hdr)
    lines.append("-" * col_w)

    wins   = {a: 0 for a in ALGOS}
    sum_f1 = {a: 0 for a in ALGOS}
    sum_f2 = {a: 0 for a in ALGOS}
    sum_t  = {a: 0.0 for a in ALGOS}
    n = len(all_raw)

    for name, res, n_req in all_raw:
        ga_f1,  ga_f2,  ga_t  = res["GA"]
        aco_f1, aco_f2, aco_t = res["ACO"]
        hc_f1,  hc_f2,  hc_t  = res["HC"]

        best_score = min((res[a][0], res[a][1]) for a in ALGOS)
        winners = [a for a in ALGOS if (res[a][0], res[a][1]) == best_score]
        for w in winners:
            wins[w] += 1
        for a in ALGOS:
            sum_f1[a] += res[a][0]
            sum_f2[a] += res[a][1]
            sum_t[a]  += res[a][2]

        row = (f"{name:<13} {n_req:>4} | "
               f"{ga_f1:>7} {ga_f2:>8} {ga_t:>4.1f}s | "
               f"{aco_f1:>7} {aco_f2:>8} {aco_t:>4.1f}s | "
               f"{hc_f1:>7} {hc_f2:>8} {hc_t:>4.1f}s | "
               f"{best_score[0]:>7} {best_score[1]:>8}  {'+'.join(winners)}")
        lines.append(row)

    lines.append("-" * col_w)
    lines.append(f"{'AVERAGE':<13} {'':>4} | "
                 f"{sum_f1['GA']/n:>7.0f} {sum_f2['GA']/n:>8.0f} {sum_t['GA']/n:>4.1f}s | "
                 f"{sum_f1['ACO']/n:>7.0f} {sum_f2['ACO']/n:>8.0f} {sum_t['ACO']/n:>4.1f}s | "
                 f"{sum_f1['HC']/n:>7.0f} {sum_f2['HC']/n:>8.0f} {sum_t['HC']/n:>4.1f}s |")
    lines.append(sep)

    # ── Summary ─────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("SUMMARY (wins = best (F1,F2) on testcase; ties counted for all winners)")
    lines.append("-" * 60)
    for a in ALGOS:
        lines.append(f"  {a:5}: {wins[a]:>2} wins / {n}  |  "
                     f"Avg F1 = {sum_f1[a]/n:>8.0f}  "
                     f"Avg F2 = {sum_f2[a]/n:>8.0f}  "
                     f"Avg time = {sum_t[a]/n:.1f}s")
    lines.append("")

    # ── CSV section (for plotting / pandas) ─────────────────────────────────
    lines.append(sep)
    lines.append("CSV")
    lines.append(sep)
    lines.append("testcase,n_req,"
                 "GA_F1,GA_F2,GA_time,"
                 "ACO_F1,ACO_F2,ACO_time,"
                 "HC_F1,HC_F2,HC_time,"
                 "winner,best_F1,best_F2")
    for name, res, n_req in all_raw:
        best_score = min((res[a][0], res[a][1]) for a in ALGOS)
        winners = "+".join(a for a in ALGOS if (res[a][0], res[a][1]) == best_score)
        lines.append(
            f"{name},{n_req},"
            f"{res['GA'][0]},{res['GA'][1]},{res['GA'][2]:.2f},"
            f"{res['ACO'][0]},{res['ACO'][1]},{res['ACO'][2]:.2f},"
            f"{res['HC'][0]},{res['HC'][1]},{res['HC'][2]:.2f},"
            f"{winners},{best_score[0]},{best_score[1]}"
        )

    return "\n".join(lines)


# ─────────────────────────────── Main ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare GA, ACO, Hill Climbing on all testcases")
    parser.add_argument("--time", type=float, default=9.0,
                        help="Time limit per algorithm (s), default 9")
    parser.add_argument("--cores", type=int, default=os.cpu_count(),
                        help="Parallel workers, default all cores")
    parser.add_argument("--out", default="result.txt",
                        help="Output file, default result.txt")
    args = parser.parse_args()

    if not os.path.isdir(TESTCASE_DIR):
        print(f"Cannot find testcase dir: {TESTCASE_DIR}")
        sys.exit(1)

    testcases = sorted(
        os.path.join(TESTCASE_DIR, f)
        for f in os.listdir(TESTCASE_DIR)
        if f.endswith(".txt")
    )
    if not testcases:
        print(f"No .txt files in {TESTCASE_DIR}")
        sys.exit(1)

    n_workers = max(1, args.cores)
    n_tc = len(testcases)
    est_wall = (-(-n_tc // n_workers)) * 3 * args.time

    print("=" * 60)
    print("ALGORITHM COMPARISON: GA vs ACO vs Hill Climbing")
    print(f"  Testcases : {n_tc}")
    print(f"  Time/algo : {args.time}s  (total/testcase: {3*args.time:.0f}s)")
    print(f"  Workers   : {n_workers}")
    print(f"  Est. wall : ~{est_wall/60:.1f} min")
    print("=" * 60)
    print("Running ...\n")

    tasks = [(tc, args.time) for tc in testcases]

    wall_start = time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        all_raw = pool.map(run_one, tasks)
    wall_time = time.time() - wall_start

    text = format_results(all_raw, args.time)
    out_path = os.path.join(ROOT, args.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(f"\nDone in {wall_time:.1f}s  ->  {out_path}")


if __name__ == "__main__":
    main()
