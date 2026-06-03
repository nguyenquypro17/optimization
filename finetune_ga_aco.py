"""
Finetune (grid search) hyperparameter cho 2 thuat toan ACO va GA
tren bo 20 testcase.

Cach lam giong finetune_hillclimb.py:
  - Moi task = 1 testcase (parse 1 lan, chay tat ca config tuan tu ben trong).
  - Song song cac testcase bang multiprocessing.
  - Cham diem moi config theo so lan ve nhat (rank-1) tren cac testcase,
    so sanh (F1, F2) theo thu tu tu dien (lexicographic).
  - Goi truc tiep run_aco / run_ga tu solver_aco.py va solver_ga.py
    => khong copy lai thuat toan, luon dong bo voi solver.

Usage:
    python finetune_ga_aco.py --algo aco            # chi ACO
    python finetune_ga_aco.py --algo ga             # chi GA
    python finetune_ga_aco.py --algo both           # ca hai (mac dinh)
    python finetune_ga_aco.py --algo aco --time 5   # 5s moi lan chay
    python finetune_ga_aco.py --algo ga --cores 8   # 8 worker
"""

import os
import sys
import time
import random
import argparse
import multiprocessing
from itertools import product

# Core thuat toan lay truc tiep tu solver (khong copy lai)
from solver_aco import run_aco, make_decoder as aco_decoder
from solver_ga import run_ga, evaluate as ga_evaluate, make_decoder as ga_decoder


# ─────────────────────────── Hyperparameter grids ──────────────────────────

# ACO: n_ants x alpha x beta x rho (Q va ls_steps giu co dinh de grid vua phai)
ACO_GRID = {
    "n_ants": [5, 10, 20],
    "alpha":  [1.0, 2.0],
    "beta":   [2.0, 3.0],
    "rho":    [0.1, 0.3, 0.5],
}
ACO_FIXED = {"Q": 1000.0, "ls_steps": 300}

# GA: pop_size x cx_rate x mut_rate x tourn_size
GA_GRID = {
    "pop_size":   [20, 40, 60],
    "cx_rate":    [0.6, 0.8],
    "mut_rate":   [0.2, 0.3, 0.4],
    "tourn_size": [2, 3, 5],
}

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTCASE_DIR = os.path.join(ROOT, "testcase")
SEED = 12345


# ─────────────────────────────── Parser ────────────────────────────────────

def parse_file(path):
    """Doc testcase tu file (giong parse() trong solver nhung tu path)."""
    with open(path, "r", encoding="utf-8") as f:
        tok = f.read().split()
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
        reqs[rid] = dict(q=(2 if size == 40 else 1),
                         a=int(tok[idx + 3]), pa=tok[idx + 4], pdur=int(tok[idx + 5]),
                         b=int(tok[idx + 6]), da=tok[idx + 7], ddur=int(tok[idx + 8]))
        idx += 9
    return dict(N=N, t=t, trailer_loc=trailer_loc, trailer_time=trailer_time,
                m=m, truck_depot=truck_depot, reqs=reqs)


# ──────────────────────── Build config list per algo ───────────────────────

def build_configs(algo):
    """Tra ve list cac dict params theo grid cua thuat toan."""
    if algo == "aco":
        keys = list(ACO_GRID.keys())
        combos = product(*(ACO_GRID[k] for k in keys))
        configs = []
        for vals in combos:
            cfg = dict(zip(keys, vals))
            cfg.update(ACO_FIXED)
            configs.append(cfg)
        return configs
    else:  # ga
        keys = list(GA_GRID.keys())
        combos = product(*(GA_GRID[k] for k in keys))
        return [dict(zip(keys, vals)) for vals in combos]


def cfg_label(algo, cfg):
    if algo == "aco":
        return (f"n_ants={cfg['n_ants']:>2} alpha={cfg['alpha']:.1f} "
                f"beta={cfg['beta']:.1f} rho={cfg['rho']:.1f}")
    return (f"pop={cfg['pop_size']:>2} cx={cfg['cx_rate']:.1f} "
            f"mut={cfg['mut_rate']:.1f} tourn={cfg['tourn_size']}")


# ──────────────────── 1 config tren 1 data => (F1, F2) ──────────────────────

def eval_config(algo, data, cfg, time_limit):
    """Chay 1 thuat toan voi 1 config, tra ve (F1, F2). Reseed de on dinh."""
    random.seed(SEED)
    if algo == "aco":
        assign, score = run_aco(data, cfg, time_limit, time.time())
        if assign is None or score is None:
            return (float("inf"), float("inf"))
        return score
    else:  # ga
        assign = run_ga(data, cfg, time_limit, time.time())
        decode = ga_decoder(data)
        return ga_evaluate(assign, decode, data["truck_depot"])


# ─────────────────────── Worker: 1 task = 1 testcase ───────────────────────

def run_testcase(args):
    """Parse 1 lan, chay tat ca config tuan tu, tra ve list (F1, F2)."""
    path, algo, configs, time_limit = args
    t0 = time.time()
    data = parse_file(path)

    scores = [eval_config(algo, data, cfg, time_limit) for cfg in configs]

    name = os.path.basename(path)
    elapsed = time.time() - t0
    best = min(scores)
    best_idx = scores.index(best)
    print(f"  [{name}] done in {elapsed:.1f}s | winner: "
          f"{cfg_label(algo, configs[best_idx])} -> F1={best[0]} F2={best[1]}",
          flush=True)
    return scores


# ──────────────────────── Grid search 1 thuat toan ─────────────────────────

def finetune(algo, testcases, time_limit, n_workers, out_lines):
    configs = build_configs(algo)
    n_tc = len(testcases)
    n_cfg = len(configs)
    est_wall = n_tc * n_cfg * time_limit / n_workers

    header = (f"\n{'='*72}\n"
              f"GRID SEARCH: {algo.upper()}\n"
              f"  Testcases : {n_tc}\n"
              f"  Configs   : {n_cfg}\n"
              f"  Time/run  : {time_limit}s\n"
              f"  Workers   : {n_workers}\n"
              f"  Est. wall : ~{est_wall/60:.1f} min\n"
              f"{'='*72}")
    print(header)
    out_lines.append(header)
    print("Running ...\n")

    tasks = [(tc, algo, configs, time_limit) for tc in testcases]

    wall_start = time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        all_results = pool.map(run_testcase, tasks)
    wall_time = time.time() - wall_start

    # all_results[i][j] = (F1, F2) cho testcase i, config j
    wins = [0] * n_cfg
    points = [0] * n_cfg  # rank-1: +3, rank-2: +2, rank-3: +1
    for i in range(n_tc):
        scores = all_results[i]
        best = min(scores)
        for j, s in enumerate(scores):
            if s == best:
                wins[j] += 1
        ranked = sorted(range(n_cfg), key=lambda j: scores[j])
        for rank, j in enumerate(ranked[:3]):
            points[j] += (3 - rank)

    avg_f1 = [sum(all_results[i][j][0] for i in range(n_tc)) / n_tc
              for j in range(n_cfg)]
    avg_f2 = [sum(all_results[i][j][1] for i in range(n_tc)) / n_tc
              for j in range(n_cfg)]

    ranked_cfgs = sorted(range(n_cfg), key=lambda j: (-wins[j], -points[j]))

    table = []
    table.append("")
    table.append(f"{'Rank':>4}  {'Wins':>5}  {'Pts':>5}  "
                 f"{'AvgF1':>10}  {'AvgF2':>12}  Config")
    table.append("-" * 72)
    for rank, j in enumerate(ranked_cfgs, 1):
        table.append(f"{rank:>4}  {wins[j]:>5}  {points[j]:>5}  "
                     f"{avg_f1[j]:>10.0f}  {avg_f2[j]:>12.0f}  "
                     f"{cfg_label(algo, configs[j])}")
    table.append(f"\nTotal wall time: {wall_time:.1f}s")

    best_j = ranked_cfgs[0]
    best_cfg = configs[best_j]
    table.append("")
    table.append("=" * 72)
    table.append(f"BEST {algo.upper()} CONFIG")
    table.append("=" * 72)
    table.append(f"  {cfg_label(algo, best_cfg)}")
    table.append(f"  Wins = {wins[best_j]}/{n_tc}   Points = {points[best_j]}")
    table.append(f"  dict = {best_cfg!r}")

    text = "\n".join(table)
    print(text)
    out_lines.append(text)


# ───────────────────────────────── Main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["aco", "ga", "both"], default="both",
                        help="Thuat toan can finetune (default: both)")
    parser.add_argument("--time", type=float, default=3.0,
                        help="Time limit moi lan chay (s), default 3")
    parser.add_argument("--cores", type=int, default=os.cpu_count(),
                        help="So worker song song (default: tat ca core)")
    parser.add_argument("--out", default="finetune_gridsearch_results.txt",
                        help="File ghi ket qua")
    args = parser.parse_args()

    time_limit = args.time
    n_workers = max(1, args.cores)

    if not os.path.isdir(TESTCASE_DIR):
        print(f"Khong tim thay thu muc testcase: {TESTCASE_DIR}")
        sys.exit(1)
    testcases = sorted(
        os.path.join(TESTCASE_DIR, f)
        for f in os.listdir(TESTCASE_DIR)
        if f.endswith(".txt")
    )
    if not testcases:
        print(f"Khong tim thay testcase trong {TESTCASE_DIR}")
        sys.exit(1)

    algos = ["aco", "ga"] if args.algo == "both" else [args.algo]

    out_lines = []
    out_lines.append("=" * 72)
    out_lines.append("FINETUNE GRID SEARCH - ACO & GA")
    out_lines.append(f"Time limit moi lan chay: {time_limit}s   "
                     f"So testcase: {len(testcases)}")
    out_lines.append("=" * 72)

    for algo in algos:
        finetune(algo, testcases, time_limit, n_workers, out_lines)

    out_path = os.path.join(ROOT, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nDa ghi ket qua: {out_path}")


if __name__ == "__main__":
    main()
