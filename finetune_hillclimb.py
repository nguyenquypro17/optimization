"""
Finetune move probabilities for solver_hillclimb.py.

Grid search over (p_relocate, p_swap) combinations.
Each config is scored by running on all 20 testcases.
Best config = most rank-1 finishes (fewest (F1,F2) lexicographically).

Usage:
    python3 finetune_hillclimb.py              # default TIME_LIMIT=5s
    python3 finetune_hillclimb.py --time 9     # 9s per run (same as solver)
    python3 finetune_hillclimb.py --time 3     # 3s for quick sweep
"""

import sys
import os
import time
import random
import multiprocessing
import argparse
from itertools import product


# ─────────────────────────── Hyperparameter grid ───────────────────────────

# RELOCATE probability
P_RELOCATE_VALS = [0.40, 0.50, 0.60, 0.70, 0.80]
# SWAP probability (marginal, not cumulative)
P_SWAP_VALS     = [0.10, 0.15, 0.20, 0.25, 0.30]

# Remaining probability goes to REORDER.


# ─────────────────────────────── Parser ────────────────────────────────────

def parse(path):
    with open(path) as f:
        tok = f.read().split()
    N = int(tok[1]); nd = int(tok[3]); idx = 4
    t = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(nd):
        a, b, d = int(tok[idx]), int(tok[idx+1]), int(tok[idx+2])
        t[a][b] = d; idx += 3
    trailer_loc = int(tok[idx+1]); trailer_time = int(tok[idx+2]); idx += 3
    m = int(tok[idx+1]); idx += 2
    truck_depot = {}
    for _ in range(m):
        truck_depot[int(tok[idx])] = int(tok[idx+1]); idx += 2
    reqs = {}
    while idx < len(tok) and tok[idx] == "REQ":
        rid  = int(tok[idx+1]); size = int(tok[idx+2])
        reqs[rid] = dict(
            a=int(tok[idx+3]), pa=tok[idx+4], pdur=int(tok[idx+5]),
            b=int(tok[idx+6]), da=tok[idx+7], ddur=int(tok[idx+8])
        )
        idx += 9
    return dict(N=N, t=t, trailer_loc=trailer_loc, trailer_time=trailer_time,
                m=m, truck_depot=truck_depot, reqs=reqs)


# ──────────────────────────── Hill climber ─────────────────────────────────

def solve(data, p_relocate, p_swap, time_limit, seed=12345):
    """Return best (F1, F2) found within time_limit seconds."""
    t    = data["t"]
    TL   = data["trailer_loc"]
    TT   = data["trailer_time"]
    reqs = data["reqs"]
    rids = list(reqs.keys())
    trucks  = list(data["truck_depot"].keys())
    depot   = data["truck_depot"]
    nT      = len(trucks)
    p_swap_cumul = p_relocate + p_swap   # cumulative threshold for SWAP branch

    rng = random.Random(seed)

    if not rids:
        return (0, 0)

    A   = {r: reqs[r]["a"]    for r in rids}
    B   = {r: reqs[r]["b"]    for r in rids}
    PD  = {r: reqs[r]["pdur"] for r in rids}
    DD  = {r: reqs[r]["ddur"] for r in rids}
    isPC = {r: reqs[r]["pa"] == "PICKUP_CONTAINER"  for r in rids}
    isDC = {r: reqs[r]["da"] == "DROP_CONTAINER"    for r in rids}

    def decode(order, dep):
        pos = dep; tm = tr = 0; hh = 0
        for r in order:
            a = A[r]; b = B[r]
            if isPC[r]:
                if hh == 0:
                    d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 1
                d = t[pos][a]; tr += d; tm += d + PD[r]; pos = a
            else:
                if hh == 1:
                    d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 0
                d = t[pos][a]; tr += d; tm += d + PD[r]; pos = a; hh = 1
            if isDC[r]:
                d = t[pos][b]; tr += d; tm += d + DD[r]; pos = b
            else:
                d = t[pos][b]; tr += d; tm += d + DD[r]; pos = b; hh = 0
        if hh == 1:
            d = t[pos][TL]; tr += d; tm += d + TT; pos = TL
        d = t[pos][dep]; tr += d; tm += d
        return tr, tm

    # Initialise: greedy assignment balanced by makespan
    assign    = {k: [] for k in trucks}
    truck_of  = {}
    travel_c  = {k: 0 for k in trucks}
    complete_c = {k: 0 for k in trucks}
    for r in sorted(rids, key=lambda r: -(PD[r] + DD[r])):
        k = min(trucks, key=lambda k: complete_c[k])
        assign[k].append(r); truck_of[r] = k
        travel_c[k], complete_c[k] = decode(assign[k], depot[k])

    F2 = sum(travel_c.values())
    def calcF1():
        return max(complete_c.values()) if complete_c else 0
    F1 = calcF1()
    best = (F1, F2)

    t_end = time.time() + time_limit
    no_improve = 0

    while time.time() < t_end:
        move = rng.random()

        # ── RELOCATE ──────────────────────────────────────────────────────
        if move < p_relocate:
            src = (max(trucks, key=lambda k: complete_c[k])
                   if no_improve % 3 == 0 else rng.choice(trucks))
            if not assign[src]:
                continue
            i = rng.randrange(len(assign[src]))
            r = assign[src][i]
            dst = rng.choice(trucks)
            old_src = assign[src]; old_dst = assign[dst]
            new_src = old_src[:i] + old_src[i+1:]
            base = new_src if dst == src else old_dst
            j = rng.randrange(len(base) + 1)
            new_dst = base[:j] + [r] + base[j:]

            if dst == src:
                ntr, ntm = decode(new_dst, depot[src])
                save_c = complete_c[src]; complete_c[src] = ntm
                nF1 = calcF1(); nF2 = F2 + ntr - travel_c[src]
                if (nF1, nF2) < (F1, F2):
                    assign[src] = new_dst; travel_c[src] = ntr
                    F1, F2 = nF1, nF2; no_improve = 0
                else:
                    complete_c[src] = save_c; no_improve += 1
            else:
                ntr_s, ntm_s = decode(new_src, depot[src])
                ntr_d, ntm_d = decode(new_dst, depot[dst])
                nF2 = F2 - travel_c[src] - travel_c[dst] + ntr_s + ntr_d
                sc_s, sc_d = complete_c[src], complete_c[dst]
                complete_c[src] = ntm_s; complete_c[dst] = ntm_d
                nF1 = calcF1()
                if (nF1, nF2) < (F1, F2):
                    assign[src] = new_src; assign[dst] = new_dst
                    travel_c[src] = ntr_s; travel_c[dst] = ntr_d
                    truck_of[r] = dst; F1, F2 = nF1, nF2; no_improve = 0
                else:
                    complete_c[src] = sc_s; complete_c[dst] = sc_d; no_improve += 1

        # ── SWAP ──────────────────────────────────────────────────────────
        elif move < p_swap_cumul and nT >= 2:
            k1, k2 = rng.sample(trucks, 2)
            if not assign[k1] or not assign[k2]:
                continue
            i1 = rng.randrange(len(assign[k1]))
            i2 = rng.randrange(len(assign[k2]))
            r1 = assign[k1][i1]; r2 = assign[k2][i2]
            new1 = list(assign[k1]); new1[i1] = r2
            new2 = list(assign[k2]); new2[i2] = r1
            ntr1, ntm1 = decode(new1, depot[k1])
            ntr2, ntm2 = decode(new2, depot[k2])
            nF2 = F2 - travel_c[k1] - travel_c[k2] + ntr1 + ntr2
            sc1, sc2 = complete_c[k1], complete_c[k2]
            complete_c[k1] = ntm1; complete_c[k2] = ntm2
            nF1 = calcF1()
            if (nF1, nF2) < (F1, F2):
                assign[k1] = new1; assign[k2] = new2
                travel_c[k1] = ntr1; travel_c[k2] = ntr2
                truck_of[r1] = k2; truck_of[r2] = k1
                F1, F2 = nF1, nF2; no_improve = 0
            else:
                complete_c[k1] = sc1; complete_c[k2] = sc2; no_improve += 1

        # ── REORDER (2-opt) ───────────────────────────────────────────────
        else:
            k = rng.choice(trucks)
            L = assign[k]
            if len(L) < 2:
                continue
            i = rng.randrange(len(L)); j = rng.randrange(len(L))
            if i == j:
                continue
            if i > j: i, j = j, i
            new = L[:i] + L[i:j+1][::-1] + L[j+1:]
            ntr, ntm = decode(new, depot[k])
            nF2 = F2 - travel_c[k] + ntr
            sc = complete_c[k]; complete_c[k] = ntm
            nF1 = calcF1()
            if (nF1, nF2) < (F1, F2):
                assign[k] = new; travel_c[k] = ntr
                F1, F2 = nF1, nF2; no_improve = 0
            else:
                complete_c[k] = sc; no_improve += 1

        if (F1, F2) < best:
            best = (F1, F2)

        # Perturbation khi bị ket local optimum
        if no_improve > 4000:
            for _ in range(max(1, len(rids) // 20)):
                r = rng.choice(rids)
                s = truck_of[r]
                assign[s].remove(r)
                d = rng.choice(trucks)
                assign[d].append(r); truck_of[r] = d
                travel_c[s], complete_c[s] = decode(assign[s], depot[s])
                travel_c[d], complete_c[d] = decode(assign[d], depot[d])
            F2 = sum(travel_c.values()); F1 = calcF1()
            no_improve = 0

    return best


# ─────────────────────── Worker: 1 task = 1 testcase ───────────────────────

def run_testcase(args):
    """Parse testcase once, run all configs sequentially, return results."""
    path, configs, time_limit = args
    t0 = time.time()
    data = parse(path)
    parse_time = time.time() - t0

    scores = []
    for pr, ps in configs:
        f1, f2 = solve(data, pr, ps, time_limit)
        scores.append((f1, f2))

    name = os.path.basename(path)
    elapsed = time.time() - t0
    best_score = min(scores)
    best_cfg_idx = scores.index(best_score)
    pr_best, ps_best = configs[best_cfg_idx]
    print(f"  [{name}] done in {elapsed:.1f}s (parse {parse_time:.1f}s) "
          f"| winner: p_rel={pr_best:.2f} p_swap={ps_best:.2f} "
          f"-> F1={best_score[0]} F2={best_score[1]}",
          flush=True)
    return scores


# ───────────────────────────────── Main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=5.0,
                        help="Time limit per run in seconds (default 5)")
    parser.add_argument("--cores", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    TIME_LIMIT = args.time
    N_WORKERS  = max(1, args.cores)

    testcase_dir = "testcase"
    testcases = sorted([
        os.path.join(testcase_dir, f)
        for f in os.listdir(testcase_dir)
        if f.endswith(".txt")
    ])
    n_tc = len(testcases)

    # Build config list: (p_relocate, p_swap_marginal)
    configs = []
    for pr in P_RELOCATE_VALS:
        for ps in P_SWAP_VALS:
            if pr + ps < 1.0:
                configs.append((pr, ps))
    n_cfg = len(configs)

    total_runs  = n_tc * n_cfg
    est_serial  = total_runs * TIME_LIMIT
    est_wall    = est_serial / N_WORKERS

    print("=" * 60)
    print(f"Testcases    : {n_tc}")
    print(f"Configs      : {n_cfg}  (p_relocate x p_swap grid)")
    print(f"Time/run     : {TIME_LIMIT}s")
    print(f"Workers      : {N_WORKERS}")
    print(f"Est. wall    : ~{est_wall/60:.1f} min")
    print("=" * 60)
    print("Configs being tested:")
    for pr, ps in configs:
        print(f"  p_rel={pr:.2f}  p_swap={ps:.2f}  p_reorder={1-pr-ps:.2f}")
    print("=" * 60)
    print("Running ...\n")

    # Each task = one testcase (parsed once, all configs run inside)
    tasks = [(tc, configs, TIME_LIMIT) for tc in testcases]

    wall_start = time.time()
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        all_results = pool.map(run_testcase, tasks)
    wall_time = time.time() - wall_start

    # all_results[i][j] = (F1, F2) for testcase i, config j
    # Count rank-1 wins
    wins       = [0] * n_cfg
    top2_wins  = [0] * n_cfg   # ties included

    for i in range(n_tc):
        scores = all_results[i]
        best_score = min(scores)
        for j, s in enumerate(scores):
            if s == best_score:
                wins[j] += 1
        # rank within testcase
        ranked_idx = sorted(range(n_cfg), key=lambda j: scores[j])
        top2_wins[ranked_idx[0]] += 2
        if n_cfg > 1:
            top2_wins[ranked_idx[1]] += 1

    # ─── Results table ───
    print("\n" + "=" * 68)
    print(f"{'Rank':>4}  {'p_rel':>6}  {'p_swap':>6}  {'p_reord':>7}  "
          f"{'Wins':>5}  {'Points':>7}  {'Avg F1':>10}  {'Avg F2':>12}")
    print("-" * 68)

    ranked_cfgs = sorted(range(n_cfg),
                         key=lambda j: (-wins[j], -top2_wins[j]))

    avg_F1 = [sum(all_results[i][j][0] for i in range(n_tc)) / n_tc
              for j in range(n_cfg)]
    avg_F2 = [sum(all_results[i][j][1] for i in range(n_tc)) / n_tc
              for j in range(n_cfg)]

    for rank, j in enumerate(ranked_cfgs, 1):
        pr, ps = configs[j]
        pr_reorder = round(1.0 - pr - ps, 4)
        print(f"{rank:>4}  {pr:>6.2f}  {ps:>6.2f}  {pr_reorder:>7.2f}  "
              f"{wins[j]:>5}  {top2_wins[j]:>7}  "
              f"{avg_F1[j]:>10.0f}  {avg_F2[j]:>12.0f}")

    best_j = ranked_cfgs[0]
    pr_b, ps_b = configs[best_j]

    print("=" * 68)
    print(f"\nTotal wall time: {wall_time:.1f}s")
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"  p_relocate = {pr_b:.2f}   -> threshold1 = {pr_b:.2f}")
    print(f"  p_swap     = {ps_b:.2f}   -> threshold2 = {pr_b + ps_b:.2f}")
    print(f"  p_reorder  = {1-pr_b-ps_b:.2f}")
    print(f"  Wins       = {wins[best_j]} / {n_tc} testcases")
    print(f"\nApply to solver_hillclimb.py:")
    print(f"  if move < {pr_b:.2f}:                     # RELOCATE")
    print(f"  elif move < {pr_b + ps_b:.2f} and nT >= 2: # SWAP")
    print(f"  else:                              # REORDER")


if __name__ == "__main__":
    main()
