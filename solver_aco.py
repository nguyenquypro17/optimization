import sys, time, random, math

# ============================================================
# Truck Container Scheduling - Ant Colony Optimization (ACO)
# Khong dung OR-Tools. Doc stdin, ghi stdout (dung cho server cham).
#
# Chay:  python3 solver_aco.py            (mac dinh 9s)
#        python3 solver_aco.py 18          (gioi han 18 giay)
#
# Y tuong:
#  - Moi con kien xay dung giai phap: lan luot chon (truck, req)
#    dua tren pheromone + heuristic cho den khi phan cong het req.
#  - 2 ma tran pheromone:
#      tau_assign[k][r]: pheromone gan req r vao truck k
#      tau_seq[r1][r2]:  pheromone cho r2 dung sau r1 cung truck
#  - Local search sau moi con kien (relocate/swap/2-opt).
#  - Cap nhat pheromone: boc hoi + nap tu global best.
#  - Inline tuning: 25% time dau thu 8 config.
# ============================================================

TIME_LIMIT = 9.0
random.seed(77)

# ----------------------------- Doc input -----------------------------
def parse():
    tok = sys.stdin.read().split()
    if not tok:
        return None
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

# ----------------------------- Bo giai ma (decoder) -----------------------------
def make_decoder(data):
    t    = data["t"]; TL = data["trailer_loc"]; TT = data["trailer_time"]
    reqs = data["reqs"]
    A    = {r: reqs[r]["a"]    for r in reqs}
    B    = {r: reqs[r]["b"]    for r in reqs}
    PD   = {r: reqs[r]["pdur"] for r in reqs}
    DD   = {r: reqs[r]["ddur"] for r in reqs}
    isPC = {r: reqs[r]["pa"] == "PICKUP_CONTAINER" for r in reqs}
    isDC = {r: reqs[r]["da"] == "DROP_CONTAINER"   for r in reqs}

    def decode(order, dep, build=False):
        pos = dep; tm = 0; tr = 0; hh = 0
        ops = [] if build else None
        for r in order:
            a = A[r]; b = B[r]
            if isPC[r]:
                if hh == 0:
                    d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 1
                    if build: ops.append((TL, "PICKUP_TRAILER", None))
                d = t[pos][a]; tr += d; tm += d + PD[r]; pos = a
                if build: ops.append((a, "PICKUP_CONTAINER", r))
            else:
                if hh == 1:
                    d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 0
                    if build: ops.append((TL, "DROP_TRAILER", None))
                d = t[pos][a]; tr += d; tm += d + PD[r]; pos = a; hh = 1
                if build: ops.append((a, "PICKUP_CONTAINER_TRAILER", r))
            if isDC[r]:
                d = t[pos][b]; tr += d; tm += d + DD[r]; pos = b
                if build: ops.append((b, "DROP_CONTAINER", r))
            else:
                d = t[pos][b]; tr += d; tm += d + DD[r]; pos = b; hh = 0
                if build: ops.append((b, "DROP_CONTAINER_TRAILER", r))
        if hh == 1:
            d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 0
            if build: ops.append((TL, "DROP_TRAILER", None))
        d = t[pos][dep]; tr += d; tm += d
        if build:
            return tr, tm, ops
        return tr, tm

    return decode

# ----------------------------- Danh gia -----------------------------
def evaluate(assign, decode, depot):
    complete_c = {}; travel_c = {}
    for k, order in assign.items():
        tr, tm = decode(order, depot[k])
        travel_c[k] = tr; complete_c[k] = tm
    F1 = max(complete_c.values()) if complete_c else 0
    F2 = sum(travel_c.values())
    return F1, F2

# ----------------------------- Heuristic marginal cost -----------------------------
def marginal_cost(assign_k, r, depot_k, decode):
    """Chi phi tang them khi gan r vao cuoi danh sach cua truck k."""
    _, tm_before = decode(assign_k, depot_k)
    _, tm_after  = decode(assign_k + [r], depot_k)
    return max(1, tm_after - tm_before)

# ----------------------------- Xay dung giai phap 1 con kien -----------------------------
def build_ant(data, decode, tau_assign, tau_seq, alpha, beta, trucks):
    depot = data["truck_depot"]
    rids  = list(data["reqs"].keys())
    assign      = {k: [] for k in trucks}
    complete_c  = {k: 0  for k in trucks}
    last_req    = {k: None for k in trucks}  # req cuoi cung tren truck k

    unassigned = list(rids)
    random.shuffle(unassigned)

    for r in unassigned:
        scores = []
        for k in trucks:
            # pheromone: tau_assign[k][r] va tau_seq[last][r]
            tau_a = tau_assign[k][r]
            last  = last_req[k]
            tau_s = tau_seq[last][r] if last is not None else 1.0
            tau   = tau_a * tau_s

            # heuristic: nghich dao cua chi phi tang them
            mc = marginal_cost(assign[k], r, depot[k], decode)
            eta = 1.0 / mc

            score = (tau ** alpha) * (eta ** beta)
            scores.append((k, score))

        # Chon truck theo xac suat
        total = sum(s for _, s in scores)
        if total <= 0:
            chosen_k = random.choice(trucks)
        else:
            rnd = random.random() * total
            cum = 0.0
            chosen_k = scores[-1][0]
            for k, s in scores:
                cum += s
                if cum >= rnd:
                    chosen_k = k
                    break

        assign[chosen_k].append(r)
        last_req[chosen_k] = r
        _, complete_c[chosen_k] = decode(assign[chosen_k], depot[chosen_k])

    return assign

# ----------------------------- Local search (cai thien cuc bo) -----------------------------
def local_search(assign, trucks, depot, decode, n_steps=300):
    travel_c   = {k: decode(assign[k], depot[k])[0] for k in trucks}
    complete_c = {k: decode(assign[k], depot[k])[1] for k in trucks}
    F2 = sum(travel_c.values())

    def calcF1():
        return max(complete_c.values()) if complete_c else 0

    F1 = calcF1()

    for _ in range(n_steps):
        move = random.random()

        if move < 0.6:
            # RELOCATE
            src = max(trucks, key=lambda k: complete_c[k]) if random.random() < 0.4 else random.choice(trucks)
            if not assign[src]:
                continue
            i = random.randrange(len(assign[src]))
            r = assign[src][i]
            dst = random.choice(trucks)
            new_src = assign[src][:i] + assign[src][i + 1:]
            base = new_src if dst == src else assign[dst]
            j = random.randrange(len(base) + 1)
            new_dst = base[:j] + [r] + base[j:]

            if dst == src:
                ntr, ntm = decode(new_dst, depot[src])
                nF2 = F2 - travel_c[src] + ntr
                sc = complete_c[src]; complete_c[src] = ntm
                nF1 = calcF1()
                if (nF1, nF2) < (F1, F2):
                    assign[src] = new_dst; travel_c[src] = ntr
                    F1, F2 = nF1, nF2
                else:
                    complete_c[src] = sc
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
                    F1, F2 = nF1, nF2
                else:
                    complete_c[src] = sc_s; complete_c[dst] = sc_d

        elif move < 0.8 and len(trucks) >= 2:
            # SWAP
            k1, k2 = random.sample(trucks, 2)
            if not assign[k1] or not assign[k2]:
                continue
            i1 = random.randrange(len(assign[k1]))
            i2 = random.randrange(len(assign[k2]))
            new1 = list(assign[k1]); new1[i1] = assign[k2][i2]
            new2 = list(assign[k2]); new2[i2] = assign[k1][i1]
            ntr1, ntm1 = decode(new1, depot[k1])
            ntr2, ntm2 = decode(new2, depot[k2])
            nF2 = F2 - travel_c[k1] - travel_c[k2] + ntr1 + ntr2
            sc1, sc2 = complete_c[k1], complete_c[k2]
            complete_c[k1] = ntm1; complete_c[k2] = ntm2
            nF1 = calcF1()
            if (nF1, nF2) < (F1, F2):
                assign[k1] = new1; assign[k2] = new2
                travel_c[k1] = ntr1; travel_c[k2] = ntr2
                F1, F2 = nF1, nF2
            else:
                complete_c[k1] = sc1; complete_c[k2] = sc2

        else:
            # 2-OPT
            k = random.choice(trucks)
            L = assign[k]
            if len(L) < 2:
                continue
            i, j = sorted(random.sample(range(len(L)), 2))
            new = L[:i] + L[i:j + 1][::-1] + L[j + 1:]
            ntr, ntm = decode(new, depot[k])
            nF2 = F2 - travel_c[k] + ntr
            sc = complete_c[k]; complete_c[k] = ntm
            nF1 = calcF1()
            if (nF1, nF2) < (F1, F2):
                assign[k] = new; travel_c[k] = ntr
                F1, F2 = nF1, nF2
            else:
                complete_c[k] = sc

    return assign, F1, F2

# ----------------------------- Cap nhat pheromone -----------------------------
def update_pheromone(tau_assign, tau_seq, best_assign, best_score, rho, Q):
    trucks = list(tau_assign.keys())
    rids   = list(tau_assign[trucks[0]].keys())

    # Boc hoi
    for k in trucks:
        for r in rids:
            tau_assign[k][r] *= (1.0 - rho)
    for r1 in rids:
        for r2 in rids:
            if r1 != r2:
                tau_seq[r1][r2] *= (1.0 - rho)

    # Nap pheromone tu global best
    F1, F2 = best_score
    deposit = Q / max(1, F1 + F2)

    for k, order in best_assign.items():
        for r in order:
            tau_assign[k][r] += deposit
        for idx in range(len(order) - 1):
            tau_seq[order[idx]][order[idx + 1]] += deposit

# ----------------------------- ACO chinh -----------------------------
def run_aco(data, params, time_budget, t_start):
    trucks  = list(data["truck_depot"].keys())
    depot   = data["truck_depot"]
    rids    = list(data["reqs"].keys())
    decode  = make_decoder(data)

    n_ants  = params["n_ants"]
    alpha   = params["alpha"]
    beta    = params["beta"]
    rho     = params["rho"]
    Q       = params.get("Q", 1000.0)
    ls_steps = params.get("ls_steps", 300)

    # Khoi tao pheromone
    tau0 = 1.0
    tau_assign = {k: {r: tau0 for r in rids} for k in trucks}
    tau_seq    = {r1: {r2: tau0 for r2 in rids} for r1 in rids}

    best_assign = None
    best_score  = None

    t_end = t_start + time_budget
    while time.time() < t_end:
        iter_best_assign = None
        iter_best_score  = None

        # Moi vong: n_ants con kien xay dung giai phap
        for _ in range(n_ants):
            if time.time() >= t_end:
                break
            ant_assign = build_ant(data, decode, tau_assign, tau_seq,
                                   alpha, beta, trucks)
            ant_assign, F1, F2 = local_search(ant_assign, trucks, depot,
                                               decode, n_steps=ls_steps)
            score = (F1, F2)

            if iter_best_score is None or score < iter_best_score:
                iter_best_score  = score
                iter_best_assign = {k: list(v) for k, v in ant_assign.items()}

        if iter_best_score is not None:
            if best_score is None or iter_best_score < best_score:
                best_score  = iter_best_score
                best_assign = {k: list(v) for k, v in iter_best_assign.items()}

            update_pheromone(tau_assign, tau_seq,
                             iter_best_assign, iter_best_score, rho, Q)

    return best_assign, best_score

# ----------------------------- Inline tuning -----------------------------
TUNE_CONFIGS = [
    dict(n_ants=5,  alpha=1.0, beta=2.0, rho=0.3, Q=1000.0, ls_steps=200),
    dict(n_ants=10, alpha=1.0, beta=2.0, rho=0.3, Q=1000.0, ls_steps=200),
    dict(n_ants=5,  alpha=2.0, beta=2.0, rho=0.1, Q=1000.0, ls_steps=300),
    dict(n_ants=10, alpha=1.0, beta=3.0, rho=0.3, Q=2000.0, ls_steps=200),
    dict(n_ants=5,  alpha=1.0, beta=1.0, rho=0.5, Q=1000.0, ls_steps=400),
    dict(n_ants=20, alpha=1.0, beta=2.0, rho=0.3, Q=1000.0, ls_steps=150),
    dict(n_ants=10, alpha=2.0, beta=3.0, rho=0.1, Q=2000.0, ls_steps=300),
    dict(n_ants=5,  alpha=1.0, beta=2.0, rho=0.3, Q=500.0,  ls_steps=500),
]

def tune(data, total_time, t_start):
    """Chay nhanh cac config, chon config tot nhat."""
    rids = list(data["reqs"].keys())
    if not rids:
        return TUNE_CONFIGS[0], {k: [] for k in data["truck_depot"]}, (0, 0)

    n_configs  = len(TUNE_CONFIGS)
    tune_budget = total_time * 0.25
    time_each   = tune_budget / n_configs

    best_params = TUNE_CONFIGS[0]
    best_score  = None
    best_assign = None

    for cfg in TUNE_CONFIGS:
        if time.time() - t_start >= tune_budget:
            break
        t0 = time.time()
        assign, score = run_aco(data, cfg, time_each, t0)
        if assign is None:
            continue
        if best_score is None or score < best_score:
            best_score  = score
            best_params = cfg
            best_assign = assign

    if best_assign is None:
        best_params = TUNE_CONFIGS[0]
        best_assign = {k: [] for k in data["truck_depot"]}
        best_score  = (0, 0)

    return best_params, best_assign, best_score

# ----------------------------- Xuat output -----------------------------
def format_output(data, assign, decode):
    depot  = data["truck_depot"]
    trucks = list(depot.keys())
    used   = [k for k in trucks if assign.get(k)]
    out    = [f"ROUTES {len(used)}"]
    for k in used:
        _, _, ops = decode(assign[k], depot[k], build=True)
        out.append(f"TRUCK {k}")
        for (loc, act, ref) in ops:
            if ref is None:
                out.append(f"{loc} {act}")
            else:
                out.append(f"{loc} {act} {ref}")
        out.append(f"{depot[k]} STOP")
        out.append("#")
    return "\n".join(out)

# ----------------------------- main -----------------------------
def main():
    global TIME_LIMIT
    if len(sys.argv) >= 2:
        try:
            TIME_LIMIT = float(sys.argv[1])
        except ValueError:
            pass

    data = parse()
    if not data:
        return

    t_start = time.time()
    rids    = list(data["reqs"].keys())
    trucks  = list(data["truck_depot"].keys())

    if not rids:
        sys.stdout.write(f"ROUTES {len(trucks)}\n")
        for k in trucks:
            dep = data["truck_depot"][k]
            sys.stdout.write(f"TRUCK {k}\n{dep} STOP\n#\n")
        return

    # Inline tuning
    best_params, warm_assign, warm_score = tune(data, TIME_LIMIT, t_start)

    # Chay ACO voi best params trong thoi gian con lai
    elapsed = time.time() - t_start
    remain  = TIME_LIMIT - elapsed

    decode = make_decoder(data)
    if remain > 0.5:
        best_assign, best_score = run_aco(data, best_params, remain, time.time())
        # Neu ACO tra ve None (het gio), dung warm
        if best_assign is None:
            best_assign = warm_assign
        elif warm_score < best_score:
            best_assign = warm_assign
    else:
        best_assign = warm_assign

    sys.stdout.write(format_output(data, best_assign, decode) + "\n")

if __name__ == "__main__":
    main()
