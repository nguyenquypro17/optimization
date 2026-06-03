import sys, time, random

# ============================================================
# Truck Container Scheduling - Genetic Algorithm (GA)
# Khong dung OR-Tools. Doc stdin, ghi stdout (dung cho server cham).
#
# Chay:  python3 solver_ga.py            (mac dinh 9s)
#        python3 solver_ga.py 18          (gioi han 18 giay)
#
# Y tuong:
#  - Chromosome = phan cong + thu tu yeu cau cho moi dau keo.
#  - Bo giai ma (decoder) tu dong chen thao tac lay/tra ro-mooc
#    => moi ca the luon sinh lo trinh HOP LE.
#  - Crossover: OX (thu tu noi bo) + hoang doi khoi yeu cau.
#  - Mutation: relocate, swap, 2-opt.
#  - Selection: tournament.
#  - Inline tuning: 25% time dau thu nhanh 6 config, chon best.
# ============================================================

TIME_LIMIT = 9.0
random.seed(42)

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
# Nhan order (danh sach req id), tra ve (travel, complete) va ops neu build=True
def make_decoder(data):
    t = data["t"]; TL = data["trailer_loc"]; TT = data["trailer_time"]
    reqs = data["reqs"]
    A   = {r: reqs[r]["a"]   for r in reqs}
    B   = {r: reqs[r]["b"]   for r in reqs}
    PD  = {r: reqs[r]["pdur"] for r in reqs}
    DD  = {r: reqs[r]["ddur"] for r in reqs}
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

# ----------------------------- Danh gia ca the -----------------------------
def evaluate(assign, decode, depot):
    travel_c = {}; complete_c = {}
    for k, order in assign.items():
        tr, tm = decode(order, depot[k])
        travel_c[k] = tr; complete_c[k] = tm
    F1 = max(complete_c.values()) if complete_c else 0
    F2 = sum(travel_c.values())
    return F1, F2

# ----------------------------- Khoi tao greedy -----------------------------
def init_greedy(trucks, depot, rids, decode):
    assign = {k: [] for k in trucks}
    complete_c = {k: 0 for k in trucks}
    travel_c   = {k: 0 for k in trucks}
    order_init = sorted(rids, key=lambda r: -(decode([r], depot[trucks[0]])[1]))
    for r in order_init:
        k = min(trucks, key=lambda k: complete_c[k])
        assign[k].append(r)
        travel_c[k], complete_c[k] = decode(assign[k], depot[k])
    return assign

def init_random(trucks, rids):
    assign = {k: [] for k in trucks}
    shuffled = list(rids); random.shuffle(shuffled)
    for i, r in enumerate(shuffled):
        assign[trucks[i % len(trucks)]].append(r)
    for k in trucks:
        random.shuffle(assign[k])
    return assign

# ----------------------------- Crossover -----------------------------
def crossover(p1, p2, trucks, cx_rate):
    """
    Crossover chon phan cong (assignment crossover):
    - Voi moi req, ngau nhien chon truck tu p1 hoac p2.
    - Giu thu tu tuong doi theo vi tri trong cha me tuong ung.
    - Luon sinh child hop le (moi req xuat hien dung 1 lan).
    """
    if random.random() > cx_rate:
        return {k: list(v) for k, v in p1.items()}

    # Vi tri cua tung req trong moi cha me (de sap xep thu tu)
    pos_p1 = {r: (k_idx, pos)
              for k_idx, k in enumerate(trucks)
              for pos, r in enumerate(p1[k])}
    pos_p2 = {r: (k_idx, pos)
              for k_idx, k in enumerate(trucks)
              for pos, r in enumerate(p2[k])}

    # Truck tuong ung trong moi cha me
    truck_p1 = {r: k for k in trucks for r in p1[k]}
    truck_p2 = {r: k for k in trucks for r in p2[k]}

    child = {k: [] for k in trucks}
    chosen_parent = {}  # r -> 1 hoac 2

    all_reqs = list(truck_p1.keys())
    for r in all_reqs:
        if random.random() < 0.5:
            child[truck_p1[r]].append(r)
            chosen_parent[r] = 1
        else:
            child[truck_p2[r]].append(r)
            chosen_parent[r] = 2

    # Sap xep thu tu trong tung truck theo vi tri goc trong cha me
    for k in trucks:
        child[k].sort(key=lambda r: pos_p1[r] if chosen_parent[r] == 1 else pos_p2[r])

    return child

# ----------------------------- Mutation -----------------------------
def mutate(assign, trucks, depot, decode, mut_rate):
    """Ap dung 1 trong 3 phep bien doi neu xac suat thoa man."""
    if random.random() > mut_rate:
        return
    move = random.random()
    if move < 0.5:
        # RELOCATE
        src = random.choice(trucks)
        if not assign[src]:
            return
        i = random.randrange(len(assign[src]))
        r = assign[src][i]
        dst = random.choice(trucks)
        assign[src] = assign[src][:i] + assign[src][i + 1:]
        base = assign[dst]
        j = random.randrange(len(base) + 1)
        assign[dst] = base[:j] + [r] + base[j:]
    elif move < 0.75 and len(trucks) >= 2:
        # SWAP giua 2 truck
        k1, k2 = random.sample(trucks, 2)
        if not assign[k1] or not assign[k2]:
            return
        i1 = random.randrange(len(assign[k1]))
        i2 = random.randrange(len(assign[k2]))
        assign[k1][i1], assign[k2][i2] = assign[k2][i2], assign[k1][i1]
    else:
        # 2-OPT noi bo
        k = random.choice(trucks)
        L = assign[k]
        if len(L) < 2:
            return
        i, j = sorted(random.sample(range(len(L)), 2))
        assign[k] = L[:i] + L[i:j + 1][::-1] + L[j + 1:]

# ----------------------------- Tournament selection -----------------------------
def tournament_select(pop, scores, tourn_size):
    candidates = random.sample(range(len(pop)), min(tourn_size, len(pop)))
    return min(candidates, key=lambda i: scores[i])

# ----------------------------- GA chinh -----------------------------
def run_ga(data, params, time_budget, t_start):
    trucks  = list(data["truck_depot"].keys())
    depot   = data["truck_depot"]
    rids    = list(data["reqs"].keys())
    decode  = make_decoder(data)

    pop_size   = params["pop_size"]
    cx_rate    = params["cx_rate"]
    mut_rate   = params["mut_rate"]
    tourn_size = params["tourn_size"]

    if not rids:
        return {k: [] for k in trucks}

    # Khoi tao
    pop = [init_greedy(trucks, depot, rids, decode)]
    while len(pop) < pop_size:
        pop.append(init_random(trucks, rids))

    scores = [evaluate(ind, decode, depot) for ind in pop]
    best_score = min(scores)
    best_idx   = scores.index(best_score)
    best_assign = {k: list(v) for k, v in pop[best_idx].items()}

    t_end = t_start + time_budget
    while time.time() < t_end:
        # Chon 2 cha me
        p1_idx = tournament_select(pop, scores, tourn_size)
        p2_idx = tournament_select(pop, scores, tourn_size)
        p1 = pop[p1_idx]; p2 = pop[p2_idx]

        child = crossover(p1, p2, trucks, cx_rate)
        mutate(child, trucks, depot, decode, mut_rate)

        child_score = evaluate(child, decode, depot)

        # Thay the ca the kem nhat trong pop (bao ve elite)
        worst_idx = max(range(len(pop)), key=lambda i: scores[i])
        elite_idx = scores.index(best_score)
        if worst_idx == elite_idx:
            sorted_idx = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
            worst_idx = next(idx for idx in sorted_idx if idx != elite_idx)
        pop[worst_idx] = child
        scores[worst_idx] = child_score

        if child_score < best_score:
            best_score = child_score
            best_assign = {k: list(v) for k, v in child.items()}

    return best_assign

# ----------------------------- Inline tuning -----------------------------
TUNE_CONFIGS = [
    dict(pop_size=20, cx_rate=0.8, mut_rate=0.3, tourn_size=3),
    dict(pop_size=40, cx_rate=0.8, mut_rate=0.2, tourn_size=3),
    dict(pop_size=40, cx_rate=0.6, mut_rate=0.4, tourn_size=5),
    dict(pop_size=60, cx_rate=0.8, mut_rate=0.2, tourn_size=2),
    dict(pop_size=20, cx_rate=0.6, mut_rate=0.4, tourn_size=2),
    dict(pop_size=60, cx_rate=0.6, mut_rate=0.3, tourn_size=5),
]

def tune(data, total_time, t_start):
    """Chay nhanh cac config, chon config tot nhat."""
    n_configs = len(TUNE_CONFIGS)
    tune_budget = total_time * 0.25
    time_each = tune_budget / n_configs

    best_params = TUNE_CONFIGS[0]
    best_score = None
    best_assign = None

    for cfg in TUNE_CONFIGS:
        if time.time() - t_start >= tune_budget:
            break
        t0 = time.time()
        assign = run_ga(data, cfg, time_each, t0)
        decode = make_decoder(data)
        score = evaluate(assign, decode, data["truck_depot"])
        if best_score is None or score < best_score:
            best_score = score
            best_params = cfg
            best_assign = assign

    return best_params, best_assign, best_score

# ----------------------------- Xuat output -----------------------------
def format_output(data, assign, decode):
    depot   = data["truck_depot"]
    trucks  = list(depot.keys())
    used    = [k for k in trucks if assign.get(k)]
    out     = [f"ROUTES {len(used)}"]
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
    rids = list(data["reqs"].keys())

    if not rids:
        trucks = list(data["truck_depot"].keys())
        sys.stdout.write(f"ROUTES {len(trucks)}\n")
        for k in trucks:
            dep = data["truck_depot"][k]
            sys.stdout.write(f"TRUCK {k}\n{dep} STOP\n#\n")
        return

    # Inline tuning
    best_params, warm_assign, _ = tune(data, TIME_LIMIT, t_start)

    # Chay GA voi best params trong thoi gian con lai
    elapsed = time.time() - t_start
    remain  = TIME_LIMIT - elapsed

    decode = make_decoder(data)
    if remain > 0.5:
        # Khoi tao pop voi warm_assign tu buoc tune
        trucks  = list(data["truck_depot"].keys())
        depot   = data["truck_depot"]
        pop_size = best_params["pop_size"]
        pop = [warm_assign]
        while len(pop) < pop_size:
            pop.append(init_random(trucks, rids))
        scores = [evaluate(ind, decode, depot) for ind in pop]
        best_score = min(scores)
        elite_idx  = scores.index(best_score)
        best_assign = {k: list(v) for k, v in pop[elite_idx].items()}

        t_end = t_start + TIME_LIMIT
        while time.time() < t_end:
            p1_idx = tournament_select(pop, scores, best_params["tourn_size"])
            p2_idx = tournament_select(pop, scores, best_params["tourn_size"])
            child  = crossover(pop[p1_idx], pop[p2_idx], trucks, best_params["cx_rate"])
            mutate(child, trucks, depot, decode, best_params["mut_rate"])
            child_score = evaluate(child, decode, depot)

            worst_idx  = max(range(len(pop)), key=lambda i: scores[i])
            cur_elite  = scores.index(best_score)
            if worst_idx == cur_elite:
                sorted_i  = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
                worst_idx = next(idx for idx in sorted_i if idx != cur_elite)
            pop[worst_idx]    = child
            scores[worst_idx] = child_score

            if child_score < best_score:
                best_score  = child_score
                best_assign = {k: list(v) for k, v in child.items()}
    else:
        best_assign = warm_assign

    sys.stdout.write(format_output(data, best_assign, decode) + "\n")

if __name__ == "__main__":
    main()
