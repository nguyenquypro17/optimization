import sys, time, random

# ============================================================
# Truck Container Scheduling - LEO DOI (Hill Climbing) thuan Python
# Khong dung OR-Tools. Doc stdin, ghi stdout (dung cho server cham).
#
# Chay:  python3 solver_hillclimb.py            (mac dinh 9s)
#        python3 solver_hillclimb.py 18          (gioi han 18 giay)
#
# Y tuong:
#  - Loi giai = phan cong + thu tu cac yeu cau cho moi dau keo.
#  - Bo giai ma (decoder) tu dong chen thao tac lay/tra ro-mooc o bai
#    => luon sinh lo trinh HOP LE, thoa rang buoc ro-mooc/suc chua.
#  - Leo doi voi cac buoc lang gieng: relocate (chuyen 1 yeu cau),
#    swap (doi cho 2 yeu cau), reorder noi bo. So sanh theo (F1, F2)
#    tu dien (F1 uu tien hon F2).
# ============================================================

TIME_LIMIT = 9.0
random.seed(12345)

# ----------------------------- Doc input -----------------------------
def parse():
    tok = sys.stdin.read().split()
    if not tok:
        return None
    N = int(tok[1]); nd = int(tok[3]); idx = 4
    t = [[0]*(N+1) for _ in range(N+1)]
    for _ in range(nd):
        a, b, d = int(tok[idx]), int(tok[idx+1]), int(tok[idx+2])
        t[a][b] = d; idx += 3
    trailer_loc, trailer_time = int(tok[idx+1]), int(tok[idx+2]); idx += 3
    m = int(tok[idx+1]); idx += 2
    truck_depot = {}
    for _ in range(m):
        truck_depot[int(tok[idx])] = int(tok[idx+1]); idx += 2
    reqs = {}
    while idx < len(tok) and tok[idx] == "REQ":
        rid = int(tok[idx+1]); size = int(tok[idx+2])
        reqs[rid] = dict(q=(2 if size == 40 else 1),
                         a=int(tok[idx+3]), pa=tok[idx+4], pdur=int(tok[idx+5]),
                         b=int(tok[idx+6]), da=tok[idx+7], ddur=int(tok[idx+8]))
        idx += 9
    return dict(N=N, t=t, trailer_loc=trailer_loc, trailer_time=trailer_time,
                m=m, truck_depot=truck_depot, reqs=reqs)

# ============================================================
def main():
    data = parse()
    t = data["t"]; TL = data["trailer_loc"]; TT = data["trailer_time"]
    trucks = list(data["truck_depot"].keys())
    depot = data["truck_depot"]
    reqs = data["reqs"]
    rids = list(reqs.keys())

    # mang thuoc tinh yeu cau (truy cap nhanh)
    A  = {r: reqs[r]["a"] for r in rids}
    B  = {r: reqs[r]["b"] for r in rids}
    PD = {r: reqs[r]["pdur"] for r in rids}
    DD = {r: reqs[r]["ddur"] for r in rids}
    isPC = {r: reqs[r]["pa"] == "PICKUP_CONTAINER" for r in rids}
    isDC = {r: reqs[r]["da"] == "DROP_CONTAINER" for r in rids}

    if not rids:
        sys.stdout.write("ROUTES 0\n")
        return

    # ---------------- Bo giai ma: order -> (travel, complete[, ops]) -------------
    def decode(order, dep, build=False):
        # tra ve (travel, complete) va (neu build) danh sach thao tac
        pos = dep; tm = 0; tr = 0; hh = 0
        ops = [] if build else None
        for r in order:
            a = A[r]; b = B[r]
            # --- lay container ---
            if isPC[r]:
                if hh == 0:                      # can ro-mooc rong -> ra bai gan
                    d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 1
                    if build: ops.append((TL, "PICKUP_TRAILER", None))
                d = t[pos][a]; tr += d; tm += d + PD[r]; pos = a
                if build: ops.append((a, "PICKUP_CONTAINER", r))
            else:                                # PCT: dau keo phai tran
                if hh == 1:                      # dang co ro-mooc -> tra ve bai truoc
                    d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 0
                    if build: ops.append((TL, "DROP_TRAILER", None))
                d = t[pos][a]; tr += d; tm += d + PD[r]; pos = a; hh = 1
                if build: ops.append((a, "PICKUP_CONTAINER_TRAILER", r))
            # --- tra container ---
            if isDC[r]:
                d = t[pos][b]; tr += d; tm += d + DD[r]; pos = b   # van giu ro-mooc
                if build: ops.append((b, "DROP_CONTAINER", r))
            else:
                d = t[pos][b]; tr += d; tm += d + DD[r]; pos = b; hh = 0
                if build: ops.append((b, "DROP_CONTAINER_TRAILER", r))
        if hh == 1:                              # con giu ro-mooc rong -> tra ve bai
            d = t[pos][TL]; tr += d; tm += d + TT; pos = TL; hh = 0
            if build: ops.append((TL, "DROP_TRAILER", None))
        d = t[pos][dep]; tr += d; tm += d; pos = dep             # ve bai dau keo
        if build:
            return tr, tm, ops
        return tr, tm

    # ---------------- Khoi tao: can bang makespan ----------------
    assign = {k: [] for k in trucks}
    truck_of = {}
    travel_c = {k: 0 for k in trucks}
    complete_c = {k: 0 for k in trucks}
    # gan moi yeu cau cho dau keo co thoi diem hoan thanh nho nhat hien tai
    order_init = sorted(rids, key=lambda r: -(PD[r] + DD[r]))
    for r in order_init:
        k = min(trucks, key=lambda k: complete_c[k])
        assign[k].append(r); truck_of[r] = k
        travel_c[k], complete_c[k] = decode(assign[k], depot[k])

    F2 = sum(travel_c.values())
    def calcF1():
        return max(complete_c.values()) if complete_c else 0
    F1 = calcF1()

    best = (F1, F2)
    best_assign = {k: list(assign[k]) for k in trucks}

    # ---------------- Leo doi ----------------
    t_end = time.time() + TIME_LIMIT
    it = 0
    no_improve = 0
    nT = len(trucks)
    while time.time() < t_end:
        it += 1
        move = random.random()

        if move < 0.6: # có thể tune
            # RELOCATE: chuyen 1 yeu cau sang vi tri khac (co the cung dau keo)
            # uu tien lay tu dau keo dang la makespan
            if no_improve % 3 == 0:
                src = max(trucks, key=lambda k: complete_c[k])
            else:
                src = random.choice(trucks)
            if not assign[src]:
                continue
            i = random.randrange(len(assign[src]))
            r = assign[src][i]
            dst = random.choice(trucks)
            old_src = assign[src]; old_dst = assign[dst]
            new_src = old_src[:i] + old_src[i+1:]
            base = new_src if dst == src else old_dst
            j = random.randrange(len(base) + 1)
            new_dst = base[:j] + [r] + base[j:]

            if dst == src:
                ntr, ntm = decode(new_dst, depot[src])
                dF2 = ntr - travel_c[src]
                nF2 = F2 + dF2
                save_c = complete_c[src]
                complete_c[src] = ntm
                nF1 = calcF1()
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
                    truck_of[r] = dst
                    F1, F2 = nF1, nF2; no_improve = 0
                else:
                    complete_c[src] = sc_s; complete_c[dst] = sc_d
                    no_improve += 1

        elif move < 0.85 and nT >= 2: # có thể tune
            # SWAP: doi cho 2 yeu cau giua 2 dau keo
            k1, k2 = random.sample(trucks, 2)
            if not assign[k1] or not assign[k2]:
                continue
            i1 = random.randrange(len(assign[k1]))
            i2 = random.randrange(len(assign[k2]))
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
                complete_c[k1] = sc1; complete_c[k2] = sc2
                no_improve += 1
        else:
            # REORDER: dao 1 doan trong cung dau keo (2-opt don gian)
            k = random.choice(trucks)
            L = assign[k]
            if len(L) < 2:
                continue
            i = random.randrange(len(L)); j = random.randrange(len(L))
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

        # luu nghiem tot nhat
        if (F1, F2) < best:
            best = (F1, F2)
            best_assign = {k: list(assign[k]) for k in trucks}

        # cu vai nghin buoc khong cai thien -> xao tron nhe (perturbation)
        if no_improve > 4000:
            for _ in range(max(1, len(rids)//20)):
                r = random.choice(rids)
                s = truck_of[r]
                assign[s].remove(r)
                d = random.choice(trucks)
                assign[d].append(r); truck_of[r] = d
                travel_c[s], complete_c[s] = decode(assign[s], depot[s])
                travel_c[d], complete_c[d] = decode(assign[d], depot[d])
            F2 = sum(travel_c.values()); F1 = calcF1()
            no_improve = 0

    # ---------------- Xuat nghiem tot nhat ----------------
    assign = best_assign
    used = [k for k in trucks if assign[k]]
    out = [f"ROUTES {len(used)}"]
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
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        try:
            TIME_LIMIT = float(sys.argv[1])
        except ValueError:
            pass
    main()
    