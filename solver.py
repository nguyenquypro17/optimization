#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Truck Container Scheduling - CP-SAT (OR-Tools)
Mo hinh hoa theo file CP_Model_TruckContainerScheduling.md

Cach dung:
    python3 solver.py input.txt              # in ket qua ra man hinh
    python3 solver.py input.txt out.txt      # ghi ket qua ra file
    python3 solver.py input.txt out.txt 30   # gioi han thoi gian 30s

Yeu cau: pip install ortools
"""
import sys
from ortools.sat.python import cp_model

# ----------------------------- Doc input -----------------------------
def parse_input(path):
    with open(path, "r", encoding="utf-8") as f:
        tokens_lines = [ln.strip() for ln in f if ln.strip() != ""]

    data = {}
    i = 0
    # Dong 1: Points N
    N = int(tokens_lines[0].split()[1]); i = 1
    # Dong 2: DISTANCES N^2
    nd = int(tokens_lines[1].split()[1]); i = 2
    # ma tran khoang cach 1-index
    t = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(nd):
        a, b, d = map(int, tokens_lines[i].split()); i += 1
        t[a][b] = d
    # TRAILER p d
    parts = tokens_lines[i].split(); i += 1
    trailer_loc = int(parts[1]); trailer_time = int(parts[2])
    # TRUCK m
    m = int(tokens_lines[i].split()[1]); i += 1
    truck_depot = {}
    for _ in range(m):
        tid, p = map(int, tokens_lines[i].split()); i += 1
        truck_depot[tid] = p
    # REQ ... cho den '#'
    reqs = {}
    while i < len(tokens_lines) and not tokens_lines[i].startswith("#"):
        p = tokens_lines[i].split(); i += 1
        # REQ id size p1 pickup_action pickup_dur p2 drop_action drop_dur
        rid = int(p[1]); size = int(p[2])
        a_loc = int(p[3]); pa = p[4]; pdur = int(p[5])
        b_loc = int(p[6]); da = p[7]; ddur = int(p[8])
        reqs[rid] = dict(size=size, q=(2 if size == 40 else 1),
                         a=a_loc, pa=pa, pdur=pdur,
                         b=b_loc, da=da, ddur=ddur)
    data.update(N=N, t=t, trailer_loc=trailer_loc, trailer_time=trailer_time,
                m=m, truck_depot=truck_depot, reqs=reqs)
    return data

# ----------------------------- Xay node -----------------------------
# Moi node: dict(kind, loc, serv, q, role, ref)
#  kind: S (start), E (end), P (pickup), D (drop), TA (attach trailer), TD (detach trailer)
#  role cho P: 'PC'|'PCT' ; cho D: 'DC'|'DCT'
def build_nodes(data):
    nodes = []
    truck_nodes = {}      # tid -> (S_idx, E_idx)
    for tid, dep in data["truck_depot"].items():
        s = len(nodes); nodes.append(dict(kind="S", loc=dep, serv=0, q=0, role=None, ref=tid))
        e = len(nodes); nodes.append(dict(kind="E", loc=dep, serv=0, q=0, role=None, ref=tid))
        truck_nodes[tid] = (s, e)

    req_nodes = {}        # rid -> (P_idx, D_idx)
    n_pc = 0; n_pct = 0
    for rid, r in data["reqs"].items():
        prole = "PC" if r["pa"] == "PICKUP_CONTAINER" else "PCT"
        drole = "DC" if r["da"] == "DROP_CONTAINER" else "DCT"
        if prole == "PC": n_pc += 1
        else: n_pct += 1
        p = len(nodes); nodes.append(dict(kind="P", loc=r["a"], serv=r["pdur"], q=r["q"], role=prole, ref=rid))
        d = len(nodes); nodes.append(dict(kind="D", loc=r["b"], serv=r["ddur"], q=r["q"], role=drole, ref=rid))
        req_nodes[rid] = (p, d)

    # Pool node ro-mooc o bai trailer.
    # Attach: du dung cho moi yeu cau PC. Detach: du tra het ro-mooc rong.
    A = n_pc
    B = n_pc + n_pct
    tloc = data["trailer_loc"]; tt = data["trailer_time"]
    attach_nodes = []
    for _ in range(A):
        idx = len(nodes); nodes.append(dict(kind="TA", loc=tloc, serv=tt, q=0, role=None, ref=None))
        attach_nodes.append(idx)
    detach_nodes = []
    for _ in range(B):
        idx = len(nodes); nodes.append(dict(kind="TD", loc=tloc, serv=tt, q=0, role=None, ref=None))
        detach_nodes.append(idx)

    return nodes, truck_nodes, req_nodes, attach_nodes, detach_nodes

# ----------------------------- Mo hinh CP-SAT -----------------------------
def solve(data, time_limit=30.0, alpha=1_000_000):
    model = cp_model.CpModel()
    t = data["t"]; trucks = list(data["truck_depot"].keys()); Q = 2
    nodes, truck_nodes, req_nodes, attach_nodes, detach_nodes = build_nodes(data)
    Nn = len(nodes)

    # shared nodes = tat ca node tru S,E cua cac truck
    start_idx = {truck_nodes[k][0] for k in trucks}
    end_idx   = {truck_nodes[k][1] for k in trucks}
    shared = [i for i in range(Nn) if i not in start_idx and i not in end_idx]

    # horizon
    maxd = max(max(row) for row in t)
    serv_sum = sum(nd["serv"] for nd in nodes)
    H = serv_sum + (Nn + 2) * maxd + 1

    # bien trang thai toan cuc (1 ban / node)
    T = [model.NewIntVar(0, H, f"T{i}") for i in range(Nn)]
    load = [model.NewIntVar(0, Q, f"L{i}") for i in range(Nn)]
    h = [model.NewBoolVar(f"h{i}") for i in range(Nn)]

    arc_lit = {}   # (k, i, j) -> BoolVar
    visit = {}     # (k, i) -> BoolVar (node shared i co thuoc lo trinh truck k)

    def loc(i): return nodes[i]["loc"]

    for k in trucks:
        Sk, Ek = truck_nodes[k]
        # tap node cua circuit truck k
        knodes = [Sk, Ek] + shared
        arcs = []
        # self-loop cho shared node = khong phuc vu boi k
        for i in shared:
            v = model.NewBoolVar(f"v_{k}_{i}")
            visit[(k, i)] = v
            arcs.append((i, i, v.Not()))
        # cac cung thuc
        def add_arc(i, j):
            a = model.NewBoolVar(f"x_{k}_{i}_{j}")
            arc_lit[(k, i, j)] = a
            arcs.append((i, j, a))
            return a
        # Sk -> {Ek} U shared
        add_arc(Sk, Ek)
        for j in shared:
            add_arc(Sk, j)
        # shared -> Ek
        for i in shared:
            add_arc(i, Ek)
        # shared -> shared
        for i in shared:
            for j in shared:
                if i != j:
                    add_arc(i, j)
        # cung dong vong Ek -> Sk
        close = model.NewBoolVar(f"close_{k}")
        arc_lit[(k, Ek, Sk)] = close
        arcs.append((Ek, Sk, close))

        model.AddCircuit(arcs)

        # trang thai dau cua truck k
        model.Add(T[Sk] == 0)
        model.Add(load[Sk] == 0)
        model.Add(h[Sk] == 0)

    # node shared duoc phuc vu boi dung 1 truck (request) hoac <=1 (trailer)
    for i in shared:
        if nodes[i]["kind"] in ("P", "D"):
            model.Add(sum(visit[(k, i)] for k in trucks) == 1)
        else:  # TA, TD optional
            model.Add(sum(visit[(k, i)] for k in trucks) <= 1)

    # pickup & drop cung truck
    for rid, (P, D) in req_nodes.items():
        for k in trucks:
            model.Add(visit[(k, P)] == visit[(k, D)])

    # ----- rang buoc theo tung cung (lan truyen thoi gian + trang thai) -----
    for (k, i, j), a in arc_lit.items():
        if j in start_idx:
            continue  # cung dong vong, khong lan truyen
        nj = nodes[j]
        # thoi gian: T[j] >= T[i] + serv(i) + t[loc_i][loc_j]
        model.Add(T[j] >= T[i] + nodes[i]["serv"] + t[loc(i)][loc(j)]).OnlyEnforceIf(a)
        kind = nj["kind"]; q = nj["q"]; role = nj["role"]
        if kind == "E":
            model.Add(h[i] == 0).OnlyEnforceIf(a)
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 0).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)
        elif kind == "P" and role == "PC":
            model.Add(h[i] == 1).OnlyEnforceIf(a)            # phai dang co ro-mooc
            model.Add(load[i] + q <= Q).OnlyEnforceIf(a)
            model.Add(h[j] == 1).OnlyEnforceIf(a)
            model.Add(load[j] == load[i] + q).OnlyEnforceIf(a)
        elif kind == "P" and role == "PCT":
            model.Add(h[i] == 0).OnlyEnforceIf(a)            # truck tran, moc ca ro-mooc co hang
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 1).OnlyEnforceIf(a)
            model.Add(load[j] == q).OnlyEnforceIf(a)
        elif kind == "D" and role == "DC":
            model.Add(h[i] == 1).OnlyEnforceIf(a)
            model.Add(load[i] >= q).OnlyEnforceIf(a)
            model.Add(h[j] == 1).OnlyEnforceIf(a)
            model.Add(load[j] == load[i] - q).OnlyEnforceIf(a)
        elif kind == "D" and role == "DCT":
            model.Add(h[i] == 1).OnlyEnforceIf(a)
            model.Add(load[i] == q).OnlyEnforceIf(a)          # ro-mooc bi bo lai mang theo container
            model.Add(h[j] == 0).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)
        elif kind == "TA":                                    # gan ro-mooc rong
            model.Add(h[i] == 0).OnlyEnforceIf(a)
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 1).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)
        elif kind == "TD":                                    # tra ro-mooc rong
            model.Add(h[i] == 1).OnlyEnforceIf(a)
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 0).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)

    # precedence: lay truoc tra
    for rid, (P, D) in req_nodes.items():
        nP, nD = nodes[P], nodes[D]
        model.Add(T[D] >= T[P] + nP["serv"] + t[nP["loc"]][nD["loc"]])

    # ----- ham muc tieu -----
    Cs = [T[truck_nodes[k][1]] for k in trucks]
    F1 = model.NewIntVar(0, H, "F1")
    model.AddMaxEquality(F1, Cs)
    F2_terms = []
    for (k, i, j), a in arc_lit.items():
        d = t[loc(i)][loc(j)]
        if d != 0:
            F2_terms.append(d * a)
    F2 = model.NewIntVar(0, H * len(trucks), "F2")
    model.Add(F2 == sum(F2_terms))
    model.Minimize(alpha * F1 + F2)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status, solver, None

    # ----- truy vet lo trinh -----
    routes = {}
    for k in trucks:
        Sk, Ek = truck_nodes[k]
        nxt = {}
        for (kk, i, j), a in arc_lit.items():
            if kk == k and j not in start_idx and solver.Value(a) == 1:
                nxt[i] = j
        seq = []
        cur = Sk
        guard = 0
        while cur in nxt and guard <= Nn + 2:
            nx = nxt[cur]
            if nx == Ek:
                break
            seq.append(nx)
            cur = nx
            guard += 1
        routes[k] = seq

    info = dict(F1=solver.Value(F1), F2=solver.Value(F2),
                obj=solver.ObjectiveValue(), Cs={k: solver.Value(c) for k, c in zip(trucks, Cs)})
    return routes, status, solver, (nodes, truck_nodes, info)

# ----------------------------- Xuat output -----------------------------
ACTION_OUT = {
    ("P", "PC"): "PICKUP_CONTAINER",
    ("P", "PCT"): "PICKUP_CONTAINER_TRAILER",
    ("D", "DC"): "DROP_CONTAINER",
    ("D", "DCT"): "DROP_CONTAINER_TRAILER",
    ("TA", None): "PICKUP_TRAILER",
    ("TD", None): "DROP_TRAILER",
}

def format_output(data, routes, nodes, truck_nodes):
    trucks = list(data["truck_depot"].keys())
    lines = [f"ROUTES {len(trucks)}"]
    for k in trucks:
        lines.append(f"TRUCK {k}")
        for idx in routes[k]:
            nd = nodes[idx]
            act = ACTION_OUT[(nd["kind"], nd["role"])]
            if nd["kind"] in ("TA", "TD"):
                lines.append(f"{nd['loc']} {act}")
            else:
                lines.append(f"{nd['loc']} {act} {nd['ref']}")
        lines.append(f"{data['truck_depot'][k]} STOP")
        lines.append("#")
    return "\n".join(lines)

# ----------------------------- main -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 solver.py input.txt [output.txt] [time_limit_s]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    tl = float(sys.argv[3]) if len(sys.argv) >= 4 else 30.0

    data = parse_input(inp)
    routes, status, solver, extra = solve(data, time_limit=tl)
    if routes is None:
        msg = f"KHONG TIM DUOC LO TRINH (status={solver.StatusName(status)})"
        print(msg)
        if out:
            open(out, "w").write(msg + "\n")
        return
    nodes, truck_nodes, info = extra
    text = format_output(data, routes, nodes, truck_nodes)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    #print(f"\n# F1={info['F1']}  F2={info['F2']}  status={solver.StatusName(status)}", file=sys.stderr)

if __name__ == "__main__":
    main()
