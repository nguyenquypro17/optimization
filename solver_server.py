import sys
from ortools.sat.python import cp_model

def parse_input():
    tokens = sys.stdin.read().split() 
    
    N = int(tokens[1])
    nd = int(tokens[3])
    idx = 4
    
    t = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(nd):
        a, b, d = int(tokens[idx]), int(tokens[idx+1]), int(tokens[idx+2])
        t[a][b] = d
        idx += 3
        
    trailer_loc, trailer_time = int(tokens[idx+1]), int(tokens[idx+2])
    idx += 3
    
    m = int(tokens[idx+1])
    idx += 2
    truck_depot = {}
    for _ in range(m):
        truck_depot[int(tokens[idx])] = int(tokens[idx+1])
        idx += 2
        
    reqs = {}
    while idx < len(tokens) and tokens[idx] == "REQ":
        rid, size = int(tokens[idx+1]), int(tokens[idx+2])
        reqs[rid] = dict(size=size, q=(2 if size == 40 else 1),
                         a=int(tokens[idx+3]), pa=tokens[idx+4], pdur=int(tokens[idx+5]),
                         b=int(tokens[idx+6]), da=tokens[idx+7], ddur=int(tokens[idx+8]))
        idx += 9
        
    return dict(N=N, t=t, trailer_loc=trailer_loc, trailer_time=trailer_time, m=m, truck_depot=truck_depot, reqs=reqs)

def build_nodes(data):
    nodes = []
    truck_nodes = {}
    for tid, dep in data["truck_depot"].items():
        s = len(nodes); nodes.append(dict(kind="S", loc=dep, serv=0, q=0, role=None, ref=tid))
        e = len(nodes); nodes.append(dict(kind="E", loc=dep, serv=0, q=0, role=None, ref=tid))
        truck_nodes[tid] = (s, e)

    req_nodes = {}
    n_pc = 0; n_pct = 0
    for rid, r in data["reqs"].items():
        prole = "PC" if r["pa"] == "PICKUP_CONTAINER" else "PCT"
        drole = "DC" if r["da"] == "DROP_CONTAINER" else "DCT"
        if prole == "PC": n_pc += 1
        else: n_pct += 1
        p = len(nodes); nodes.append(dict(kind="P", loc=r["a"], serv=r["pdur"], q=r["q"], role=prole, ref=rid))
        d = len(nodes); nodes.append(dict(kind="D", loc=r["b"], serv=r["ddur"], q=r["q"], role=drole, ref=rid))
        req_nodes[rid] = (p, d)

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

def solve(data, time_limit=10.0, alpha=1_000_000):
    model = cp_model.CpModel()
    t = data["t"]; trucks = list(data["truck_depot"].keys()); Q = 2
    nodes, truck_nodes, req_nodes, attach_nodes, detach_nodes = build_nodes(data)
    Nn = len(nodes)

    start_idx = {truck_nodes[k][0] for k in trucks}
    end_idx   = {truck_nodes[k][1] for k in trucks}
    shared = [i for i in range(Nn) if i not in start_idx and i not in end_idx]

    maxd = max(max(row) for row in t)
    serv_sum = sum(nd["serv"] for nd in nodes)
    H = serv_sum + (Nn + 2) * maxd + 1

    T = [model.NewIntVar(0, H, f"T{i}") for i in range(Nn)]
    load = [model.NewIntVar(0, Q, f"L{i}") for i in range(Nn)]
    h = [model.NewBoolVar(f"h{i}") for i in range(Nn)]

    arc_lit = {}
    visit = {}

    def loc(i): return nodes[i]["loc"]

    for k in trucks:
        Sk, Ek = truck_nodes[k]
        arcs = []
        for i in shared:
            v = model.NewBoolVar(f"v_{k}_{i}")
            visit[(k, i)] = v
            arcs.append((i, i, v.Not()))
        
        def add_arc(i, j):
            a = model.NewBoolVar(f"x_{k}_{i}_{j}")
            arc_lit[(k, i, j)] = a
            arcs.append((i, j, a))
            return a
            
        add_arc(Sk, Ek)
        for j in shared: add_arc(Sk, j)
        for i in shared: add_arc(i, Ek)
        for i in shared:
            for j in shared:
                if i != j: add_arc(i, j)
                
        close = model.NewBoolVar(f"close_{k}")
        arc_lit[(k, Ek, Sk)] = close
        arcs.append((Ek, Sk, close))

        model.AddCircuit(arcs)

        model.Add(T[Sk] == 0)
        model.Add(load[Sk] == 0)
        model.Add(h[Sk] == 0)

    for i in shared:
        if nodes[i]["kind"] in ("P", "D"):
            model.Add(sum(visit[(k, i)] for k in trucks) == 1)
        else:
            model.Add(sum(visit[(k, i)] for k in trucks) <= 1)

    for rid, (P, D) in req_nodes.items():
        for k in trucks:
            model.Add(visit[(k, P)] == visit[(k, D)])

    for (k, i, j), a in arc_lit.items():
        if j in start_idx: continue
        nj = nodes[j]
        model.Add(T[j] >= T[i] + nodes[i]["serv"] + t[loc(i)][loc(j)]).OnlyEnforceIf(a)
        kind = nj["kind"]; q = nj["q"]; role = nj["role"]
        
        if kind == "E":
            model.Add(h[i] == 0).OnlyEnforceIf(a)
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 0).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)
        elif kind == "P" and role == "PC":
            model.Add(h[i] == 1).OnlyEnforceIf(a)
            model.Add(load[i] + q <= Q).OnlyEnforceIf(a)
            model.Add(h[j] == 1).OnlyEnforceIf(a)
            model.Add(load[j] == load[i] + q).OnlyEnforceIf(a)
        elif kind == "P" and role == "PCT":
            model.Add(h[i] == 0).OnlyEnforceIf(a)
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
            model.Add(load[i] == q).OnlyEnforceIf(a)
            model.Add(h[j] == 0).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)
        elif kind == "TA":
            model.Add(h[i] == 0).OnlyEnforceIf(a)
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 1).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)
        elif kind == "TD":
            model.Add(h[i] == 1).OnlyEnforceIf(a)
            model.Add(load[i] == 0).OnlyEnforceIf(a)
            model.Add(h[j] == 0).OnlyEnforceIf(a)
            model.Add(load[j] == 0).OnlyEnforceIf(a)

    for rid, (P, D) in req_nodes.items():
        nP, nD = nodes[P], nodes[D]
        model.Add(T[D] >= T[P] + nP["serv"] + t[nP["loc"]][nD["loc"]])

    Cs = [T[truck_nodes[k][1]] for k in trucks]
    F1 = model.NewIntVar(0, H, "F1")
    model.AddMaxEquality(F1, Cs)
    
    F2_terms = []
    for (k, i, j), a in arc_lit.items():
        d = t[loc(i)][loc(j)]
        if d != 0: F2_terms.append(d * a)
        
    F2 = model.NewIntVar(0, H * len(trucks), "F2")
    model.Add(F2 == sum(F2_terms))
    model.Minimize(alpha * F1 + F2)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status, solver, None

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
            if nx == Ek: break
            seq.append(nx)
            cur = nx
            guard += 1
        routes[k] = seq

    info = dict(F1=solver.Value(F1), F2=solver.Value(F2),
                obj=solver.ObjectiveValue(), Cs={k: solver.Value(c) for k, c in zip(trucks, Cs)})
    return routes, status, solver, (nodes, truck_nodes, info)

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
    used_trucks = [k for k in trucks if routes[k]]
    
    lines = [f"ROUTES {len(used_trucks)}"]
    for k in used_trucks:
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

def main():
    data = parse_input()
    if not data: return
    routes, status, solver, extra = solve(data, time_limit=10.0)
    if routes: 
        print(format_output(data, routes, extra[0], extra[1]))

if __name__ == "__main__":
    main()