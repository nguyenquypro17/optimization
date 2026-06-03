"""
Generator test case cho bai toan Truck Container Scheduling.

Cu phap:
    python3 gen.py [options] [output_file]

Options:
    -N <int>        So diem (Points), mac dinh: 10
    -r <int>        So yeu cau (requests), mac dinh: 5
    -k <int>        So dau keo (trucks), mac dinh: 3
    -s <int>        Random seed, mac dinh: 42
    --sparse        Chi sinh N*(N-1) cap (khong co duong di mat dinh), mac dinh: full N^2
    --coord         Dung toa do 2D Euclid (thuc te hon), mac dinh: random matrix

Vi du:
    python3 gen.py -N 15 -r 8 -k 4 output.txt
    python3 gen.py -N 20 -r 10 -k 5 -s 123 --coord test.txt
    python3 gen.py                          # in ra stdout
"""

import sys
import math
import random
import argparse


def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-N", type=int, default=10, help="So diem")
    ap.add_argument("-r", type=int, default=5,  help="So yeu cau")
    ap.add_argument("-k", type=int, default=3,  help="So dau keo")
    ap.add_argument("-s", type=int, default=42, help="Random seed")
    ap.add_argument("--coord",  action="store_true", help="Dung toa do 2D Euclid")
    ap.add_argument("--sparse", action="store_true", help="Bo qua duong tu dinh toi chinh no (=0)")
    ap.add_argument("output", nargs="?", default=None, help="File dau ra (mac dinh stdout)")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()
    if args.help:
        print(__doc__)
        sys.exit(0)
    return args


def build_distance_matrix(N, use_coord, rng):
    """Tra ve t[1..N][1..N] (1-indexed)."""
    t = [[0] * (N + 1) for _ in range(N + 1)]

    if use_coord:
        # Toa do ngau nhien trong [0, 100]^2
        xs = [0] + [rng.uniform(0, 100) for _ in range(N)]
        ys = [0] + [rng.uniform(0, 100) for _ in range(N)]
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j:
                    d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    t[i][j] = max(1, round(d * 6))   # scale ~1-850
    else:
        # Ma tran ngau nhien, thoa man bat dang thuc tam giac xap xi
        base = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                d = rng.randint(10, 100)
                base[i][j] = d
                base[j][i] = d
        t = base

    return t


def gen_testcase(N, num_req, num_truck, seed, use_coord):
    rng = random.Random(seed)
    points = list(range(1, N + 1))

    # ---- Ma tran khoang cach ----
    t = build_distance_matrix(N, use_coord, rng)

    # ---- Bai ro-mooc ----
    trailer_loc = rng.randint(1, N)
    trailer_time = rng.randint(100, 400)   # thoi gian gan/tha ro-mooc

    # ---- Dau keo ----
    # Moi dau keo co the xuat phat tu bai khac nhau
    truck_depots = [rng.randint(1, N) for _ in range(num_truck)]

    # ---- Yeu cau ----
    pickup_actions = ["PICKUP_CONTAINER", "PICKUP_CONTAINER_TRAILER"]
    drop_actions   = ["DROP_CONTAINER",   "DROP_CONTAINER_TRAILER"]

    reqs = []
    for rid in range(1, num_req + 1):
        size = rng.choice([20, 40])
        a_loc = rng.randint(1, N)
        b_loc = rng.randint(1, N)
        # Tranh pickup == drop de testcase co y nghia hon
        while b_loc == a_loc and N > 1:
            b_loc = rng.randint(1, N)
        pa   = rng.choice(pickup_actions)
        da   = rng.choice(drop_actions)
        pdur = rng.randint(100, 600)
        ddur = rng.randint(100, 600)
        reqs.append((rid, size, a_loc, pa, pdur, b_loc, da, ddur))

    return t, trailer_loc, trailer_time, truck_depots, reqs


def format_output(N, t, trailer_loc, trailer_time, truck_depots, reqs):
    lines = []
    lines.append(f"Points {N}")

    # Distances N^2 dong
    lines.append(f"DISTANCES {N * N}")
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lines.append(f"{i} {j} {t[i][j]}")

    lines.append(f"TRAILER {trailer_loc} {trailer_time}")

    num_truck = len(truck_depots)
    lines.append(f"TRUCK {num_truck}")
    for idx, dep in enumerate(truck_depots, start=1):
        lines.append(f"{idx} {dep}")

    for rid, size, a_loc, pa, pdur, b_loc, da, ddur in reqs:
        lines.append(f"REQ {rid} {size} {a_loc} {pa} {pdur} {b_loc} {da} {ddur}")

    lines.append("#")
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.r < 1:
        sys.exit("Loi: so yeu cau phai >= 1")
    if args.k < 1:
        sys.exit("Loi: so dau keo phai >= 1")
    if args.N < 2:
        sys.exit("Loi: so diem phai >= 2")

    t, trailer_loc, trailer_time, truck_depots, reqs = gen_testcase(
        N=args.N,
        num_req=args.r,
        num_truck=args.k,
        seed=args.s,
        use_coord=args.coord,
    )

    out = format_output(args.N, t, trailer_loc, trailer_time, truck_depots, reqs)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        # In thong tin tom tat ra stderr
        print(f"[gen] Points={args.N}, Requests={args.r}, Trucks={args.k}, "
              f"seed={args.s}, coord={'yes' if args.coord else 'no'} -> {args.output}",
              file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
