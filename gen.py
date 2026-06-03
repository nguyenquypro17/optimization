import random
import sys
import os

def generate(N, r, k, seed, outpath):
    rng = random.Random(seed)

    lines = []

    # Header
    nd = N * N
    lines.append(f"Points {N}")
    lines.append(f"DISTANCES {nd}")

    # Full distance matrix (self = 0, others 1-100)
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            d = 0 if a == b else rng.randint(1, 100)
            lines.append(f"{a} {b} {d}")

    # Trailer
    trailer_loc = rng.randint(1, N)
    trailer_time = rng.randint(30, 300)
    lines.append(f"TRAILER {trailer_loc} {trailer_time}")

    # Trucks
    lines.append(f"TRUCK {k}")
    for truck_id in range(1, k + 1):
        depot = rng.randint(1, N)
        lines.append(f"{truck_id} {depot}")

    # Requests
    pickup_actions = ["PICKUP_CONTAINER", "PICKUP_CONTAINER_TRAILER"]
    drop_actions = ["DROP_CONTAINER", "DROP_CONTAINER_TRAILER"]
    for rid in range(1, r + 1):
        size = rng.choice([20, 40])
        a = rng.randint(1, N)
        pa = rng.choice(pickup_actions)
        pdur = rng.randint(100, 999)
        b = rng.randint(1, N)
        da = rng.choice(drop_actions)
        ddur = rng.randint(100, 999)
        lines.append(f"REQ {rid} {size} {a} {pa} {pdur} {b} {da} {ddur}")

    lines.append("#")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {outpath}  (seed={seed})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  type=int, default=10,   help="Starting testcase index")
    parser.add_argument("--count",  type=int, default=11,   help="Number of testcases to generate")
    parser.add_argument("--N",      type=int, default=None, help="Fixed N (overrides ramp)")
    parser.add_argument("--r",      type=int, default=None, help="Fixed r (overrides ramp)")
    parser.add_argument("--k",      type=int, default=None, help="Fixed k (overrides ramp)")
    parser.add_argument("--N0",     type=int, default=1200, help="Ramp start N")
    parser.add_argument("--N1",     type=int, default=1200, help="Ramp end N")
    parser.add_argument("--r0",     type=int, default=1500, help="Ramp start r")
    parser.add_argument("--r1",     type=int, default=1500, help="Ramp end r")
    parser.add_argument("--k0",     type=int, default=600,  help="Ramp start k")
    parser.add_argument("--k1",     type=int, default=600,  help="Ramp end k")
    parser.add_argument("--seed",   type=int, default=42,   help="Base random seed")
    parser.add_argument("--outdir", type=str, default="testcase")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n = args.count

    for i in range(n):
        idx = args.start + i
        seed = args.seed + i * 137

        # Linear ramp if not fixed
        frac = i / max(n - 1, 1)
        N = args.N if args.N else round(args.N0 + frac * (args.N1 - args.N0))
        r = args.r if args.r else round(args.r0 + frac * (args.r1 - args.r0))
        k = args.k if args.k else round(args.k0 + frac * (args.k1 - args.k0))

        outpath = os.path.join(args.outdir, f"test{idx}.txt")
        generate(N, r, k, seed, outpath)

    print("Done.")
