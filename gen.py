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
    N = 1200
    r = 1500
    k = 600
    base_seed = 42
    out_dir = "testcase"
    os.makedirs(out_dir, exist_ok=True)

    # Determine starting index (existing test files: test1..test9 => start at 10)
    start_idx = 10
    num_cases = 11

    for i in range(num_cases):
        idx = start_idx + i
        seed = base_seed + i * 137
        outpath = os.path.join(out_dir, f"test{idx}.txt")
        generate(N, r, k, seed, outpath)

    print("Done.")
