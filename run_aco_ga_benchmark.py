"""Chay ACO va GA tren tat ca testcase, ghi ket qua vao 1 file txt."""
import os
import sys
import time
import subprocess
import tempfile

from check import check

# Theo CP_Model: min F = alpha*F1 + F2, Score = 10^9 - F (solver.py dung alpha=1e6)
ALPHA = 1_000_000
SCORE_BASE = 10**9

TESTCASE_DIR = "testcase"
OUT_FILE = "aco_ga_results.txt"
TIME_LIMIT = 9.0


def objective(f1, f2, alpha=ALPHA):
    f = alpha * f1 + f2
    return f, SCORE_BASE - f
if len(sys.argv) >= 2:
    try:
        TIME_LIMIT = float(sys.argv[1])
    except ValueError:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
SOLVERS = [
    ("ACO", "solver_aco.py"),
    ("GA", "solver_ga.py"),
]


def run_solver(solver_path, inp_path, time_limit):
    with open(inp_path, "r", encoding="utf-8") as f:
        inp = f.read()
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, solver_path, str(time_limit)],
        input=inp,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=time_limit + 30,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        return None, elapsed, f"exit {proc.returncode}: {err}"
    return proc.stdout, elapsed, None


def main():
    testcase_dir = os.path.join(ROOT, TESTCASE_DIR)
    testcases = sorted(
        os.path.join(testcase_dir, f)
        for f in os.listdir(testcase_dir)
        if f.endswith(".txt")
    )
    if not testcases:
        print(f"Khong tim thay testcase trong {testcase_dir}")
        sys.exit(1)

    lines = []
    lines.append("=" * 72)
    lines.append("BENCHMARK ACO vs GA - Truck Container Scheduling")
    lines.append(f"Time limit moi lan chay: {TIME_LIMIT}s")
    lines.append(f"So testcase: {len(testcases)}")
    lines.append(f"Ham muc tieu: F = {ALPHA}*F1 + F2,  Score = {SCORE_BASE} - F")
    lines.append("=" * 72)
    lines.append("")

    summary = {
        name: {"f": 0, "f1": 0, "f2": 0, "time": 0.0, "ok": 0, "fail": 0}
        for name, _ in SOLVERS
    }

    for tc_path in testcases:
        name = os.path.basename(tc_path)
        lines.append(f"--- {name} ---")
        print(f"Running {name} ...", flush=True)

        for algo, script in SOLVERS:
            solver_path = os.path.join(ROOT, script)
            out, elapsed, err = run_solver(solver_path, tc_path, TIME_LIMIT)
            if err:
                lines.append(f"  {algo}: LOI - {err}")
                summary[algo]["fail"] += 1
                print(f"  {algo}: FAIL", flush=True)
                continue

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(out)
                tmp_path = tmp.name
            try:
                f1, f2 = check(tc_path, tmp_path)
            except Exception as e:
                lines.append(f"  {algo}: LOI kiem tra - {e}")
                summary[algo]["fail"] += 1
                print(f"  {algo}: INVALID", flush=True)
                continue
            finally:
                os.unlink(tmp_path)

            f_val, score = objective(f1, f2)
            lines.append(
                f"  {algo}: F={f_val}  Score={score}  "
                f"(F1={f1}, F2={f2})  time={elapsed:.2f}s"
            )
            summary[algo]["f"] += f_val
            summary[algo]["f1"] += f1
            summary[algo]["f2"] += f2
            summary[algo]["time"] += elapsed
            summary[algo]["ok"] += 1
            print(f"  {algo}: F={f_val} ({elapsed:.1f}s)", flush=True)

        lines.append("")

    lines.append("=" * 72)
    lines.append("TONG KET (chi testcase hop le)")
    lines.append("=" * 72)
    for algo, _ in SOLVERS:
        s = summary[algo]
        n = s["ok"]
        if n:
            f_tong = s["f"]
            lines.append(
                f"  {algo}: F_tong={f_tong}  F_tb={f_tong/n:.0f}  "
                f"Score_tong={SCORE_BASE*n - f_tong}  "
                f"(F1_tong={s['f1']}, F2_tong={s['f2']})  "
                f"time_tong={s['time']:.1f}s  ok={n}  fail={s['fail']}"
            )
        else:
            lines.append(f"  {algo}: khong co ket qua hop le (fail={s['fail']})")

    out_path = os.path.join(ROOT, OUT_FILE)
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print()
    print(f"Da ghi ket qua: {out_path}")
    print(text)


if __name__ == "__main__":
    main()
