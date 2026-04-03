"""
run_all.py — Chạy tuần tự 3 notebook: EDA → Train → Test với thanh tiến trình.
Cách dùng: python run_all.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_runner import run_notebook, BOLD, GREEN, RED, CYAN, YELLOW, RESET

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("01_eda.ipynb",    "EDA — Pima Indians Diabetes"),
    ("02_train.ipynb",  "Train Model — Pima Indians Diabetes"),
    ("03_test.ipynb",   "Test Model — Pima Indians Diabetes"),
]

if __name__ == "__main__":
    total_steps = len(STEPS)
    t_all = time.time()

    print(f"\n{BOLD}{CYAN}{'═'*62}{RESET}")
    print(f"{BOLD}  PIPELINE  —  Pima Indians Diabetes  ({total_steps} bước){RESET}")
    print(f"{BOLD}{CYAN}{'═'*62}{RESET}")

    results = []
    for idx, (fname, label) in enumerate(STEPS, 1):
        nb_path = os.path.join(BASE, fname)
        print(f"\n{YELLOW}{BOLD}[{idx}/{total_steps}]{RESET}  {label}")
        if not os.path.exists(nb_path):
            print(f"  {RED}✗ Không tìm thấy file: {nb_path}{RESET}")
            results.append((fname, False, 0))
            print(f"\n{RED}{BOLD}Pipeline dừng do thiếu file.{RESET}")
            break
        t0  = time.time()
        ok  = run_notebook(nb_path, label=label)
        elapsed = time.time() - t0
        results.append((fname, ok, elapsed))
        if not ok:
            print(f"\n{RED}{BOLD}Pipeline dừng do lỗi ở bước {idx}: {fname}{RESET}")
            break

    # ── Tổng kết ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_all
    print(f"\n{BOLD}{CYAN}{'═'*62}{RESET}")
    print(f"{BOLD}  KẾT QUẢ PIPELINE{RESET}")
    print(f"{BOLD}{CYAN}{'─'*62}{RESET}")
    all_ok = True
    for fname, ok, elapsed in results:
        icon   = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        status = f"{GREEN}OK{RESET}"  if ok else f"{RED}FAILED{RESET}"
        print(f"  {icon}  {fname:20s}  {status}   ({elapsed:.1f}s)")
        if not ok:
            all_ok = False

    not_run = [s for s in STEPS if s[0] not in [r[0] for r in results]]
    for fname, _ in not_run:
        print(f"  {YELLOW}–{RESET}  {fname:20s}  {YELLOW}SKIPPED{RESET}")

    print(f"{BOLD}{CYAN}{'─'*62}{RESET}")
    print(f"  Tổng thời gian: {total_elapsed:.1f}s")
    if all_ok and len(results) == total_steps:
        print(f"\n  {GREEN}{BOLD}✓ Pipeline hoàn thành thành công!{RESET}")
    else:
        print(f"\n  {RED}{BOLD}✗ Pipeline kết thúc với lỗi.{RESET}")
    print(f"{BOLD}{CYAN}{'═'*62}{RESET}\n")

    sys.exit(0 if (all_ok and len(results) == total_steps) else 1)
