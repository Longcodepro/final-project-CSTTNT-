"""
run_03_test.py — Chạy notebook 03_test.ipynb với thanh tiến trình.
Cách dùng: python run_03_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_runner import run_notebook

if __name__ == "__main__":
    nb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_test.ipynb")
    ok = run_notebook(nb, label="Test Model — Pima Indians Diabetes")
    sys.exit(0 if ok else 1)
