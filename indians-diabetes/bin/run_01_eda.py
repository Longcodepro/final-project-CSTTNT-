"""
run_01_eda.py — Chạy notebook 01_eda.ipynb với thanh tiến trình.
Cách dùng: python run_01_eda.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_runner import run_notebook

if __name__ == "__main__":
    nb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_eda.ipynb")
    ok = run_notebook(nb, label="EDA — Pima Indians Diabetes")
    sys.exit(0 if ok else 1)
