"""
run_02_train.py — Chạy notebook 02_train.ipynb với thanh tiến trình.
Cách dùng: python run_02_train.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _notebook_runner import run_notebook

if __name__ == "__main__":
    nb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_train.ipynb")
    ok = run_notebook(nb, label="Train Model — Pima Indians Diabetes")
    sys.exit(0 if ok else 1)
