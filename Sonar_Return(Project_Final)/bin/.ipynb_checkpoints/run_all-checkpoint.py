"""
run_all.py – Chạy toàn bộ pipeline: EDA → Train → Test
Sử dụng: python bin/run_all.py [DATA_NAME]
Ví dụ  : python bin/run_all.py sonar

Pipeline sẽ chạy:
  1. run_eda.py   sonar
  2. run_train.py sonar
  3. run_test.py  sonar sonar   (test trên chính bộ đó)

Mỗi bước thất bại → dừng toàn bộ pipeline.
"""
import subprocess, sys, os
from datetime import datetime


def run_step(step_label, script, args, base):
    """Chạy 1 bước trong pipeline. Trả về True nếu thành công."""
    cmd = [sys.executable, os.path.join(base, 'bin', script)] + args
    print()
    print('─' * 60)
    print(f'  ▶  {step_label}')
    print(f'     {" ".join(os.path.basename(c) if i > 0 else c for i, c in enumerate(cmd))}')
    print('─' * 60)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    data_name = sys.argv[1] if len(sys.argv) > 1 else 'sonar'

    base      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    start_time = datetime.now()

    print('=' * 60)
    print('  PIPELINE: EDA → TRAIN → TEST')
    print('=' * 60)
    print(f'  Data      : {data_name}')
    print(f'  Bắt đầu   : {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    steps = [
        ('Bước 1/3 – EDA',   'run_eda.py',   [data_name]),
        ('Bước 2/3 – Train', 'run_train.py',  [data_name]),
        ('Bước 3/3 – Test',  'run_test.py',   [data_name, data_name]),
    ]

    for label, script, args in steps:
        ok = run_step(label, script, args, base)
        if not ok:
            elapsed = (datetime.now() - start_time).seconds
            print()
            print('=' * 60)
            print(f'❌ PIPELINE THẤT BẠI tại: {label}')
            print(f'   Thời gian đã chạy : {elapsed}s')
            print('=' * 60)
            sys.exit(1)

    elapsed = (datetime.now() - start_time).seconds
    print()
    print('=' * 60)
    print('✅ TOÀN BỘ PIPELINE HOÀN THÀNH!')
    print('=' * 60)
    print(f'  Data      : {data_name}')
    print(f'  Thời gian : {elapsed}s')
    print()
    print('  📁 Kết quả:')
    print(f'     exps_/{data_name}_eda_log.xlsx')
    print(f'     exps_/{data_name}_train_log.xlsx')
    print(f'     exps_/{data_name}__test_{data_name}_log.xlsx')
    print(f'     model/{data_name}/')
    print('=' * 60)


if __name__ == '__main__':
    main()