import os
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BIN_DIR  = os.path.join(BASE_DIR, 'bin')

if __name__ == '__main__':
    steps = [
        ('EDA',      os.path.join(BIN_DIR, 'run_eda.py')),
        ('Training', os.path.join(BIN_DIR, 'run_train.py')),
    ]

    for step_name, script in steps:
        print(f'\n{"="*50}')
        print(f'  BƯỚC: {step_name}')
        print(f'{"="*50}')
        result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
        if result.returncode != 0:
            print(f'\n❌ Pipeline dừng tại bước: {step_name}')
            sys.exit(1)

    print(f'\n{"="*50}')
    print('  ✅ PIPELINE HOÀN THÀNH')
    print(f'{"="*50}')

