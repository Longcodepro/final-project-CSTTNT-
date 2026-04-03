"""
bin/run_test.py  –  Chạy 03_test.ipynb với thanh tiến trình
Cách dùng:
    python bin/run_test.py
"""
import subprocess, sys, time, os, json
from datetime import datetime

GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'prj', '03_test.ipynb')
if not os.path.exists(NOTEBOOK):
    NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '03_test.ipynb')

def count_code_cells(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    return sum(1 for c in nb['cells'] if c['cell_type'] == 'code')

def progress_bar(done, total, width=45, label=''):
    pct    = done / total if total else 0
    filled = int(width * pct)
    bar    = '█' * filled + '░' * (width - filled)
    print(f'\r  {CYAN}[{bar}]{RESET} {YELLOW}{pct*100:5.1f}%{RESET}  {label:<35}', end='', flush=True)

def print_header(title):
    line = '═' * 60
    print(f'\n{BOLD}{CYAN}{line}{RESET}')
    print(f'{BOLD}{CYAN}  {title}{RESET}')
    print(f'{BOLD}{CYAN}{line}{RESET}')

def main():
    print_header('📊  BƯỚC 3 – TEST (03_test.ipynb)')
    print(f'  📄 File    : {os.path.basename(NOTEBOOK)}')
    print(f'  🕐 Bắt đầu : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    if not os.path.exists(NOTEBOOK):
        print(f'\n{RED}❌ Không tìm thấy: {NOTEBOOK}{RESET}')
        sys.exit(1)

    total_cells = count_code_cells(NOTEBOOK)
    print(f'  📊 Code cells: {total_cells}\n')

    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        '--ExecutePreprocessor.timeout=300',
        NOTEBOOK
    ]

    stages = [
        (0,  20,  'Khởi tạo & kiểm tra điều kiện...'),
        (20, 40,  'Load model từ pkl...'),
        (40, 70,  'Đánh giá trên tập test...'),
        (70, 90,  'Vẽ Confusion Matrix & ROC...'),
        (90, 100, 'Ghi log & tổng kết...'),
    ]

    start = time.time()
    progress_bar(0, 100, label='Đang khởi động...')

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    estimated_total = 90  # ~1.5 phút cho test
    while proc.poll() is None:
        elapsed  = time.time() - start
        pct_time = min((elapsed / estimated_total) * 100, 98)
        label = stages[0][2]
        for s_start, s_end, s_label in stages:
            if s_start <= pct_time < s_end:
                label = s_label
                break
        progress_bar(pct_time, 100, label=label)
        time.sleep(1)

    elapsed = time.time() - start
    _, stderr = proc.communicate()

    if proc.returncode == 0:
        progress_bar(100, 100, label='Hoàn thành!            ')
        print(f'\n\n  {GREEN}✅ Test hoàn thành trong {elapsed:.1f}s{RESET}')
        print(f'  🕐 Kết thúc: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    else:
        progress_bar(0, 100, label='LỖI                    ')
        print(f'\n\n  {RED}❌ Lỗi khi chạy Test:{RESET}')
        err_lines = [l for l in stderr.strip().split('\n') if l.strip()]
        for line in err_lines[-20:]:
            print(f'  {RED}{line}{RESET}')
        sys.exit(1)

if __name__ == '__main__':
    main()
