"""
bin/run_eda.py  –  Chạy 01_eda.ipynb với thanh tiến trình
Cách dùng:
    python bin/run_eda.py
"""
import subprocess, sys, time, os, json
from datetime import datetime

# ── Màu terminal ─────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'prj', '01_eda.ipynb')
if not os.path.exists(NOTEBOOK):
    NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '01_eda.ipynb')

# ── Đọc tổng số cell code để tính % ─────────────────────────
def count_code_cells(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    return sum(1 for c in nb['cells'] if c['cell_type'] == 'code')

# ── Thanh tiến trình ─────────────────────────────────────────
def progress_bar(done, total, width=45, label=''):
    pct   = done / total if total else 0
    filled = int(width * pct)
    bar   = '█' * filled + '░' * (width - filled)
    print(f'\r  {CYAN}[{bar}]{RESET} {YELLOW}{pct*100:5.1f}%{RESET}  {label:<30}', end='', flush=True)

def print_header(title):
    line = '═' * 60
    print(f'\n{BOLD}{CYAN}{line}{RESET}')
    print(f'{BOLD}{CYAN}  {title}{RESET}')
    print(f'{BOLD}{CYAN}{line}{RESET}')

# ── Main ─────────────────────────────────────────────────────
def main():
    print_header('🔍  BƯỚC 1 – EDA (01_eda.ipynb)')
    print(f'  📄 File    : {os.path.basename(NOTEBOOK)}')
    print(f'  🕐 Bắt đầu : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    if not os.path.exists(NOTEBOOK):
        print(f'\n{RED}❌ Không tìm thấy: {NOTEBOOK}{RESET}')
        sys.exit(1)

    total_cells = count_code_cells(NOTEBOOK)
    print(f'  📊 Code cells: {total_cells}\n')

    # nbconvert chạy notebook và lưu output vào chính file đó
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        '--ExecutePreprocessor.timeout=600',
        NOTEBOOK
    ]

    start = time.time()
    progress_bar(0, total_cells, label='Đang khởi động...')

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Giả lập tiến trình – nbconvert không stream từng cell
    # nên dùng elapsed time để ước tính
    estimated_total = 120  # giây ước tính cho EDA
    done = 0
    while proc.poll() is None:
        elapsed = time.time() - start
        estimated_done = min(int((elapsed / estimated_total) * total_cells), total_cells - 1)
        label = 'Đang phân tích dữ liệu...' if elapsed < 30 else \
                'Đang vẽ biểu đồ...'        if elapsed < 60 else \
                'Đang chuẩn hóa & lưu...'
        progress_bar(estimated_done, total_cells, label=label)
        time.sleep(1)

    elapsed = time.time() - start
    _, stderr = proc.communicate()

    if proc.returncode == 0:
        progress_bar(total_cells, total_cells, label='Hoàn thành!       ')
        print(f'\n\n  {GREEN}✅ EDA hoàn thành trong {elapsed:.1f}s{RESET}')
        print(f'  🕐 Kết thúc: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    else:
        progress_bar(0, total_cells, label='LỖI               ')
        print(f'\n\n  {RED}❌ Lỗi khi chạy EDA:{RESET}')
        # In 20 dòng cuối stderr
        err_lines = [l for l in stderr.strip().split('\n') if l.strip()]
        for line in err_lines[-20:]:
            print(f'  {RED}{line}{RESET}')
        sys.exit(1)

if __name__ == '__main__':
    main()
