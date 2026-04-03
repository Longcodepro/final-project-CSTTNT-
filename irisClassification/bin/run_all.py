"""
bin/run_all.py  –  Chạy toàn bộ pipeline: EDA → Train → Test
Cách dùng:
    python bin/run_all.py
"""
import subprocess, sys, time, os, json
from datetime import datetime

GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
BLUE   = '\033[94m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
RESET  = '\033[0m'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_notebook(name):
    for sub in ['prj', '']:
        p = os.path.join(BASE_DIR, sub, name)
        if os.path.exists(p):
            return p
    return None

NOTEBOOKS = [
    ('01_eda.ipynb',   '🔍 EDA',   120,  600),   # (file, label, est_sec, timeout)
    ('02_train.ipynb', '🤖 Train', 600, 1800),
    ('03_test.ipynb',  '📊 Test',   90,  300),
]

def count_code_cells(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    return sum(1 for c in nb['cells'] if c['cell_type'] == 'code')

def progress_bar(done, total, width=45, label='', color=CYAN):
    pct    = done / total if total else 0
    filled = int(width * pct)
    bar    = '█' * filled + '░' * (width - filled)
    print(f'\r  {color}[{bar}]{RESET} {YELLOW}{pct*100:5.1f}%{RESET}  {label:<38}',
          end='', flush=True)

def overall_bar(step, total_steps, width=55):
    filled = int(width * step / total_steps)
    bar    = '▓' * filled + '░' * (width - filled)
    pct    = step / total_steps * 100
    print(f'\r  {BLUE}[{bar}]{RESET} {BOLD}{pct:5.1f}%{RESET}  Pipeline tổng thể',
          end='', flush=True)

def print_separator():
    print(f'\n  {DIM}{"─" * 60}{RESET}')

def print_banner():
    print(f'\n{BOLD}{CYAN}{"═" * 62}{RESET}')
    print(f'{BOLD}{CYAN}  🌸  IRIS CLASSIFICATION – FULL PIPELINE{RESET}')
    print(f'{BOLD}{CYAN}{"═" * 62}{RESET}')
    print(f'  {DIM}EDA  →  Train  →  Test{RESET}')
    print(f'  🕐 Bắt đầu: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

def run_notebook(nb_name, label, est_sec, timeout):
    nb_path = find_notebook(nb_name)
    if nb_path is None:
        print(f'\n  {RED}❌ Không tìm thấy {nb_name}{RESET}')
        return False, 0

    total_cells = count_code_cells(nb_path)
    print(f'\n  {BOLD}{label}{RESET}')
    print(f'  {DIM}📄 {nb_path}{RESET}')
    print(f'  {DIM}📊 {total_cells} code cells  |  ⏱ Ước tính ~{est_sec//60} phút{RESET}\n')

    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        f'--ExecutePreprocessor.timeout={timeout}',
        nb_path
    ]

    start = time.time()
    progress_bar(0, 100, label='Đang khởi động...')

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while proc.poll() is None:
        elapsed  = time.time() - start
        pct_time = min((elapsed / est_sec) * 100, 98)
        progress_bar(pct_time, 100, label=f'Đang xử lý... {elapsed:.0f}s')
        time.sleep(1.5)

    elapsed = time.time() - start
    _, stderr = proc.communicate()

    if proc.returncode == 0:
        progress_bar(100, 100, label='Hoàn thành!               ')
        print(f'\n  {GREEN}✅ {label} xong — {elapsed:.1f}s ({elapsed/60:.1f} phút){RESET}')
        return True, elapsed
    else:
        progress_bar(0, 100, label='LỖI                       ')
        print(f'\n\n  {RED}❌ Lỗi khi chạy {label}:{RESET}')
        err_lines = [l for l in stderr.strip().split('\n') if l.strip()]
        for line in err_lines[-25:]:
            print(f'  {RED}{line}{RESET}')
        return False, elapsed

def main():
    print_banner()

    total_steps   = len(NOTEBOOKS)
    results       = []
    total_elapsed = 0
    pipeline_start = time.time()

    for step, (nb_name, label, est_sec, timeout) in enumerate(NOTEBOOKS):
        print_separator()
        print(f'\n  {BOLD}[{step+1}/{total_steps}] {label}{RESET}')

        # Thanh tổng thể
        overall_bar(step, total_steps)
        print()

        ok, elapsed = run_notebook(nb_name, label, est_sec, timeout)
        total_elapsed += elapsed
        results.append((label, ok, elapsed))

        if not ok:
            print_separator()
            print(f'\n  {RED}{BOLD}Pipeline dừng lại do lỗi ở bước {step+1}.{RESET}')
            print(f'  {YELLOW}Hãy kiểm tra lỗi trên rồi chạy lại từ bước {step+1}.{RESET}')
            print(f'  {DIM}Lệnh chạy lại:{RESET}')
            scripts = ['run_eda.py', 'run_train.py', 'run_test.py']
            print(f'  {CYAN}python bin/{scripts[step]}{RESET}\n')
            sys.exit(1)

    # ── Tổng kết ─────────────────────────────────────────────
    print_separator()
    overall_bar(total_steps, total_steps)
    print(f'\n\n{BOLD}{GREEN}{"═" * 62}{RESET}')
    print(f'{BOLD}{GREEN}  ✅  PIPELINE HOÀN THÀNH!{RESET}')
    print(f'{BOLD}{GREEN}{"═" * 62}{RESET}')
    print(f'\n  {"Bước":<20} {"Thời gian":>10}  {"Kết quả"}')
    print(f'  {"─"*20} {"─"*10}  {"─"*8}')
    for label, ok, elapsed in results:
        status = f'{GREEN}✅ OK{RESET}' if ok else f'{RED}❌ LỖI{RESET}'
        print(f'  {label:<20} {elapsed:>8.1f}s  {status}')

    total_pipeline = time.time() - pipeline_start
    print(f'\n  ⏱  Tổng thời gian   : {total_pipeline:.1f}s ({total_pipeline/60:.1f} phút)')
    print(f'  🕐 Kết thúc         : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'\n  {DIM}Kết quả lưu trong thư mục exps_/{RESET}\n')

if __name__ == '__main__':
    main()
