"""
run_test.py – Chạy 03_test.ipynb qua papermill
Sử dụng: python bin/run_test.py [TRAIN_DATA_NAME] [TEST_DATA_NAME]
Ví dụ  : python bin/run_test.py sonar sonar        (test trên chính bộ train)
         python bin/run_test.py sonar ionosphere    (cross-test)

Lưu ý  : Tự động kiểm tra model và EDA của bộ test trước khi gọi papermill.
"""
import subprocess, sys, os
from datetime import datetime


def check_test_ready(base, train_name, test_name):
    """Kiểm tra đủ điều kiện để test."""
    exp_dir   = os.path.join(base, 'exps_')
    model_dir = os.path.join(base, 'model', train_name)
    errors    = []

    # 1. Kiểm tra model folder và .pkl
    if not os.path.isdir(model_dir):
        errors.append(f'[THIẾU MODEL FOLDER] model/{train_name}/ chưa tồn tại')
    else:
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if len(pkl_files) == 0:
            errors.append(f'[THIẾU MODEL] Không có file .pkl trong model/{train_name}/')

    # 2. Kiểm tra EDA của bộ test
    test_eda = os.path.join(exp_dir, f'{test_name}_eda_log.xlsx')
    if not os.path.exists(test_eda):
        errors.append(f'[CHƯA EDA TEST DATA] {os.path.basename(test_eda)} chưa tồn tại')

    # 3. Kiểm tra CSV test
    for sn in ['raw', 'minmax', 'standard']:
        fp = os.path.join(exp_dir, f'{test_name}_test_{sn}.csv')
        if not os.path.exists(fp):
            errors.append(f'[THIẾU CSV] {test_name}_test_{sn}.csv')

    return errors


def main():
    train_name = sys.argv[1] if len(sys.argv) > 1 else 'sonar'
    test_name  = sys.argv[2] if len(sys.argv) > 2 else train_name

    base   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nb_in  = os.path.join(base, 'prj', '03_test.ipynb')
    nb_out = os.path.join(base, 'prj',
                          f'03_test__{train_name}__test_{test_name}_output.ipynb')

    # Kiểm tra notebook tồn tại
    if not os.path.exists(nb_in):
        print(f'❌ Không tìm thấy: {nb_in}')
        sys.exit(1)

    # Kiểm tra điều kiện sớm
    errors = check_test_ready(base, train_name, test_name)
    if errors:
        print('=' * 65)
        print(f'❌ TỪ CHỐI TEST – Các điều kiện chưa đủ!')
        print('=' * 65)
        print(f'   Train model : {train_name}')
        print(f'   Test data   : {test_name}')
        print(f'   Lỗi phát hiện:')
        for e in errors:
            print(f'      {e}')
        print()
        if any('MODEL' in e for e in errors):
            print(f'   → Hãy train trước:')
            print(f'     python bin/run_train.py {train_name}')
        if any('EDA' in e or 'CSV' in e for e in errors):
            print(f'   → Hãy EDA bộ test trước:')
            print(f'     python bin/run_eda.py {test_name}')
        print('=' * 65)
        sys.exit(1)

    print('=' * 65)
    print(f'  BƯỚC TEST')
    print('=' * 65)
    print(f'  Thời gian    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Train model  : {train_name}  (load từ model/{train_name}/)')
    print(f'  Test data    : {test_name}   (đọc từ exps_/{test_name}_test_*.csv)')
    print(f'  Output       : prj/03_test__{train_name}__test_{test_name}_output.ipynb')
    print(f'  Check model  : ✅  Check EDA test: ✅')
    print('=' * 65)

    cmd = [
        sys.executable, '-m', 'papermill',
        nb_in, nb_out,
        '-p', 'TRAIN_DATA_NAME', train_name,
        '-p', 'TEST_DATA_NAME',  test_name,
        '--cwd', base,
        '--log-output',
    ]
    result = subprocess.run(cmd)

    print()
    if result.returncode == 0:
        print('=' * 65)
        print(f'✅ TEST HOÀN THÀNH!')
        print('=' * 65)
        print(f'  Test log  : exps_/{train_name}__test_{test_name}_log.xlsx')
        print(f'  Biểu đồ  : exps_/{train_name}__test_{test_name}_plot_*.png')
        print('=' * 65)
    else:
        print('=' * 65)
        print(f'❌ LỖI khi chạy test!')
        print(f'   Xem chi tiết tại: prj/03_test__{train_name}__test_{test_name}_output.ipynb')
        print('=' * 65)
        sys.exit(1)


if __name__ == '__main__':
    main()