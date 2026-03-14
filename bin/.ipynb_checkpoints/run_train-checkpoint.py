"""
run_train.py – Chạy 02_train.ipynb qua papermill
Sử dụng: python bin/run_train.py [TRAIN_DATA_NAME]
Ví dụ  : python bin/run_train.py sonar

Lưu ý  : Tự động kiểm tra EDA đã chạy chưa trước khi gọi papermill.
          Nếu chưa EDA sẽ báo lỗi và từ chối train ngay tại đây
          (không cần đợi papermill khởi động).
"""
import subprocess, sys, os
from datetime import datetime


def check_eda_ready(base, train_name):
    """Kiểm tra EDA đã chạy đủ cho TRAIN_DATA_NAME chưa."""
    exp_dir      = os.path.join(base, 'exps_')
    eda_log      = os.path.join(exp_dir, f'{train_name}_eda_log.xlsx')
    required_csv = [
        f'{train_name}_train_raw.csv',
        f'{train_name}_train_minmax.csv',
        f'{train_name}_train_standard.csv',
        f'{train_name}_test_raw.csv',
        f'{train_name}_test_minmax.csv',
        f'{train_name}_test_standard.csv',
    ]
    missing = []
    if not os.path.exists(eda_log):
        missing.append(os.path.basename(eda_log))
    for f in required_csv:
        if not os.path.exists(os.path.join(exp_dir, f)):
            missing.append(f)
    return missing


def main():
    train_name = sys.argv[1] if len(sys.argv) > 1 else 'sonar'

    base   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nb_in  = os.path.join(base, 'prj', '02_train.ipynb')
    nb_out = os.path.join(base, 'prj', f'02_train__{train_name}_output.ipynb')

    # Kiểm tra notebook tồn tại
    if not os.path.exists(nb_in):
        print(f'❌ Không tìm thấy: {nb_in}')
        sys.exit(1)

    # Kiểm tra EDA sớm – không cần đợi papermill
    missing = check_eda_ready(base, train_name)
    if missing:
        print('=' * 60)
        print(f'❌ TỪ CHỐI TRAIN – Bộ dữ liệu chưa được EDA!')
        print('=' * 60)
        print(f'   Train data : {train_name}')
        print(f'   Các file còn thiếu:')
        for f in missing:
            print(f'      [THIẾU] {f}')
        print()
        print(f'   → Hãy chạy EDA trước:')
        print(f'     python bin/run_eda.py {train_name}')
        print('=' * 60)
        sys.exit(1)

    print('=' * 60)
    print(f'  BƯỚC TRAIN – {train_name.upper()}')
    print('=' * 60)
    print(f'  Thời gian : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Input     : prj/02_train.ipynb')
    print(f'  Output    : prj/02_train__{train_name}_output.ipynb')
    print(f'  EDA check : ✅ Đã sẵn sàng')
    print('=' * 60)

    cmd = [
        sys.executable, '-m', 'papermill',
        nb_in, nb_out,
        '-p', 'TRAIN_DATA_NAME', train_name,
        '--cwd', base,
        '--log-output',
    ]
    result = subprocess.run(cmd)

    print()
    if result.returncode == 0:
        print('=' * 60)
        print(f'✅ TRAIN [{train_name}] HOÀN THÀNH!')
        print('=' * 60)
        print(f'  Model folder : model/{train_name}/')
        print(f'  Train log    : exps_/{train_name}_train_log.xlsx')
        print('=' * 60)
        print(f'➡ Bước tiếp theo: python bin/run_test.py {train_name} {train_name}')
    else:
        print('=' * 60)
        print(f'❌ LỖI khi chạy train [{train_name}]!')
        print(f'   Xem chi tiết tại: prj/02_train__{train_name}_output.ipynb')
        print('=' * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()