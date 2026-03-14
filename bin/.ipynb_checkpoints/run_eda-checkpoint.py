"""
run_eda.py – Chạy 01_eda.ipynb qua papermill
Sử dụng: python bin/run_eda.py [DATA_NAME]
Ví dụ  : python bin/run_eda.py sonar
"""
import subprocess, sys, os
from datetime import datetime


def main():
    data_name = sys.argv[1] if len(sys.argv) > 1 else 'sonar'

    base   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nb_in  = os.path.join(base, 'prj', '01_eda.ipynb')
    nb_out = os.path.join(base, 'prj', f'01_eda__{data_name}_output.ipynb')

    # Kiểm tra notebook tồn tại
    if not os.path.exists(nb_in):
        print(f'❌ Không tìm thấy: {nb_in}')
        sys.exit(1)

    # Kiểm tra file dữ liệu tồn tại
    data_file = os.path.join(base, 'data', f'{data_name}.csv')
    if not os.path.exists(data_file):
        print('=' * 60)
        print(f'❌ TỪ CHỐI – Không tìm thấy file dữ liệu!')
        print('=' * 60)
        print(f'   Tên data  : {data_name}')
        print(f'   Tìm tại   : {data_file}')
        print()
        print(f'   → Hãy đặt file {data_name}.csv vào thư mục data/')
        print('=' * 60)
        sys.exit(1)

    print('=' * 60)
    print(f'  BƯỚC EDA – {data_name.upper()}')
    print('=' * 60)
    print(f'  Thời gian : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Input     : prj/01_eda.ipynb')
    print(f'  Output    : prj/01_eda__{data_name}_output.ipynb')
    print(f'  Data      : data/{data_name}.csv')
    print('=' * 60)

    cmd = [
        sys.executable, '-m', 'papermill',
        nb_in, nb_out,
        '-p', 'DATA_NAME', data_name,
        '--cwd', base,
        '--log-output',
    ]
    result = subprocess.run(cmd)

    print()
    if result.returncode == 0:
        print('=' * 60)
        print(f'✅ EDA [{data_name}] HOÀN THÀNH!')
        print('=' * 60)
        print(f'  EDA log   : exps_/{data_name}_eda_log.xlsx')
        print(f'  Train CSV : exps_/{data_name}_train_*.csv')
        print(f'  Test  CSV : exps_/{data_name}_test_*.csv')
        print(f'  Scaler    : exps_/{data_name}_scaler_*.pkl')
        print('=' * 60)
        print(f'➡ Bước tiếp theo: python bin/run_train.py {data_name}')
    else:
        print('=' * 60)
        print(f'❌ LỖI khi chạy EDA [{data_name}]!')
        print(f'   Xem chi tiết tại: prj/01_eda__{data_name}_output.ipynb')
        print('=' * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()