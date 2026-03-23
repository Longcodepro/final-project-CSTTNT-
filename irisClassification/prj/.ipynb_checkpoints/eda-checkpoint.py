import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data(filepath):
    """Đọc file csv, tự động thêm header nếu không có"""
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    try:
        df = pd.read_csv(filepath, header=0)
        # Kiểm tra nếu header là số thì file không có header
        if df.columns[0].replace('.','').isdigit():
            df = pd.read_csv(filepath, header=None, names=col_names)
    except:
        df = pd.read_csv(filepath, header=None, names=col_names)
    return df

def run_eda(filepath, output_excel):
    """Chạy EDA và ghi kết quả vào Excel"""
    print("=" * 50)
    print("BẮT ĐẦU PHÂN TÍCH DỮ LIỆU (EDA)")
    print("=" * 50)

    df = load_data(filepath)

    # ── 1. Thông tin cơ bản ──────────────────────────
    info = {
        'Thời gian chạy': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'File dữ liệu':   [filepath],
        'Số dòng':        [df.shape[0]],
        'Số cột':         [df.shape[1]],
        'Tên cột':        [', '.join(df.columns.tolist())],
    }
    df_info = pd.DataFrame(info)
    print(f"✔ Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")

    # ── 2. Missing values ────────────────────────────
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Cột', 'Số giá trị thiếu']
    missing['% thiếu'] = (missing['Số giá trị thiếu'] / len(df) * 100).round(2)
    print(f"✔ Missing values:\n{missing.to_string(index=False)}")

    # ── 3. Thống kê mô tả ────────────────────────────
    df_stats = df.describe().T.reset_index()
    df_stats.columns = ['Cột', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    df_stats = df_stats.round(4)
    print(f"✔ Thống kê mô tả:\n{df_stats.to_string(index=False)}")

    # ── 4. Phân phối nhãn ────────────────────────────
    label_col = df.columns[-1]
    df_class = df[label_col].value_counts().reset_index()
    df_class.columns = ['Lớp', 'Số lượng']
    df_class['Tỉ lệ %'] = (df_class['Số lượng'] / len(df) * 100).round(2)
    print(f"✔ Phân phối lớp:\n{df_class.to_string(index=False)}")

    # ── 5. Ghi Excel ─────────────────────────────────
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_info.to_excel(writer,   sheet_name='Thong_tin_chung', index=False)
        missing.to_excel(writer,   sheet_name='Missing_Values',  index=False)
        df_stats.to_excel(writer,  sheet_name='Thong_ke_mo_ta',  index=False)
        df_class.to_excel(writer,  sheet_name='Phan_phoi_lop',   index=False)

    print(f"\n✅ Đã ghi kết quả EDA vào: {output_excel}")
    print("=" * 50)
    return df
