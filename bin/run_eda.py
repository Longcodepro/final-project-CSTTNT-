import os
import papermill as pm

# ── Tham số thực nghiệm ──────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTEBOOK_PATH = os.path.join(BASE_DIR, 'prj', '01_eda.ipynb')

# ── Chạy notebook ────────────────────────────────────
if __name__ == '__main__':
    print('=' * 50)
    print('BẮT ĐẦU CHẠY EDA')
    print(f'Notebook : {NOTEBOOK_PATH}')
    print(f'Thư mục  : {BASE_DIR}')
    print('=' * 50)

    try:
        pm.execute_notebook(
            NOTEBOOK_PATH,   # notebook đầu vào
            NOTEBOOK_PATH,   # ghi đè lại chính nó
            cwd=BASE_DIR     # chạy từ thư mục gốc
        )
        print('✅ Chạy EDA xong!')
        print('📊 Kết quả đã ghi vào exps_/eda_log.xlsx')
        print('📁 Các file CSV đã lưu vào exps_/')
    except Exception as e:
        print(f'❌ Có lỗi khi chạy EDA: {e}')
