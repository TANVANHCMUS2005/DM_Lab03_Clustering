# Đồ án 3: Phân cụm và Ứng dụng - Khai thác Dữ liệu và Ứng dụng

**Bài báo:** Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis.

## Cấu trúc thư mục định hướng

* `data/`        : Thư mục chứa dữ liệu đầu vào.
* `docs/`        : Báo cáo định dạng PDF.
* `notebooks/`   : Các script/notebooks tái tạo thực nghiệm.
* `paper/`       : Chứa bản PDF của bài báo gốc.
* `src/`         : Mã nguồn Python thực hiện chính thuật toán và đo lường.

## Hướng dẫn cài đặt và chạy code

1. Cài đặt các thư viện cần thiết thông qua pip:
    ```bash
    pip install -r requirements.txt
    ```

2. Các bước thực thi chính sẽ được tiến hành ở thư mục `notebooks/` để hiển thị trực quan nhất tiến trình cũng như kết quả đồ thị.

## Dữ liệu Thực nghiệm (Dataset)
Nhóm hiện dùng dataset mô phỏng **Simple Combination Game** để tái hiện thực nghiệm chính của bài báo.

- File dữ liệu: `data/simple_combination_game.csv`
- File sinh dữ liệu: `src/synthetic_data.py`
- Mô tả: 20 phần tử, mỗi composition có 3 phần tử, 100,000 matches, xác suất thắng theo công thức trong Section 4.1.1 của paper.

Dataset này được sinh lại từ mô tả của bài báo, không phụ thuộc nguồn mã công khai bên ngoài.

## Kết quả phần Thành viên 2

- Notebook thực nghiệm chính: `notebooks/01_main_experiments.ipynb`
- Báo cáo phần TV2: `docs/member2_report.md`
- Hình pipeline: `docs/member2_pipeline.svg`
- Bảng kết quả: `docs/member2_summary.csv`, `docs/member2_aggregate.csv`, `docs/member2_comparison.csv`
- Hình kết quả huấn luyện: `docs/member2_training_and_accuracy.png`

## Phân công (Tham khảo `PhanCong.md`)
- Thành viên 1: Setup data, xử lý dữ liệu và tính toán tỷ lệ Counter (utils.py).
- Thành viên 2: Xây dựng giải thuật phân cụm chính (model.py) và metrics.py.
- Thành viên 3: Phân tích đánh giá, thiết kế Ablation Study.
- Thành viên 4: Chạy thử trên Dataset mới và hoàn thiện Report.
