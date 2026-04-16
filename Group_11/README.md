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
Do giới hạn dung lượng của tệp tin, dữ liệu được nén tại thư mục `data/` trong kho lưu trữ này (dưới dạng file `.zip`).
Tuy nhiên, để phục vụ yêu cầu kiểm tra và dự phòng, toàn bộ Data (bao gồm file CSV và JSON gốc) đã được upload bổ sung qua Google Drive.
👉 **https://drive.google.com/file/d/1tT2zWW_E_LeHuCMADsTn649UYk0TqwAe/view?usp=sharing**

## Phân công (Tham khảo `PhanCong.md`)
- Thành viên 1: Setup data, xử lý dữ liệu và tính toán tỷ lệ Counter (utils.py).
- Thành viên 2: Xây dựng giải thuật phân cụm chính (model.py) và metrics.py.
- Thành viên 3: Phân tích đánh giá, thiết kế Ablation Study.
- Thành viên 4: Chạy thử trên Dataset mới và hoàn thiện Report.
