# PHỤ LỤC: BẢNG PHÂN CÔNG CÔNG VIỆC NHÓM

**Tên bài báo:** Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis
**Môn học:** Khai thác dữ liệu và ứng dụng (CSC14004)
**Số lượng thành viên:** 4

## 1. Tóm tắt phân công (Ma trận công việc)

| Thành viên | Trọng tâm công việc (Lý thuyết & Báo cáo) | Trọng tâm thực nghiệm (Code & Data) | Mức độ hoàn thành |
| :--- | :--- | :--- | :---: |
| **[Tên TV 1]** | Giới thiệu, Bối cảnh & Mô hình toán học | Thiết lập Repo, Tiền xử lý dữ liệu gốc & Code Base | 100% |
| **[Tên TV 2]** | Kiến trúc thuật toán phân cụm | Code thuật toán từ đầu & Chạy thực nghiệm chính | 100% |
| **[Tên TV 3]** | Phân tích phê bình (Critical Thinking) | Thiết kế & Code Ablation Study, Phân tích sai lệch | 100% |
| **[Tên TV 4]** | Kết luận, Format Báo cáo & Tổng hợp PDF | Tìm kiếm dataset mới & Chạy thực nghiệm trên Data mới | 100% |

---

## 2. Chi tiết công việc từng thành viên

### Thành viên 1: [Nhập tên Thành viên 1] - *Data & Mathematical Setup*
* **Nhiệm vụ Code (Thực nghiệm):**
  * Khởi tạo cấu trúc thư mục chuẩn theo yêu cầu (`Group_ID/`, `src/`, `notebooks/`,...).
  * Tìm kiếm/sử dụng tập dữ liệu game PvP (MOBA/Hero Shooter) tương đương với bài báo gốc.
  * Code phần tiền xử lý dữ liệu (tính toán tỷ lệ thắng/thua, trích xuất đặc trưng đội hình).
  * Viết module tính toán mô hình "Counter Relationship" (Mối quan hệ khắc chế) theo công thức toán học của bài báo (`src/utils.py`).
* **Nhiệm vụ Báo cáo (Phần A & B):**
  * Viết **Chương 1 - Giới thiệu**: Trình bày bối cảnh, động lực, vấn đề cân bằng game PvP, đóng góp của bài báo (12% điểm).
  * Viết phần **Ký hiệu và bài toán hình thức** (trong Chương 2).

### Thành viên 2: [Nhập tên Thành viên 2] - *Core Algorithm Implementer*
* **Nhiệm vụ Code (Thực nghiệm):**
  * Tự cài đặt thuật toán Phân cụm (Clustering) được đề xuất trong bài báo từ đầu (from scratch), tuyệt đối không copy mã nguồn có sẵn (`src/model.py`).
  * Chạy tái hiện các thực nghiệm chính (Main experiments) dựa trên dữ liệu TV1 đã xử lý (`notebooks/01_main_experiments.ipynb`).
  * Cài đặt các độ đo đánh giá phân cụm (ví dụ: Silhouette Score, Davies-Bouldin Index... nếu không có nhãn thật) (`src/metrics.py`).
* **Nhiệm vụ Báo cáo (Phần A & B):**
  * Vẽ sơ đồ cấu trúc/pipeline của phương pháp (dùng TikZ hoặc draw.io).
  * Viết phần **Các thành phần kỹ thuật, Tính chất lý thuyết & Độ phức tạp** (trong Chương 2) (21% điểm).
  * Viết phần **Tái hiện kết quả chính** (trong Chương 4).

### Thành viên 3: [Nhập tên Thành viên 3] - *Analyst & Ablation Study*
* **Nhiệm vụ Code (Thực nghiệm):**
  * Đọc code của TV2 và đối chiếu với bài báo để **Phân tích sai lệch**.
  * Tự thiết kế và code ít nhất **01 Ablation Study** (Ví dụ: Thử bỏ đi một hàm trọng số trong việc tính counter relationship, hoặc thay đổi siêu tham số của thuật toán phân cụm) (`notebooks/02_ablation_study.ipynb`).
* **Nhiệm vụ Báo cáo (Phần A & B):**
  * Viết **Chương 3 - Phân tích phê bình**: Đánh giá khách quan 3 điểm mạnh, 2 điểm yếu, các giả định tiềm ẩn về mặt dữ liệu game, và đề xuất hướng mở rộng (12% điểm).
  * Viết phần **Phân tích sai lệch & Ablation study** (trong Chương 4) (15% điểm).

### Thành viên 4: [Nhập tên Thành viên 4] - *New Domain & Finalization*
* **Nhiệm vụ Code (Thực nghiệm):**
  * Áp dụng toàn bộ pipeline (của TV1 & TV2) lên **01 tập dữ liệu game PvP hoàn toàn mới** (Ví dụ: Bài báo làm Dota 2 thì nhóm kiếm data Liên Minh Huyền Thoại, hoặc Overwatch) (`notebooks/03_new_dataset.ipynb`) (5% + 15% điểm).
  * Viết file `README.md` hướng dẫn chạy code chi tiết.
  * Gom code, dọn dẹp Notebook (Restart & Run All).
* **Nhiệm vụ Báo cáo (Phần B & Format):**
  * Viết phần **Đánh giá trên tập dữ liệu mới** (trong Chương 4).
  * Viết **Tóm tắt (Abstract)** và **Chương 5 - Kết luận**.
  * Chịu trách nhiệm tổng hợp file PDF cuối cùng, format caption hình ảnh, bảng biểu, công thức toán học đúng chuẩn và đóng gói file zip nộp bài.

---

## 3. Quy định chung của nhóm
* **Liêm chính học thuật:** Code tự viết, nếu tham khảo AI (ChatGPT/Copilot) phải hiểu rõ từng dòng code để trả lời vấn đáp. Không đạo văn, không chép nguyên xi mã nguồn trên mạng.
* **Môi trường code:** Cài đặt cố định `random_state` ở tất cả các Notebook để kết quả có thể tái lập (Reproducible).
* **Ngôn ngữ:** Báo cáo được viết bằng [Tiếng Việt / Tiếng Anh]. Thống nhất giữ nguyên thuật ngữ chuyên ngành.