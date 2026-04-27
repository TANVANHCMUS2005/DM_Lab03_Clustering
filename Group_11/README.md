# Đồ án 3: Phân cụm và Ứng dụng — Khai thác Dữ liệu và Ứng dụng (CSC14004)

<table>
<tr><td><b>Nhóm</b></td><td>11</td></tr>
<tr><td><b>Bài báo</b></td><td><i>Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis</i></td></tr>
<tr><td><b>Ngôn ngữ</b></td><td>Python 3.9+, PyTorch</td></tr>
<tr><td><b>Môi trường chạy</b></td><td>Google Colab (GPU T4) &amp; Local Jupyter Notebook</td></tr>
</table>

---

## 1. Tổng quan bài toán

Bài báo đề xuất phương pháp **phân cụm mối quan hệ khắc chế (Counter Relationship)** giữa các đội hình trong game PvP nhằm hỗ trợ phân tích cân bằng game hiệu quả. Ý tưởng cốt lõi gồm hai giai đoạn:

1. **Neural Rating Table (NRT)** — Mô hình Bradley-Terry phi tuyến sử dụng MLP 4 lớp ẩn với đầu ra exponential, ước lượng **sức mạnh nội tại** (rating) của mỗi đội hình. Xác suất thắng được tính theo công thức:

   ```
   P(A thắng B) = R(A) / (R(A) + R(B))
   ```

2. **Neural Counter Table (NCT)** — Mô hình dư (residual) kết hợp **Vector Quantization (VQ)** để:
   - Phân cụm các đội hình thành các nhóm tương đương về chiến thuật (composition categories).
   - Dự đoán phần **chênh lệch khắc chế** (counter residual) mà NRT không bắt được:
   ```
   W_final(A, B) = P_NRT(A, B) + W_residual(A, B)
   ```

**Pipeline tổng thể:**

```
Dữ liệu trận đấu → Multi-hot Encoding → Huấn luyện NRT (Bradley-Terry)
    → Tính Residual → Huấn luyện NCT (VQ + Decoder) → Phân cụm & Đánh giá
```

---

## 2. Cấu trúc thư mục

```
Group_11/
│
├── README.md                       ← Hướng dẫn tổng quan (file này)
├── requirements.txt                ← Danh sách thư viện Python
│
├── data/
│   └── simple_combination_game.csv ← Dataset mô phỏng (100K matches, tự sinh)
│
├── src/                            ← Toàn bộ mã nguồn Python
│   ├── utils.py                    ← [TV1] Tiền xử lý dữ liệu, tính Counter Relationship
│   ├── model.py                    ← [TV2] Kiến trúc NRT, NCT, VectorQuantizer, hàm huấn luyện
│   ├── metrics.py                  ← [TV2] Các độ đo đánh giá (Accuracy, Clustering metrics)
│   ├── synthetic_data.py           ← [TV1] Sinh dataset Simple Combination Game theo paper
│   └── run_main_experiments_colab.py ← [TV2] Script chạy thực nghiệm chính trên Colab (xem Mục 3)
│
├── notebooks/                      ← Jupyter Notebooks thực nghiệm
│   ├── 01_main_experiments.ipynb   ← Tái hiện thực nghiệm chính (xem Mục 3)
│   ├── 02_ablation_study.ipynb     ← [TV3] Ablation Study
│   └── 03_new_dataset.ipynb        ← [TV4] Đánh giá trên dataset mới
│
├── results/                        ← Kết quả thực nghiệm (CSV, JSON)
│   ├── member2_cv_fold_summary.csv ← Chi tiết kết quả từng fold
│   ├── member2_cv_aggregate.csv    ← Kết quả tổng hợp trung bình ± std
│   ├── member2_cv_comparison.csv   ← Bảng so sánh với giá trị trong paper
│   └── member2_cv_results.json     ← Toàn bộ kết quả dạng JSON (bao gồm config)
│
├── docs/
│   └── Report.pdf                  ← Báo cáo PDF cuối cùng
│
└── paper/
    └── paper.pdf                   ← Bản PDF bài báo gốc được tái hiện
```

---

## 3. Hướng dẫn chạy code chi tiết

### 3.1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Các thư viện chính:**

| Thư viện | Phiên bản yêu cầu | Vai trò |
|:---|:---|:---|
| `torch` | ≥ 2.2.0 | Framework deep learning (NRT, NCT, VQ) |
| `numpy` | ≥ 1.24.0 | Tính toán ma trận |
| `pandas` | ≥ 2.0.0 | Xử lý dữ liệu dạng bảng |
| `scikit-learn` | ≥ 1.3.0 | Đánh giá phân cụm (Silhouette, DBI, CHI) |
| `scipy` | ≥ 1.10.0 | Tính toán khoa học bổ trợ |
| `matplotlib` | ≥ 3.7.0 | Vẽ đồ thị |
| `seaborn` | ≥ 0.12.0 | Visualization nâng cao |
| `jupyter` | ≥ 1.0.0 | Chạy Notebook |

### 3.2. Chạy thực nghiệm chính (Notebook `01_main_experiments.ipynb`)

> **⚠️ LƯU Ý QUAN TRỌNG VỀ MÔI TRƯỜNG THỰC THI:**
>
> Notebook `01_main_experiments.ipynb` đã được nhóm **chạy trên Google Colab** (runtime GPU T4) thông qua script `src/run_main_experiments_colab.py` để tận dụng GPU tăng tốc quá trình huấn luyện mô hình deep learning.
>
> - **Lý do sử dụng Colab:** Quá trình huấn luyện NRT và NCT yêu cầu nhiều epoch (100 epoch mỗi mô hình), lặp qua nhiều cấu hình M ∈ {3, 9, 27, 81} và 5-fold cross-validation. Trên CPU local, thời gian chạy rất lâu; trên GPU Colab T4 giảm đáng kể.
> - **Script Colab:** File `src/run_main_experiments_colab.py` chứa **logic tương đương hoàn toàn** với notebook `01_main_experiments.ipynb`, được viết dưới dạng script `.py` thuần để chạy command-line trên Colab. Script này import và gọi đúng các module trong `src/model.py`, `src/metrics.py`, `src/synthetic_data.py`.
> - **Kết quả đã lưu:** Toàn bộ output (CSV, JSON, hình ảnh) đã được lưu trong thư mục `results/` và được sử dụng trong báo cáo.
>
> Nếu Thầy/Cô muốn **tái chạy** để kiểm tra, có thể thực hiện theo một trong hai cách dưới đây.

#### Cách 1: Chạy trên Google Colab (khuyến nghị — nhanh nhất)

1. Upload toàn bộ thư mục `Group_11/` lên Google Drive hoặc clone repo lên Colab.
2. Mở một Notebook mới trên Colab, chọn runtime **GPU**.
3. Chạy lệnh:

```python
# Nếu clone repo trực tiếp trên Colab:
!git clone https://github.com/TANVANHCMUS2005/DM_Lab03_Clustering.git /content/DM_Lab03_Clustering
!pip install -r /content/DM_Lab03_Clustering/Group_11/requirements.txt

!python /content/DM_Lab03_Clustering/Group_11/src/run_main_experiments_colab.py \
    --project-root /content/DM_Lab03_Clustering
```

```python
# Hoặc nếu upload lên Google Drive:
from google.colab import drive
drive.mount('/content/drive')

!python /content/drive/MyDrive/DM_Lab03_Clustering/Group_11/src/run_main_experiments_colab.py \
    --project-root /content/drive/MyDrive/DM_Lab03_Clustering
```

**Các tham số có thể điều chỉnh:**

| Tham số | Giá trị mặc định | Mô tả |
|:---|:---:|:---|
| `--seed` | 42 | Random seed để tái lập kết quả |
| `--num-matches` | 100,000 | Số trận đấu trong dataset |
| `--n-splits` | 5 | Số fold Cross-Validation |
| `--nrt-epochs` | 100 | Số epoch huấn luyện NRT |
| `--nct-epochs` | 100 | Số epoch huấn luyện NCT |
| `--learning-rate` | 2.5e-4 | Tốc độ học |
| `--beta-n` | 0.01 | Hệ số commit loss (VQ) |
| `--beta-m` | 0.25 | Hệ số mean loss (VQ) |
| `--m-values` | 3 9 27 81 | Các giá trị M (số cụm codebook) |
| `--force-cpu` | — | Buộc chạy trên CPU (không dùng GPU) |

#### Cách 2: Chạy local bằng Jupyter Notebook

```bash
cd Group_11
jupyter notebook notebooks/01_main_experiments.ipynb
```

> **Lưu ý:** Chạy local trên CPU sẽ chậm hơn đáng kể so với Colab GPU. Notebook này sử dụng train/test split 80/20 với 5 random seeds, trong khi script Colab sử dụng 5-fold CV — cả hai đều là phương pháp đánh giá hợp lệ, chỉ khác ở chiến lược chia dữ liệu.

### 3.3. Chạy Ablation Study

```bash
jupyter notebook notebooks/02_ablation_study.ipynb
```

Notebook này khảo sát ảnh hưởng của các siêu tham số lên chất lượng phân cụm (TV3).

### 3.4. Chạy thực nghiệm trên Dataset mới

```bash
jupyter notebook notebooks/03_new_dataset.ipynb
```

Áp dụng toàn bộ pipeline NRT → NCT lên tập dữ liệu game PvP hoàn toàn mới (TV4).

---

## 4. Dữ liệu thực nghiệm

### 4.1. Dataset chính: Simple Combination Game

| Thông tin | Chi tiết |
|:---|:---|
| **File dữ liệu** | `data/simple_combination_game.csv` |
| **Script sinh dữ liệu** | `src/synthetic_data.py` |
| **Tham chiếu paper** | Section 4.1.1 |
| **Số phần tử (elements)** | 20 (đánh số 1–20) |
| **Kích thước đội hình** | 3 phần tử / composition |
| **Tổng số compositions** | C(20, 3) = 1,140 |
| **Số trận đấu** | 100,000 matches (uniform sampling) |
| **Công thức xác suất thắng** | `P(A thắng) = s_A² / (s_A² + s_B²)`, với `s = Σ element_id` |
| **Kết quả trận đấu** | Bernoulli sampling từ xác suất thắng |

> **Tính minh bạch:** Dataset được nhóm **tự sinh hoàn toàn** từ mô tả toán học trong bài báo gốc. Không sử dụng bất kỳ mã nguồn công khai nào bên ngoài.

### 4.2. Dataset mới

Chi tiết về tập dữ liệu mới được trình bày trong notebook `03_new_dataset.ipynb` và trong báo cáo (Chương 4).

---

## 5. Mô tả chi tiết các module mã nguồn

### `src/model.py` — Kiến trúc mô hình & Huấn luyện

| Class / Function | Mô tả |
|:---|:---|
| `RatingEncoder` | MLP 4 lớp ẩn (128 units) + exp output → rating luôn dương |
| `NeuralRatingTable` | Bradley-Terry model: `P(A) = R(A) / (R(A) + R(B))` |
| `VectorQuantizer` | VQ với nearest-neighbor lookup, straight-through estimator, hỗ trợ codebook loss + commit loss + mean loss |
| `NeuralCounterTable` | Encoder → VQ → Decoder dự đoán `W_residual = (D(qA, qB) - D(qB, qA)) / 2` |
| `train_nrt` / `train_nrt_fullbatch` | Huấn luyện NRT với MSE loss (mini-batch hoặc full-batch) |
| `train_nct` / `train_nct_fullbatch` | Huấn luyện NCT với MSE + auxiliary VQ losses |
| `predict_nrt` / `predict_nct` | Inference batched (hỗ trợ dataset lớn) |

### `src/metrics.py` — Độ đo đánh giá

| Function | Mô tả |
|:---|:---|
| `strength_relation_accuracy` | Accuracy phân loại {weaker, same, stronger} — metric chính trong paper |
| `unsupervised_clustering_scores` | **Silhouette Score**, **Davies-Bouldin Index**, **Calinski-Harabasz Index** |
| `codebook_utilization` | Tỷ lệ codebook VQ thực sự được sử dụng |
| `pairwin_table` / `attach_pairwin_from_table` | Tính PairWin ground truth cho cross-validation |
| `relative_deviation` | Độ lệch tương đối so với giá trị trong paper |

### `src/utils.py` — Tiền xử lý dữ liệu

| Function | Mô tả |
|:---|:---|
| `load_data` | Đọc file CSV trận đấu |
| `preprocess_data` | Làm sạch, trích xuất & sắp xếp đội hình |
| `extract_team_features` | Multi-hot encoding cho đội hình |
| `calculate_win_loss_rates` | Thống kê tỷ lệ thắng từng tướng |
| `calculate_counter_relationship` | Tính `W_res = W_actual - P_BT(A,B)` (công thức 7 paper) |

### `src/synthetic_data.py` — Sinh dữ liệu mô phỏng

| Function | Mô tả |
|:---|:---|
| `enumerate_compositions` | Liệt kê tất cả C(20,3) = 1,140 đội hình |
| `win_probability` | `s_A² / (s_A² + s_B²)` — Section 4.1.1 |
| `sample_simple_combination_matches` | Sinh 100K trận với uniform sampling + Bernoulli outcome |
| `build_feature_matrix` | Chuyển compositions → ma trận multi-hot |

### `src/run_main_experiments_colab.py` — Script chạy trên Colab

Script này đóng gói toàn bộ logic của `01_main_experiments.ipynb` thành file `.py` chạy được trên command-line. Bao gồm:

- **Tự động detect project root** (hỗ trợ cả Colab, Google Drive, local)
- **5-Fold Cross-Validation** với data augmentation (hoán đổi A↔B)
- Huấn luyện NRT + NCT qua nhiều giá trị M ∈ {3, 9, 27, 81}
- Đánh giá Strength Relation Accuracy + Clustering metrics
- **Xuất kết quả**: CSV tổng hợp, JSON chi tiết, hình training curves
- So sánh với giá trị paper (tính relative deviation)

---

## 6. Tổng quan kết quả

### Phương pháp đánh giá

| Phương pháp | Ý nghĩa |
|:---|:---|
| **5-Fold Cross-Validation** | Chia dữ liệu thành 5 phần, lần lượt dùng 1 phần test và 4 phần train, lấy trung bình |
| **Data Augmentation** | Với mỗi cặp (A, B, label), thêm (B, A, 1−label) để đảm bảo tính đối xứng |
| **Strength Relation Accuracy** | Metric chính: phân loại đúng {weaker, same, stronger} so với ground truth |

### Mô hình so sánh

| Mô hình | Mô tả |
|:---|:---|
| **NRT** (Baseline) | Chỉ dùng Bradley-Terry phi tuyến |
| **NRT + NCT (M=3,9,27,81)** | Kết hợp NRT với NCT có codebook size M |

Chi tiết kết quả nằm trong:
- **File CSV:** `results/member2_cv_aggregate.csv`, `results/member2_cv_comparison.csv`
- **File JSON:** `results/member2_cv_results.json` (bao gồm cấu hình, kết quả từng fold)
- **Báo cáo:** `docs/Report.pdf` (Chương 4)

---


