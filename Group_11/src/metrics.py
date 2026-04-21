import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(latent_features: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Đánh giá chất lượng phân cụm trên không gian ẩn z_e sử dụng các độ đo không giám sát.
    Theo yêu cầu §3.2.4 của đề bài.
    
    Args:
        latent_features (np.ndarray): Các vector đặc trưng ẩn z_e (đã chuyển về numpy).
        cluster_labels (np.ndarray): Nhãn phân cụm idx_A (từ 0 đến M-1).
        
    Returns:
        dict: Chứa 3 độ đo đánh giá.
    """
    # Các độ đo này yêu cầu ít nhất 2 cụm và không được là 1 cụm chứa toàn bộ dữ liệu
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2 or len(unique_labels) == len(latent_features):
        return {
            "Silhouette": -1.0, 
            "Davies-Bouldin": -1.0, 
            "Calinski-Harabasz": 0.0
        }
        
    sil_score = silhouette_score(latent_features, cluster_labels)
    db_score = davies_bouldin_score(latent_features, cluster_labels)
    ch_score = calinski_harabasz_score(latent_features, cluster_labels)
    
    return {
        "Silhouette": sil_score,        # Càng gần 1 càng tốt 
        "Davies-Bouldin": db_score,     # Càng nhỏ càng tốt 
        "Calinski-Harabasz": ch_score   # Càng lớn càng tốt 
    }

def classify_strength(win_values: np.ndarray) -> np.ndarray:
    """
    Phân loại strength relation từ win values theo Section A.4.1 của bài báo.
    
    Quy tắc:
    - win_value > 0.501 → 1  (stronger)
    - win_value < 0.499 → -1 (weaker)
    - 0.499 ≤ win_value ≤ 0.501 → 0 (same)
    
    Args:
        win_values (np.ndarray): Mảng win value trong khoảng [0, 1].
        
    Returns:
        np.ndarray: Mảng nhãn strength relation (-1, 0, 1).
    """
    return np.where(win_values > 0.501, 1,
                    np.where(win_values < 0.499, -1, 0))

def compute_strength_accuracy(pred_win_values: np.ndarray, true_win_values: np.ndarray) -> float:
    """
    Tính Strength Relation Classification Accuracy theo Section 4.2 và A.4.1 của bài báo.
    
    Ground truth được xác định bằng tabular PairWin (trung bình win value thực tế
    cho mỗi cặp matchup). Accuracy đo tỷ lệ dự đoán đúng nhãn stronger/same/weaker.
    
    Đây là metric chính trong Table 1 và Table 2 của bài báo.
    
    Args:
        pred_win_values (np.ndarray): Win value dự đoán bởi model (NRT hoặc NCT).
        true_win_values (np.ndarray): Win value thực tế từ kết quả trận đấu.
        
    Returns:
        float: Accuracy (%) của strength relation classification.
    """
    pred_labels = classify_strength(pred_win_values)
    true_labels = classify_strength(true_win_values)
    accuracy = np.mean(pred_labels == true_labels) * 100.0
    return accuracy

def codebook_utilization(cluster_labels: np.ndarray, num_embeddings: int) -> int:
    """
    Đo số lượng category thực sự được sử dụng trong codebook (Used M).
    Theo Section 4.3, Table 3, Table 4 của bài báo.
    
    Mục tiêu là Used M ≈ M (number of embeddings), thể hiện codebook không bị
    sụp đổ (collapse). VQ Mean Loss (Eq. 11) giúp tăng Used M.
    
    Args:
        cluster_labels (np.ndarray): Nhãn cụm (0 đến M-1).
        num_embeddings (int): Tổng số codebook vectors (M).
        
    Returns:
        int: Số lượng category đã được sử dụng (Used M ≤ M).
    """
    unique_used = len(np.unique(cluster_labels))
    return unique_used