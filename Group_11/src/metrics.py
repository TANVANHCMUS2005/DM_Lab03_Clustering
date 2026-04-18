import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(latent_features: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Đánh giá chất lượng phân cụm trên không gian ẩn z_e sử dụng các độ đo không giám sát.
    
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