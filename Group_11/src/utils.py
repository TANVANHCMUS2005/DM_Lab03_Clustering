"""
utils.py

Module chứa các hàm tiện ích, thực hiện tiền xử lý dữ liệu và tính toán
mối quan hệ khắc chế (Counter Relationship) cho bài toán phân cụm đội hình.
Phụ trách chính: Thành viên 1.
"""

import pandas as pd
import numpy as np
import json

def load_data(filepath: str) -> pd.DataFrame:
    """
    Đọc dữ liệu trận đấu (match data) từ file CSV.
    
    Args:
        filepath (str): Đường dẫn đến file dữ liệu (ví dụ: 'data/games.csv').
        
    Returns:
        pd.DataFrame: Dữ liệu thô đã load.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Đã load thành công {len(df)} trận đấu từ {filepath}")
        return df
    except Exception as e:
        print(f"Lỗi khi load dữ liệu: {e}")
        return pd.DataFrame()

def load_champion_info(filepath: str) -> dict:
    """
    Đọc thông tin ánh xạ ID tướng sang tên tướng từ file JSON.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    champ_dict = {}
    for champ_id, info in data['data'].items():
        champ_dict[int(champ_id)] = info['name']
    return champ_dict

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý dữ liệu: làm sạch, loại bỏ dữ liệu thiếu, trích xuất hai đội hình.
    Trong dữ liệu LOL này, Team 1 (winner=1) và Team 2 (winner=2).
    
    Args:
        df (pd.DataFrame): Dataframe nguyên gốc từ games.csv.
        
    Returns:
        pd.DataFrame: Dataframe đã qua tiền xử lý gồm [team_1_champs, team_2_champs, winner].
    """
    # Lấy các cột t1_champ và t2_champ
    t1_cols = [f't1_champ{i}id' for i in range(1, 6)]
    t2_cols = [f't2_champ{i}id' for i in range(1, 6)]
    
    processed_df = pd.DataFrame()
    # Chuyển ID các tướng của mỗi team thành một list (hoặc tuple đã sort để dễ so sánh)
    processed_df['team_1'] = df[t1_cols].values.tolist()
    processed_df['team_2'] = df[t2_cols].values.tolist()
    
    # Sắp xếp ID trong đội hình để coi {1,2,3,4,5} cũng giống {5,4,3,2,1}
    processed_df['team_1'] = processed_df['team_1'].apply(lambda x: tuple(sorted(x)))
    processed_df['team_2'] = processed_df['team_2'].apply(lambda x: tuple(sorted(x)))
    
    processed_df['winner'] = df['winner']
    
    return processed_df

def calculate_win_loss_rates(df: pd.DataFrame) -> dict:
    """
    Tính toán tỷ lệ thắng/thua của từng tướng dựa trên tổng hợp lịch sử trận đấu.
    
    Args:
        df (pd.DataFrame): Dữ liệu các trận đấu đã tiền xử lý.
        
    Returns:
        dict: Tỷ lệ thắng tương ứng của từng hero/champion id.
    """
    champ_stats = {}
    
    for _, row in df.iterrows():
        t1 = row['team_1']
        t2 = row['team_2']
        winner = row['winner']
        
        # Duyệt qua tướng team 1
        for champ in t1:
            if champ not in champ_stats:
                champ_stats[champ] = {'matches': 0, 'wins': 0}
            champ_stats[champ]['matches'] += 1
            if winner == 1:
                champ_stats[champ]['wins'] += 1
                
        # Duyệt qua tướng team 2
        for champ in t2:
            if champ not in champ_stats:
                champ_stats[champ] = {'matches': 0, 'wins': 0}
            champ_stats[champ]['matches'] += 1
            if winner == 2:
                champ_stats[champ]['wins'] += 1
                
    # Tính win rate
    win_rates = {}
    for champ, stats in champ_stats.items():
        win_rates[champ] = stats['wins'] / stats['matches']
        
    return win_rates

def extract_team_features(df: pd.DataFrame, num_champions: int = 200) -> np.ndarray:
    """
    Trích xuất đặc trưng đội hình từ các trận đấu.
    Sử dụng kỹ thuật Multi-hot Encoding để vector hoá đội hình (đánh dấu 1 ở ID tướng có mặt).
    
    Args:
        df (pd.DataFrame): Dữ liệu chi tiết trận đấu với thành phần đội hình.
        num_champions (int): Số lượng tướng tối đa (LOL có khoảng ~150 tướng, lấy 200 cho an toàn về mặt không gian ID).
        
    Returns:
        np.ndarray: Ma trận (N_samples, 2, num_champions) lưu features của team 1 và team 2.
    """
    n_samples = len(df)
    features = np.zeros((n_samples, 2, num_champions), dtype=np.float32)
    
    for idx, row in enumerate(df.itertuples()):
        t1 = row.team_1
        t2 = row.team_2
        
        for champ_id in t1:
            if champ_id < num_champions:
                features[idx, 0, champ_id] = 1.0
                
        for champ_id in t2:
            if champ_id < num_champions:
                features[idx, 1, champ_id] = 1.0
                
    return features

def bradley_terry_prob(rating_a: float, rating_b: float) -> float:
    """
    Công thức số (1) trong bài báo: Mô hình Bradley-Terry.
    Tính xác suất đội A thắng đội B dựa trên sức mạnh (rating).
    """
    return rating_a / (rating_a + rating_b)

def calculate_counter_relationship(actual_win_val: float, rating_a: float, rating_b: float) -> float:
    """
    Tính toán đại lượng dư (Residual Win Value) - Mô phỏng Mối quan hệ khắc chế.
    Theo công thức (7) của bài báo: W_res = W_m - R(A)/(R(A)+R(B)).
    Phần này đại diện cho sự khắc chế sau khi trừ đi tỷ lệ thắng kỳ vọng bằng sức mạnh thuần túy.
    
    Args:
        actual_win_val (float): Kết quả thực tế (1 = A thắng, 0 = A thua, 0.5 = Hòa).
        rating_a (float): Sức mạnh tính toán được của đội A.
        rating_b (float): Sức mạnh tính toán được của đội B.
        
    Returns:
        float: Giá trị W_res thể hiện đội A khắc chế đội B ra sao (trị số âm hoặc dương).
    """
    expected_win = bradley_terry_prob(rating_a, rating_b)
    residual_win_value = actual_win_val - expected_win
    return residual_win_value
