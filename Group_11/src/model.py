import torch
import torch.nn as nn
import torch.nn.functional as F

class RatingEncoder(nn.Module):
    """
    Bộ mã hóa cho Bảng xếp hạng Neural (NRT).
    Kiến trúc: 4 lớp ẩn FC với LeakyReLU, output Exponential (Figure 16 trong bài báo).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Hàm kích hoạt mũ (Exponential) đảm bảo rating luôn dương 
        return torch.exp(self.net(x))

class NRT(nn.Module):
    """
    Bảng Xếp hạng Neural (Neural Rating Table) dự đoán tỷ lệ thắng bằng Bradley-Terry.
    Theo Section 3.1 và Figure 16 trong bài báo.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = RatingEncoder(input_dim)

    def forward(self, comp_A, comp_B):
        rating_A = self.encoder(comp_A)
        rating_B = self.encoder(comp_B)
        # Mô hình Bradley-Terry (Công thức 1)
        expected_win_A = rating_A / (rating_A + rating_B)
        return expected_win_A, rating_A, rating_B

class VQLayer(nn.Module):
    """
    Lớp Lượng tử hóa Vector (Vector Quantization).
    Triển khai theo Section 3.2.2, Equations (5), (10), (11), (12) trong bài báo.

    Trả về 3 loss riêng biệt tách theo gradient flow (Eq. 12):
    - loss_codebook: MSE(sg[ze], zq) → cập nhật codebook di chuyển về phía encoder output
    - loss_commit:   MSE(ze, sg[zq]) → cập nhật encoder di chuyển về phía codebook
    - loss_mean:     MSE(sg[ze], mean_ek) → cập nhật codebook chống sụp đổ từ điển mã (Eq.11)
    """
    def __init__(self, num_embeddings, embedding_dim, beta_N=0.01, beta_M=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Khởi tạo các siêu tham số theo thiết lập chuẩn của bài báo (Section 4.3)
        self.beta_N = beta_N 
        self.beta_M = beta_M 
        
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        """
        Args:
            z_e: Latent code liên tục từ encoder, shape (batch_size, embedding_dim).

        Returns:
            z_q_st:    Quantized code với straight-through gradient (Eq. 6).
            loss_codebook: Codebook loss → gradient chỉ đến codebook.
            loss_commit:   Commitment loss → gradient chỉ đến encoder.
            loss_mean:     VQ Mean loss → gradient chỉ đến codebook (Eq. 11).
            min_encoding_indices: Chỉ số cụm gần nhất, shape (batch_size, 1).
        """
        # Tính khoảng cách Euclid từ z_e đến các vector trong codebook
        distances = (torch.sum(z_e**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(z_e, self.codebook.weight.t()))
        
        # Lấy index của vector gần nhất (nearest neighbor) 
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.codebook(min_encoding_indices).squeeze(1)

        # --- Eq. (12): Gradient flow cho từng thành phần ---

        # Codebook loss (Lvq cho codebook): MSE(sg[ze], zq)
        # Gradient chỉ chảy đến codebook, đẩy codebook vectors về phía encoder output
        loss_codebook = F.mse_loss(z_e.detach(), z_q)

        # Commitment loss (Lvq cho encoder): MSE(ze, sg[zq])
        # Gradient chỉ chảy đến encoder, đẩy encoder output về phía codebook
        loss_commit = F.mse_loss(z_q.detach(), z_e)

        # VQ Mean Loss (Eq. 11): MSE(sg[ze], mean_ek)
        # Gradient chỉ chảy đến codebook, chống sụp đổ từ điển mã
        mean_codebook = torch.mean(self.codebook.weight, dim=0)
        loss_mean = F.mse_loss(z_e.detach(), mean_codebook.expand_as(z_e))

        # Stop-gradient / Straight-through estimator (Eq. 6)
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, loss_codebook, loss_commit, loss_mean, min_encoding_indices

class CounterDecoder(nn.Module):
    """
    Bộ giải mã cho phần dư khắc chế.
    Kiến trúc: 3 lớp ẩn FC với LeakyReLU, output Tanh (Figure 17 trong bài báo).
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Tanh() # Sử dụng Tanh để nén giá trị vào khoảng [-1, 1]
        )

    def forward(self, z_q_A, z_q_B):
        x = torch.cat([z_q_A, z_q_B], dim=1)
        return self.net(x)

class NCT(nn.Module):
    """
    Bảng Khắc chế Neural (Neural Counter Table) với Mạng Siamese.
    Triển khai theo Section 3.2.2, Figure 2 và Figure 17 trong bài báo.

    Returns:
        residual:      Phần dư khắc chế dự đoán (x1 - x2) / 2 (Figure 2).
        loss_codebook: Loss cập nhật codebook (Eq. 12: ∂Lvq/∂Ck).
        loss_commit:   Loss commitment cho encoder (Eq. 12: βN × ∂Lvq/∂Ce).
        loss_mean:     VQ Mean Loss cho codebook (Eq. 12: βM × ∂Lmean/∂Ck).
        idx_A, idx_B:  Nhãn cụm (cluster indices) của team A và B.
    """
    def __init__(self, input_dim, num_embeddings=9, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, embedding_dim)
        )
        self.vq = VQLayer(num_embeddings, embedding_dim)
        self.decoder = CounterDecoder(embedding_dim)

    def forward(self, comp_A, comp_B):
        # 1. Nhúng đội hình thành latent code liên tục (z_e) 
        z_e_A = self.encoder(comp_A)
        z_e_B = self.encoder(comp_B)

        # 2. Lượng tử hóa vector (VQ) 
        z_q_A, loss_cb_A, loss_cm_A, loss_mean_A, idx_A = self.vq(z_e_A)
        z_q_B, loss_cb_B, loss_cm_B, loss_mean_B, idx_B = self.vq(z_e_B)

        # 3. Tính toán phần dư đối xứng ngược (Figure 2)
        x1 = self.decoder(z_q_A, z_q_B)
        x2 = self.decoder(z_q_B, z_q_A)
        residual = (x1 - x2) / 2.0

        # Trung bình losses theo Eq. (10) và (11)
        loss_codebook = (loss_cb_A + loss_cb_B) / 2.0
        loss_commit = (loss_cm_A + loss_cm_B) / 2.0
        loss_mean = (loss_mean_A + loss_mean_B) / 2.0

        return residual, loss_codebook, loss_commit, loss_mean, idx_A, idx_B