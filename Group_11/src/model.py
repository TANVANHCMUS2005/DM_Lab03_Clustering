import torch
import torch.nn as nn
import torch.nn.functional as F

class RatingEncoder(nn.Module):
    """
    Bộ mã hóa cho Bảng xếp hạng Neural (NRT).
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
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = RatingEncoder(input_dim)

    def forward(self, comp_A, comp_B):
        rating_A = self.encoder(comp_A)
        rating_B = self.encoder(comp_B)
        # Mô hình Bradley-Terry 
        expected_win_A = rating_A / (rating_A + rating_B)
        return expected_win_A, rating_A, rating_B

class VQLayer(nn.Module):
    """
    Lớp Lượng tử hóa Vector (Vector Quantization) tích hợp VQ Mean Loss.
    """
    def __init__(self, num_embeddings, embedding_dim, beta_N=0.01, beta_M=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Khởi tạo các siêu tham số theo thiết lập chuẩn của bài báo 
        self.beta_N = beta_N 
        self.beta_M = beta_M 
        
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e shape: (batch_size, embedding_dim)
        # Tính khoảng cách Euclid từ z_e đến các vector trong codebook
        distances = (torch.sum(z_e**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(z_e, self.codebook.weight.t()))
        
        # Lấy index của vector gần nhất (nearest neighbor) 
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.codebook(min_encoding_indices).squeeze(1)

        # Suy hao VQ tiêu chuẩn (L_vq) 
        loss_vq = F.mse_loss(z_q.detach(), z_e)

        # Suy hao VQ Mean (L_mean) để chống sụp đổ từ điển mã 
        mean_codebook = torch.mean(self.codebook.weight, dim=0)
        loss_mean = F.mse_loss(z_e, mean_codebook.expand_as(z_e).detach())

        # Stop-gradient (gradient copying) 
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss_vq, loss_mean, min_encoding_indices

class CounterDecoder(nn.Module):
    """
    Bộ giải mã cho phần dư khắc chế.
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
        z_q_A, loss_vq_A, loss_mean_A, idx_A = self.vq(z_e_A)
        z_q_B, loss_vq_B, loss_mean_B, idx_B = self.vq(z_e_B)

        # 3. Tính toán phần dư đối xứng ngược (x1 - x2) / 2 
        x1 = self.decoder(z_q_A, z_q_B)
        x2 = self.decoder(z_q_B, z_q_A)
        residual = (x1 - x2) / 2.0

        loss_vq = (loss_vq_A + loss_vq_B) / 2.0
        loss_mean = (loss_mean_A + loss_mean_B) / 2.0

        return residual, loss_vq, loss_mean, idx_A, idx_B