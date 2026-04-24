"""Core models and training utilities for member 2's reproduction work.

This module implements the two main learning components described in the paper:

1. Neural Rating Table (NRT): a non-linear Bradley-Terry approximation.
2. Neural Counter Table (NCT): a vector-quantized residual model that captures
   counter relationships between team compositions.

The code is designed to be reusable from notebooks and simple scripts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Create a simple MLP with LeakyReLU activations."""

    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class RatingEncoder(nn.Module):
    """Encoder used by the Neural Rating Table.

    The paper uses a four-hidden-layer MLP with exponential output so that the
    resulting rating is always positive for the Bradley-Terry formulation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = _make_mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
            output_dim=1,
        )

    def forward(self, composition: torch.Tensor) -> torch.Tensor:
        """Return a positive scalar rating for each composition."""

        return torch.exp(self.network(composition))


class NeuralRatingTable(nn.Module):
    """Non-linear Bradley-Terry model for composition strength."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = RatingEncoder(input_dim=input_dim, hidden_dim=hidden_dim)

    def encode(self, compositions: torch.Tensor) -> torch.Tensor:
        """Encode compositions into positive ratings."""

        return self.encoder(compositions)

    def forward(
        self,
        comp_a: torch.Tensor,
        comp_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict the Bradley-Terry win value for A against B."""

        rating_a = self.encode(comp_a)
        rating_b = self.encode(comp_b)
        win_prob_a = rating_a / (rating_a + rating_b + 1e-12)
        return {
            "win_prob_a": win_prob_a,
            "rating_a": rating_a,
            "rating_b": rating_b,
        }


class VectorQuantizer(nn.Module):
    """Deterministic vector quantization with VQ mean loss support."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantize latent vectors with nearest-neighbor lookup."""

        distances = (
            latent.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2.0 * latent @ self.codebook.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices)

        # Straight-through estimator.
        quantized_st = latent + (quantized - latent).detach()

        mean_code = self.codebook.weight.mean(dim=0, keepdim=True).expand_as(latent)
        codebook_loss = F.mse_loss(latent.detach(), quantized)
        commit_loss = F.mse_loss(latent, quantized.detach())
        mean_loss = F.mse_loss(latent.detach(), mean_code)

        return {
            "quantized": quantized_st,
            "indices": indices,
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "mean_loss": mean_loss,
            "distances": distances,
        }


class CounterResidualDecoder(nn.Module):
    """Residual predictor used by the Neural Counter Table."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = _make_mlp(
            input_dim=embedding_dim * 2,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=1,
            final_activation=nn.Tanh(),
        )

    def forward(self, quantized_a: torch.Tensor, quantized_b: torch.Tensor) -> torch.Tensor:
        """Estimate the residual win value from quantized composition codes."""

        return self.network(torch.cat([quantized_a, quantized_b], dim=1))


class NeuralCounterTable(nn.Module):
    """Vector-quantized residual model for counter relationships."""

    def __init__(
        self,
        input_dim: int,
        num_embeddings: int = 9,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.category_encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=embedding_dim,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.decoder = CounterResidualDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

    def encode_latent(self, compositions: torch.Tensor) -> torch.Tensor:
        """Return continuous latent codes before quantization."""

        return self.category_encoder(compositions)

    def forward(
        self,
        comp_a: torch.Tensor,
        comp_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict the residual win value W_res(A, B)."""

        latent_a = self.encode_latent(comp_a)
        latent_b = self.encode_latent(comp_b)

        quantized_a = self.quantizer(latent_a)
        quantized_b = self.quantizer(latent_b)

        x_ab = self.decoder(quantized_a["quantized"], quantized_b["quantized"])
        x_ba = self.decoder(quantized_b["quantized"], quantized_a["quantized"])
        residual = (x_ab - x_ba) / 2.0

        return {
            "residual": residual,
            "latent_a": latent_a,
            "latent_b": latent_b,
            "category_a": quantized_a["indices"],
            "category_b": quantized_b["indices"],
            "codebook_loss": (quantized_a["codebook_loss"] + quantized_b["codebook_loss"]) / 2.0,
            "commit_loss": (quantized_a["commit_loss"] + quantized_b["commit_loss"]) / 2.0,
            "mean_loss": (quantized_a["mean_loss"] + quantized_b["mean_loss"]) / 2.0,
        }


@dataclass
class EpochHistory:
    """Training curves for a model."""

    total_loss: List[float]
    mse_loss: List[float]
    aux_loss: List[float]
    learning_rate: List[float]


class MatchTensorDataset(Dataset):
    """Simple dataset wrapper for pairwise composition matches."""

    def __init__(self, comp_a: np.ndarray, comp_b: np.ndarray, label: np.ndarray) -> None:
        self.comp_a = torch.tensor(comp_a, dtype=torch.float32)
        self.comp_b = torch.tensor(comp_b, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "comp_a": self.comp_a[index],
            "comp_b": self.comp_b[index],
            "label": self.label[index],
        }


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a dictionary of tensors to a target device."""

    return {key: value.to(device) for key, value in batch.items()}


def train_nrt(
    model: NeuralRatingTable,
    dataloader,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> EpochHistory:
    """Train the Neural Rating Table with an MSE objective."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1.0 - (epoch / max(epochs, 1))),
    )

    history = EpochHistory(total_loss=[], mse_loss=[], aux_loss=[], learning_rate=[])
    model.to(device)
    model.train()

    for _ in range(epochs):
        epoch_total = 0.0
        epoch_mse = 0.0
        sample_count = 0

        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch["comp_a"], batch["comp_b"])
            mse_loss = F.mse_loss(outputs["win_prob_a"].squeeze(1), batch["label"])
            mse_loss.backward()
            optimizer.step()

            batch_size = batch["label"].shape[0]
            sample_count += batch_size
            epoch_total += mse_loss.item() * batch_size
            epoch_mse += mse_loss.item() * batch_size

        scheduler.step()
        history.total_loss.append(epoch_total / max(sample_count, 1))
        history.mse_loss.append(epoch_mse / max(sample_count, 1))
        history.aux_loss.append(0.0)
        history.learning_rate.append(optimizer.param_groups[0]["lr"])

    return history


def train_nrt_fullbatch(
    model: NeuralRatingTable,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    label: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> EpochHistory:
    """Train NRT using full-batch optimization for faster synthetic experiments."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1.0 - (epoch / max(epochs, 1))),
    )

    batch_a = torch.tensor(comp_a, dtype=torch.float32, device=device)
    batch_b = torch.tensor(comp_b, dtype=torch.float32, device=device)
    batch_y = torch.tensor(label, dtype=torch.float32, device=device)

    history = EpochHistory(total_loss=[], mse_loss=[], aux_loss=[], learning_rate=[])
    model.to(device)
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(batch_a, batch_b)
        mse_loss = F.mse_loss(outputs["win_prob_a"].squeeze(1), batch_y)
        mse_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(mse_loss.item())
        history.total_loss.append(loss_value)
        history.mse_loss.append(loss_value)
        history.aux_loss.append(0.0)
        history.learning_rate.append(optimizer.param_groups[0]["lr"])

    return history


def train_nct(
    model: NeuralCounterTable,
    frozen_nrt: NeuralRatingTable,
    dataloader,
    *,
    epochs: int,
    learning_rate: float,
    beta_n: float,
    beta_m: float,
    device: torch.device,
) -> EpochHistory:
    """Train the Neural Counter Table on residual win values."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1.0 - (epoch / max(epochs, 1))),
    )

    frozen_nrt.to(device)
    frozen_nrt.eval()
    for parameter in frozen_nrt.parameters():
        parameter.requires_grad_(False)

    history = EpochHistory(total_loss=[], mse_loss=[], aux_loss=[], learning_rate=[])
    model.to(device)
    model.train()

    for _ in range(epochs):
        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_aux = 0.0
        sample_count = 0

        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()

            with torch.no_grad():
                nrt_outputs = frozen_nrt(batch["comp_a"], batch["comp_b"])
                bt_prediction = nrt_outputs["win_prob_a"].squeeze(1)

            residual_target = batch["label"] - bt_prediction
            outputs = model(batch["comp_a"], batch["comp_b"])
            residual_prediction = outputs["residual"].squeeze(1)

            mse_loss = F.mse_loss(residual_prediction, residual_target)
            aux_loss = (
                outputs["codebook_loss"]
                + beta_n * outputs["commit_loss"]
                + beta_m * outputs["mean_loss"]
            )
            total_loss = mse_loss + aux_loss
            total_loss.backward()
            optimizer.step()

            batch_size = batch["label"].shape[0]
            sample_count += batch_size
            epoch_total += total_loss.item() * batch_size
            epoch_mse += mse_loss.item() * batch_size
            epoch_aux += aux_loss.item() * batch_size

        scheduler.step()
        history.total_loss.append(epoch_total / max(sample_count, 1))
        history.mse_loss.append(epoch_mse / max(sample_count, 1))
        history.aux_loss.append(epoch_aux / max(sample_count, 1))
        history.learning_rate.append(optimizer.param_groups[0]["lr"])

    return history


def train_nct_fullbatch(
    model: NeuralCounterTable,
    frozen_nrt: NeuralRatingTable,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    label: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    beta_n: float,
    beta_m: float,
    device: torch.device,
) -> EpochHistory:
    """Train NCT using full-batch optimization for faster synthetic experiments."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1.0 - (epoch / max(epochs, 1))),
    )

    batch_a = torch.tensor(comp_a, dtype=torch.float32, device=device)
    batch_b = torch.tensor(comp_b, dtype=torch.float32, device=device)
    batch_y = torch.tensor(label, dtype=torch.float32, device=device)

    frozen_nrt.to(device)
    frozen_nrt.eval()
    for parameter in frozen_nrt.parameters():
        parameter.requires_grad_(False)

    history = EpochHistory(total_loss=[], mse_loss=[], aux_loss=[], learning_rate=[])
    model.to(device)
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            nrt_outputs = frozen_nrt(batch_a, batch_b)
            bt_prediction = nrt_outputs["win_prob_a"].squeeze(1)

        residual_target = batch_y - bt_prediction
        outputs = model(batch_a, batch_b)
        residual_prediction = outputs["residual"].squeeze(1)

        mse_loss = F.mse_loss(residual_prediction, residual_target)
        aux_loss = (
            outputs["codebook_loss"]
            + beta_n * outputs["commit_loss"]
            + beta_m * outputs["mean_loss"]
        )
        total_loss = mse_loss + aux_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        history.total_loss.append(float(total_loss.item()))
        history.mse_loss.append(float(mse_loss.item()))
        history.aux_loss.append(float(aux_loss.item()))
        history.learning_rate.append(optimizer.param_groups[0]["lr"])

    return history


@torch.no_grad()
def predict_nrt(
    model: NeuralRatingTable,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run batched NRT inference and return win probabilities."""

    model.to(device)
    model.eval()
    predictions: List[np.ndarray] = []

    for start in range(0, len(comp_a), batch_size):
        end = start + batch_size
        batch_a = torch.tensor(comp_a[start:end], dtype=torch.float32, device=device)
        batch_b = torch.tensor(comp_b[start:end], dtype=torch.float32, device=device)
        outputs = model(batch_a, batch_b)
        predictions.append(outputs["win_prob_a"].squeeze(1).cpu().numpy())

    return np.concatenate(predictions, axis=0)


@torch.no_grad()
def predict_nct(
    nrt_model: NeuralRatingTable,
    nct_model: NeuralCounterTable,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """Run batched NCT inference and return full predictions and categories."""

    nrt_model.to(device)
    nct_model.to(device)
    nrt_model.eval()
    nct_model.eval()

    win_values: List[np.ndarray] = []
    residuals: List[np.ndarray] = []
    categories_a: List[np.ndarray] = []
    categories_b: List[np.ndarray] = []
    latents_a: List[np.ndarray] = []
    latents_b: List[np.ndarray] = []

    for start in range(0, len(comp_a), batch_size):
        end = start + batch_size
        batch_a = torch.tensor(comp_a[start:end], dtype=torch.float32, device=device)
        batch_b = torch.tensor(comp_b[start:end], dtype=torch.float32, device=device)

        nrt_outputs = nrt_model(batch_a, batch_b)
        nct_outputs = nct_model(batch_a, batch_b)
        final_win = (
            nrt_outputs["win_prob_a"].squeeze(1)
            + nct_outputs["residual"].squeeze(1)
        ).clamp(0.0, 1.0)

        win_values.append(final_win.cpu().numpy())
        residuals.append(nct_outputs["residual"].squeeze(1).cpu().numpy())
        categories_a.append(nct_outputs["category_a"].cpu().numpy())
        categories_b.append(nct_outputs["category_b"].cpu().numpy())
        latents_a.append(nct_outputs["latent_a"].cpu().numpy())
        latents_b.append(nct_outputs["latent_b"].cpu().numpy())

    return {
        "win_value": np.concatenate(win_values, axis=0),
        "residual": np.concatenate(residuals, axis=0),
        "category_a": np.concatenate(categories_a, axis=0),
        "category_b": np.concatenate(categories_b, axis=0),
        "latent_a": np.concatenate(latents_a, axis=0),
        "latent_b": np.concatenate(latents_b, axis=0),
    }
