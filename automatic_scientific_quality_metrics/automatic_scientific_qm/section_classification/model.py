"""
PyTorch Model for Acadamic Section Classification
"""

import torch.nn as nn
import torch


class SectionClassifierTransformer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.projection = nn.Linear(768, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            dropout=config["dropout"],
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )
        self.label_predictor = nn.Linear(512, config["num_classes"])

    def forward(self, data: dict) -> torch.Tensor:
        embeddings = self.projection(data["embeddings"])
        transformed_embeddings = self.transformer(
            embeddings, src_key_padding_mask=data["mask"]
        )
        return self.label_predictor(transformed_embeddings.mean(dim=1))
