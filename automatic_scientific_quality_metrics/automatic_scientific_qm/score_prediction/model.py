"""
Score Model
"""

import torch
import torch.nn as nn


class ScoreModel(nn.Module):
    def __init__(self, config: dict, use_context: bool) -> None:
        super().__init__()

        self.use_context = use_context
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=config["nheads"],
                dim_feedforward=1024,
                dropout=config["dropout"],
                activation="relu",
                batch_first=True,
            ),
            num_layers=config["num_layers"],
        )
        self.dropout = nn.Dropout(config["dropout"])
        self.mlp = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor for padding.

        Returns:
            torch.Tensor: Predicted score of the model.
        """
        if self.use_context:
            paper_embeddings = self.encoder(data, src_key_padding_mask=mask)
            # Select the paper embedding
            B = mask.shape[0]
            paper_embeddings = paper_embeddings[
                torch.arange(B), torch.zeros(B, dtype=torch.long)
            ]
        else:
            paper_embeddings = data
            paper_embeddings = self.dropout(paper_embeddings)

        # Map to target_score
        output = self.mlp(paper_embeddings)

        return output
