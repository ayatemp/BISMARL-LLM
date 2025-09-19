"""
PyTorch Lightning Module for training the section classifier
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from adapters import AutoAdapterModel
from torch.optim import Adam
from transformers import AutoTokenizer

from automatic_scientific_qm.section_classification.model import (
    SectionClassifierTransformer,
)


class SectionClassifier(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_classes = config["model"]["num_classes"]
        self.model = SectionClassifierTransformer(config["model"])
        self.lr = config["lr"]
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def training_step(self, data: dict, data_idx: int) -> torch.Tensor:
        predicted_labels = self.model(data)
        loss = self.loss(predicted_labels, data["labels"])
        acc = (predicted_labels.argmax(dim=1) == data["labels"]).float().mean()
        self.log("training/accuracy", acc)
        return loss

    def validation_step(self, data: dict, data_idx: int) -> torch.Tensor:
        predicted_labels = self.model(data)
        loss = self.loss(predicted_labels, data["labels"])
        self.compute_metrics(predicted_labels, data["labels"])
        return loss

    def test_step(self, data: dict, data_idx: int) -> torch.Tensor:
        predicted_labels = self.model(data)
        loss = self.loss(predicted_labels, data["labels"])
        self.compute_metrics(predicted_labels, data["labels"], "test")
        return loss

    def compute_metrics(
        self, predicted_labels: torch.Tensor, labels: torch.Tensor, split="validation"
    ) -> None:
        # Compute overall accuracy
        acc = (predicted_labels.argmax(dim=1) == labels).float().mean()
        self.log(f"{split}/accuracy", acc, batch_size=predicted_labels.shape[0])

        # Compute accuracy per class
        for cl in range(self.num_classes):
            mask = labels == cl
            if mask.sum() > 0:
                class_acc = (predicted_labels[mask].argmax(dim=1) == cl).float().mean()
                self.log(f"{split}/accuracy_{cl}", class_acc, batch_size=mask.sum())

    def predict(self, data: dict) -> torch.Tensor:
        predicted_labels = self.model(data)
        return predicted_labels

    def load_preprocessing_utils(self, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.embedding_model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_base"
        ).to(device)
        self.embedding_model.load_adapter(
            "allenai/specter2", source="hf", load_as="classification", set_active=True
        )
        self.embedding_model = self.embedding_model.to(device)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=self.lr)
