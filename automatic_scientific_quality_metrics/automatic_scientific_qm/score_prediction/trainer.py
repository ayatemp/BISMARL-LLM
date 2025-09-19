"""
PyTorch Lightning Trainer for the score prediction models.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

from automatic_scientific_qm.score_prediction.model import ScoreModel


class ScorePrediction(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.context_type = config["data"]["context_type"]
        if config["data"]["context_type"] == "no_context":
            use_context = False
        else:
            use_context = True
        self.score_model = ScoreModel(config["trainer"]["model"], use_context)
        self.loss = nn.L1Loss()
        self.lr = config["trainer"]["lr"]
        self.optimizer = self.configure_optimizers()
        self.save_hyperparameters()

    def get_loss(self, data: dict) -> torch.Tensor:
        papers, masks, scores = data
        predicted_scores = self.score_model(papers, masks).squeeze(1)
        loss = self.loss(predicted_scores, scores)
        return loss

    def training_step(self, data: dict, data_idx: int) -> torch.Tensor:
        loss = self.get_loss(data)
        self.log("training/loss", loss)
        return loss

    def validation_step(self, data: dict, data_idx: int) -> None:
        loss = self.get_loss(data)
        self.log("validation/loss", loss)

    def predict(self, data: dict) -> torch.Tensor:
        papers, masks, _ = data
        predicted_scores = self.score_model(papers, masks).squeeze(1)
        return predicted_scores

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PairwiseComparison(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.context_type = config["data"]["context_type"]
        if config["data"]["context_type"] == "no_context":
            use_context = False
        else:
            use_context = True
        self.score_model = ScoreModel(config["trainer"]["model"], use_context)
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = config["trainer"]["lr"]
        self.optimizer = self.configure_optimizers()
        self.save_hyperparameters()

    def get_loss(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        papers1, masks1, papers2, masks2, scores = self.transform_data(data)
        predicted_scores1 = self.score_model(papers1, masks1).squeeze(1)
        predicted_scores2 = self.score_model(papers2, masks2).squeeze(1)
        score_diff = predicted_scores1 - predicted_scores2
        loss = self.loss(score_diff, scores)
        acc = ((score_diff > 0) == scores).float().mean()
        return loss, acc

    def training_step(self, data: dict, data_idx: int) -> torch.Tensor:
        loss, acc = self.get_loss(data)
        self.log("training/loss", loss)
        self.log("training/acc", acc)
        return loss

    def validation_step(self, data: dict, data_idx: int) -> torch.Tensor:
        loss, acc = self.get_loss(data)
        self.log("validation/loss", loss)
        self.log("validation/acc", acc)
        return loss

    def predict(self, data: dict) -> torch.Tensor:
        papers1, masks1, papers2, masks2, _ = self.transform_data(data)
        predicted_scores1 = self.score_model(papers1, masks1).squeeze(1)
        predicted_scores2 = self.score_model(papers2, masks2).squeeze(1)
        return predicted_scores1 >= predicted_scores2

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
