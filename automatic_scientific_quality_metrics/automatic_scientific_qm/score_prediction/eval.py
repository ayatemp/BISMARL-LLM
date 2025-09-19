"""
Evaluation loop for score prediction and pairwise comparison models.
"""

import itertools
import random

from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
import tqdm
import torch
import wandb

from torch.utils.data import DataLoader


def eval(
    best_model: pl.LightningModule, test_dataloader: DataLoader, config: dict
) -> None:
    """
    Evaluate the performance of the best_model on the test dataset.

    Args:
        best_model (pl.LightningModule): The trained model to be evaluated.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """
    best_model.eval()
    if config["data"]["pairwise_comparison"]:
        eval_pairwise_comparison(best_model, test_dataloader, config)
    else:
        eval_regression(best_model, test_dataloader, config)


def eval_pairwise_comparison(
    best_model: pl.LightningModule, test_dataloader: DataLoader, config: dict
) -> None:
    """
    Evaluate pairwise comparisons between papers using a given model.

    Args:
        best_model (pl.LightningModule): The trained model used for prediction.
        test_dataloader (DataLoader): The dataloader containing the test dataset.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """

    test_dataset = test_dataloader.dataset
    n = len(test_dataset)
    paper_ids = list(range(n))

    # Subsample the paper-ids to reduce number of computations
    paper_ids = random.sample(paper_ids, min(len(paper_ids), 1000))

    n_comparisons = 0
    n_correct_comparisons = 0
    id2score = {}
    id2wins = defaultdict(int)
    pbar = tqdm.tqdm(total=len(paper_ids) * (len(paper_ids) - 1) // 2)
    for paper_id_a, paper_id_b in itertools.combinations(paper_ids, 2):
        score_a, paper_representation_a, reference_a, full_paper_a = test_dataset[
            paper_id_a
        ]
        score_b, paper_representation_b, reference_b, full_paper_b = test_dataset[
            paper_id_b
        ]
        scores = torch.tensor([score_a, score_b]).float().to(config["device"])
        paper_representations = [paper_representation_a, paper_representation_b]
        references = [reference_a, reference_b]
        full_papers = [full_paper_a, full_paper_b]

        data_sample = {
            "scores": scores,
            "paper_representations": paper_representations,
            "references": references,
            "full_papers": full_papers,
        }

        gt = 1 if score_a > score_b else 0
        with torch.no_grad():
            predicted_comparison = best_model.predict(data_sample)

        n_comparisons += 1
        if predicted_comparison == gt:
            n_correct_comparisons += 1

        if predicted_comparison == 1:
            id2wins[paper_id_a] += 1
        else:
            id2wins[paper_id_b] += 1

        id2score[paper_id_a] = score_a
        id2score[paper_id_b] = score_b
        pbar.update(1)

    # Log Results
    if n_comparisons > 0:
        wandb.log({"comparison_accuracy": n_correct_comparisons / n_comparisons})

    df = pd.DataFrame({"score": id2score, "wins": id2wins})
    spearman_corr = df.corr(method="spearman").iloc[0, 1]
    wandb.log({"spearman_rank_corr": spearman_corr})


def eval_regression(
    best_model: pl.LightningModule, test_dataloader: DataLoader, config: dict
) -> None:
    """
    Evaluate the regression performance of a given model on a test dataset.

    Args:
        best_model (Model): The trained regression model to evaluate.
        test_dataloader (DataLoader): The data loader for the test dataset.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """
    id2score = {}
    id2pred_score = defaultdict(float)
    abs_errors = []
    for k, data in tqdm.tqdm(enumerate(test_dataloader)):
        data = [d.to(config["device"]) for d in data]
        _, _, scores = data
        with torch.no_grad():
            predicted_score = best_model.predict(data)

        id2score[k] = scores
        id2pred_score[k] = predicted_score
        abs_errors.append(abs(scores - predicted_score))

    # Log Results
    if len(abs_errors) > 0:
        mean_absolute_error = torch.tensor(abs_errors).nanmean().item()
        wandb.log({"mean_absolute_error": mean_absolute_error})

    df = pd.DataFrame({"citation_score": id2score, "predicted_score": id2pred_score})
    spearman_corr = df.corr(method="spearman").iloc[0, 1]
    wandb.log({"spearman_rank_corr": spearman_corr})
