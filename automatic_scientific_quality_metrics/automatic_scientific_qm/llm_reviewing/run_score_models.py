"""
Run the score models on the subset of OpenReview data selected for the LLM reviewing task
"""
import argparse

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from automatic_scientific_qm.llm_reviewing.swiss_tournament import tournament_ranking
from automatic_scientific_qm.score_prediction.trainer import (
    ScorePrediction,
    PairwiseComparison,
)
from automatic_scientific_qm.score_prediction.data import transform_data_factory


def evaluate_score_model(config: dict) -> None:
    """
    Evaluate the score model based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Raises:
        NotImplementedError: If the dataset specified in the configuration is not supported.
        ValueError: If the evaluation algorithm specified in the configuration is not supported.
    """

    # Load dataset
    if config["dataset"] == "iclr":
        samples = torch.load("./data/iclr_200_subset_data.pt")
    elif config["dataset"] == "neurips":
        samples = torch.load("./data/neurips_200_subset_data.pt")
    else:
        raise NotImplementedError(
            f"Only iclr and neurips as datasets are supported and not {config['dataset']}"
        )

    # Choose evaluation algorithm
    if config["evaluation_algorithm"] == "swiss_tournament":
        run_swiss_tournament(samples, config)
    elif config["evaluation_algorithm"] == "direct_prediction":
        run_direct_prediction(samples, config)
    else:
        raise ValueError(
            f"Evaluation algorithm must be either swiss_tournament or direct_prediction and not {config['evaluation_algorithm']}."
        )


def run_swiss_tournament(samples: list, config) -> None:
    """
    Runs a Swiss tournament using pairwise comparison models.

    Args:
        samples (list): A list of samples.
        config: The configuration for the tournament.

    Returns:
        None
    """
    rsp_model = PairwiseComparison.load_from_checkpoint(
        config["model_checkpoint"], map_location=config["device"]
    )

    paperhash2sample = {}
    papers = []
    for sample in samples:
        paperhash = sample["paperhash"]
        paperhash2sample[paperhash] = sample
        papers.append(sample)

    tournament_ranking(
        papers,
        rsp_model,
        config["seed"],
        config["dataset"],
        paperhash2sample,
        max_round=5,
        config=config,
    )


def run_direct_prediction(samples: list, config: dict) -> None:
    """
    Runs direct prediction on a list of papers using a score prediction model.

    Args:
        samples (list): A list of papers to predict scores for.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        None
    """
    # Get model
    rsp_model = ScorePrediction.load_from_checkpoint(
        config["model_checkpoint"], map_location=config["device"]
    )

    # Get data transformation
    data_config = {
        "data": {"pairwise_comparison": False, "context_type": rsp_model.context_type}
    }
    transform_data = transform_data_factory(data_config)

    abs_errors = []
    true_scores = []
    pred_scores = []

    paperhash2sample = {}
    for sample in samples:
        paperhash = sample["paperhash"]
        paperhash2sample[paperhash] = sample

    paper_representation = f"{config['paper_representation']}_emb"
    for paperhash, sample in paperhash2sample.items():
        true_score = torch.tensor(sample["mean_score"]).float()
        paper = torch.tensor(sample[paper_representation], dtype=torch.float)
        full_paper = np.stack(
            [
                sample["intro_emb"],
                sample["method_emb"],
                sample["result_experiment_emb"],
                sample["background_emb"],
                sample["conclusion_emb"],
            ],
            axis=0,
        )
        full_paper = torch.tensor(full_paper)
        references = torch.tensor(sample["reference_emb"])

        data_sample = transform_data([paper], [references], [full_paper], true_score)
        data_sample = [t.to(config["device"]) for t in data_sample]
        predicted_score = rsp_model.predict(data_sample).item()

        true_scores.append(true_score)
        pred_scores.append(predicted_score)
        abs_errors.append(abs(true_score - predicted_score))

    wandb.log({"mean_absolute_error": np.mean(abs_errors)})

    df = pd.DataFrame({"true_scores": true_scores, "pred_scores": pred_scores})
    wandb.log({"pearson_correlation": df.corr().iloc[0, 1]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, choices=["iclr", "neurips"], default=None
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if args.dataset:
        config["dataset"] = args.dataset
    config["checkpoint"] = args.checkpoint

    wandb.init(**config["logging"])

    evaluate_score_model(config)
