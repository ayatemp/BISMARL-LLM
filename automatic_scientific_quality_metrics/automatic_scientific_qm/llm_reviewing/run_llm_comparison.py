"""
Code to run a swiss tournament between OpenReview papers using LLM as the underlying comparison model
"""

import argparse

import torch
import yaml

from datasets import load_from_disk

from automatic_scientific_qm.llm_reviewing.swiss_tournament import tournament_ranking

"""
This code includes portions from https://github.com/NoviScl/AI-Researcher by Chenglei Si,
licensed under the MIT License (see below or at https://github.com/NoviScl/AI-Researcher?tab=MIT-1-ov-file#readme).
Copyright (c) 2024 Chenglei Si
"""


def run_llm_ranking(config: dict) -> None:
    """
    Runs an X round swiss tournamnet as a ranking process based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """

    # Load papers
    if config["dataset"] == "iclr":
        samples = torch.load("./data/iclr_200_subset_data.pt")
        dataset = load_from_disk("../../data/processed/openreview-iclr")
    elif config["dataset"] == "neurips":
        samples = torch.load("./data/neurips_200_subset_data.pt")
        dataset = load_from_disk("../../data/processed/openreview-neurips")

    else:
        raise NotImplementedError("Only iclr and neurips are supported")

    paperhash2sample = {sample["paperhash"]: sample for sample in samples}
    paperhashes = list(paperhash2sample.keys())
    papers = []
    for sample in dataset["test"]:
        if sample["paperhash"] in paperhashes:
            papers.append(sample)

    # Load dataset
    tournament_ranking(
        papers,
        config["model"],
        config["seed"],
        config["dataset"],
        paperhash2sample,
        max_round=5,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--llm_provider", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["model_type"] = "llm"
    if args.llm_provider:
        config["llm_provider"] = args.llm_provider

    if args.dataset:
        config["dataset"] = args.dataset

    if config["llm_provider"] == "anthropic":
        config["model"] = config["anthropic_model"]
    elif config["llm_provider"] == "openai":
        config["model"] = config["openai_model"]

    run_llm_ranking(config)
