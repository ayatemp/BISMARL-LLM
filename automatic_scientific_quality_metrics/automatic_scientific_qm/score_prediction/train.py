"""
Training Setup for Score Prediction
"""

import argparse
import os

import pytorch_lightning as pl
import wandb
import yaml

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import isolate_rng

from automatic_scientific_qm.score_prediction.data import get_data
from automatic_scientific_qm.score_prediction.eval import eval
from automatic_scientific_qm.score_prediction.trainer import (
    ScorePrediction,
    PairwiseComparison,
)
from automatic_scientific_qm.utils.utils import seeding


def train(config: dict) -> None:
    """
    Trains a score prediction model using the provided configuration.

    Args:
        config (dict): The configuration dictionary containing the model and training parameters.

    Returns:
        None
    """
    with isolate_rng():
        # Model Directory
        model_directory = os.path.join(
            config["model_directory"], "score_prediction_model"
        )

        # Setup logger
        logger = WandbLogger(
            project=config["wandb"]["project"], save_dir=model_directory
        )

        # Load dataset
        train_dataloader, val_dataloader, test_dataloader = get_data(config)

        # Create Model
        if config["trainer"]["load"]["load"]:
            if config["data"]["pairwise_comparison"]:
                model = PairwiseComparison.load_from_checkpoint(
                    os.path.join(
                        config["trainer"]["load"]["checkpoint"],
                    )
                )
            else:
                model = ScorePrediction.load_from_checkpoint(
                    os.path.join(
                        config["trainer"]["load"]["checkpoint"],
                    )
                )
        else:
            if config["data"]["pairwise_comparison"]:
                model = PairwiseComparison(config)
            else:
                model = ScorePrediction(config)

        # Prepare callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="validation/loss",  # Metric to monitor
            filename="best_model",  # Name of the checkpoint file
            save_top_k=1,  # Save only the best model
            mode="min",  # 'min' means we want to minimize val_loss
            verbose=True,  # Print information when saving the model
        )

        # Create Trainer
        if config["training"]["distributed"]:
            if config["training"]["accelerator"] == "gpu":
                trainer = pl.Trainer(
                    default_root_dir=model_directory,
                    max_epochs=config["training"]["epochs"],
                    logger=logger,
                    accelerator="gpu",
                    devices=config["training"]["gpus"],
                    strategy=config["training"]["strategy"],
                    callbacks=[checkpoint_callback],
                )
            else:
                trainer = pl.Trainer(
                    default_root_dir=model_directory,
                    max_epochs=config["training"]["epochs"],
                    logger=logger,
                    accelerator="cpu",
                    callbacks=[checkpoint_callback],
                )
        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_epochs=config["training"]["epochs"],
                logger=logger,
                callbacks=[checkpoint_callback],
            )

        # Training
        trainer.fit(model, train_dataloader, val_dataloader)

        # Evaluation
        best_model_path = checkpoint_callback.best_model_path

        if config["data"]["pairwise_comparison"]:
            best_model = PairwiseComparison.load_from_checkpoint(best_model_path)
        else:
            best_model = ScorePrediction.load_from_checkpoint(best_model_path)

        eval(best_model, test_dataloader, config)


def get_experiment_name(config: dict) -> str:
    """
    Generates an experiment name based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        str: The generated experiment name.
    """
    experiment_name = "score_prediction"
    experiment_name += f"_{config['data']['dataset']}"
    experiment_name += (
        "_pairwise" if config["data"]["pairwise_comparison"] else "_exact"
    )
    experiment_name += f"_{config['data']['context_type']}"
    experiment_name += f"_{config['data']['score_type']}"
    experiment_name += f"_{config['data']['paper_representation']}"

    return experiment_name


def update_config_cli(args: argparse.Namespace, config: dict) -> dict:
    """
    Update the configuration dictionary based on the command-line arguments.

    Args:
        args (argparse.Namespace): The command-line arguments.
        config (dict): The configuration dictionary.

    Returns:
        dict: The updated configuration dictionary.
    """
    if args.percentage:
        config["data"]["percentage"] = args.percentage

    if args.seed:
        config["seed"] = args.seed

    if args.dataset:
        config["data"]["dataset"] = args.dataset

    if args.pairwise:
        config["data"]["pairwise_comparison"] = True

    if args.score:
        config["data"]["score_to_predict"] = args.score

    if args.paper_representation:
        config["data"]["paper_representation"] = args.paper_representation
        if args.paper_representation == "hypothesis":
            config["data"]["filters"].append("hypothesis")

    if args.topic_id:
        config["data"]["topic_id"] = args.topic_id

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="Percentage of training data to use",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use")
    parser.add_argument(
        "--pairwise", action="store_true", help="Use pairwise comparison"
    )
    parser.add_argument("--score", type=str, default=None, help="Score to predict")
    parser.add_argument(
        "--paper_representation",
        type=str,
        default=None,
        help="Representation of the paper",
    )
    parser.add_argument(
        "--topic_id", type=str, default=None, help="Topic id to sample the data by"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Update config with CL arguments
    config = update_config_cli(args, config)

    # Create experiment name from config
    config["wandb"]["name"] = get_experiment_name(config)

    # Seeding
    seeding(config["seed"])

    # Update wandb config
    wandb.init(**config["wandb"])
    wandb.config.update(config)

    train(config)
