"""
Training Loop for Section Classifier of Academic Papers
"""

import argparse
import os

import pytorch_lightning as pl
import wandb
import yaml

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import isolate_rng

from automatic_scientific_qm.section_classification.data import get_data
from automatic_scientific_qm.utils.utils import seeding
from automatic_scientific_qm.section_classification.trainer import SectionClassifier


def train_section_classifier(config: dict) -> None:
    """
    Trains a section classifier model based on the provided configuration.

    Args:
        config (dict): Configuration parameters for training the model.

    Returns:
        None
    """
    with isolate_rng():
        # Model Directory
        model_directory = os.path.join(config["model_directory"], "section_classifier")
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Setup logger
        logger = WandbLogger(
            project=config["logging"]["project"], save_dir=model_directory
        )

        train_dataloader, val_dataloader, test_dataloader = get_data(config)

        # Create Model
        if config["trainer"]["load"]["load"]:
            model = SectionClassifier.load_from_checkpoint(
                config["trainer"]["load"]["checkpoint"],
            )
        else:
            model = SectionClassifier(config["trainer"])

        # Prepare callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="validation/accuracy",
            filename="best_model",
            save_top_k=1,
            mode="max",
            verbose=True,
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

        # Evaluate best checkpoint
        best_model = SectionClassifier.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        trainer.test(best_model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a section classifier for academic papers"
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    wandb.init(**config["logging"])
    seeding(config["seed"])
    train_section_classifier(config)
