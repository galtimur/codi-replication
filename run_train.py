"""
File is taken from kotlin-initiative repo
"""


import os
import random

import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from codi_model import BaseModel, CODIModel
from data import get_dataloader
from trainer import PytorchTrainer


def setup_seed(seed=0xDEADC0DE, cudnn_benchmark=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark


def train(config: DictConfig) -> None:
    wandb.login(key=os.environ["WANDB_KEY"], host=os.environ["WANDB_HOST"])
    wandb.init(
        project="codi",
        name=config.model.model_type+"-"+os.environ["WANDB_NAME"],
        # mode="disabled",
    )
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb.config.update(config_dict)
    setup_seed()

    # Initialize dataloader and model
    train_dataloader, val_dataloaders = get_dataloader(config)
    if config.model.model_type == "base":
        model = BaseModel(config=config)
    elif config.model.model_type == "codi":
        model = CODIModel(config=config)
    else:
        raise ValueError("Invalid model type")
    print("Training {config.model.model_type} model")

    # Initialize trainer
    trainer = PytorchTrainer(
        config.train,
        model,
        train_dataloader,
        val_dataloaders,
        perform_sanity_check=True,
        model_type=config.model.model_type,
    )

    trainer.run_training()
    wandb.finish()


def main(config_path: str = "configs/config.yaml"):
    config = OmegaConf.load(config_path)

    load_dotenv()
    train(config)

    wandb.finish()


if __name__ == "__main__":
    os.environ["WANDB_NAME"] = wandb.util.generate_id()
    main()
