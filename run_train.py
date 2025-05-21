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
    # wandb.login(key=os.environ["WANDB_KEY"], host=os.environ["WANDB_HOST"])
    wandb.init(
        project="codi",
        name=os.environ["WANDB_NAME"],
        mode="disabled",
    )
    # wandb.config.update(config)
    setup_seed()

    # Initialize dataloader and model
    train_dataloader, val_dataloader = get_dataloader(config)
    model = BaseModel(config=config)

    # Initialize trainer
    trainer = PytorchTrainer(
        config.train,
        model,
        train_dataloader,
        val_dataloader,
        perform_sanity_check=True,
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
