"""
Main training script using PyTorch Lightning.
"""

import argparse
import os
from pprint import pprint

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils.config import load_yaml
from src.training.load_model import load_model
from src.dataset.dataset import load_datasets
from src.training.collator import CFGDataCollator

from src.lightning.lightning_module import CFGLightningModule
from src.lightning.datamodule import CFGDataModule

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset config yaml",
    )

    parser.add_argument(
        "--model",
        default="configs/model/olmo_7b.yaml",
    )

    parser.add_argument(
        "--training",
        default="configs/training/default.yaml",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Arguments")
    print(args)
    print("=" * 80)

    ###########################################################
    # Load Configs
    ###########################################################

    dataset_cfg = load_yaml(args.dataset)
    model_cfg = load_yaml(args.model)
    training_cfg = load_yaml(args.training)

    print("=" * 80)
    print("Dataset Config")
    pprint(dataset_cfg)

    print("=" * 80)
    print("Model Config")
    pprint(model_cfg)

    print("=" * 80)
    print("Training Config")
    pprint(training_cfg)

    ###########################################################
    # Seed
    ###########################################################

    L.seed_everything(training_cfg.get("seed", 42), workers=True)

    ###########################################################
    # Load model
    ###########################################################

    print("=" * 80)
    print("Loading Model")
    print("=" * 80)

    model, tokenizer = load_model(model_cfg)

    ###########################################################
    # Load datasets
    ###########################################################

    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    train_dataset, val_dataset, test_dataset = load_datasets(dataset_cfg)

    print(f"Train : {len(train_dataset)}")
    print(f"Valid : {len(val_dataset)}")

    ###########################################################
    # DataModule
    ###########################################################

    datamodule = CFGDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 4),
        dropout_prob=training_cfg.get("dropout_prob", 0.2),
        max_length=dataset_cfg["max_length"],
        conditioning_mode=training_cfg["conditioning"]["mode"],
    )

    ###########################################################
    # Optimizer
    ###########################################################

    lr = float(training_cfg["learning_rate"])

    optimizer_partial = lambda params: AdamW(
        params,
        lr=lr,
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    ###########################################################
    # Scheduler
    ###########################################################

    steps_per_epoch = (
        len(train_dataset)
        + training_cfg["batch_size"]
        - 1
    ) // training_cfg["batch_size"]

    total_steps = (
        steps_per_epoch
        * training_cfg["epochs"]
    ) // training_cfg["gradient_accumulation_steps"]

    def scheduler_partial(optimizer):
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_cfg.get("warmup_steps", 0),
            num_training_steps=total_steps,
        )

    ###########################################################
    # Lightning Module
    ###########################################################

    lightning_model = CFGLightningModule(
        model=model,
        optimizer_partial=optimizer_partial,
        scheduler_partials=[scheduler_partial],
    )

    ###########################################################
    # Logging
    ###########################################################

    logger = TensorBoardLogger(
        save_dir="logs",
        name="cfg",
    )

    ###########################################################
    # Callbacks
    ###########################################################

    callbacks = [

        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="best-{epoch}-{val_loss:.4f}",
        ),

        LearningRateMonitor(
            logging_interval="step"
        ),

        EarlyStopping(
            monitor="val/loss",
            patience=5,
            mode="min",
        ),
    ]

    ###########################################################
    # Trainer
    ###########################################################

    trainer = L.Trainer(

        accelerator="gpu",

        devices=training_cfg.get(
            "devices",
            1,
        ),

        strategy=training_cfg.get(
            "strategy",
            "auto",
        ),

        precision=(
            "bf16-mixed"
            if training_cfg.get("bf16", False)
            else "16-mixed"
        ),

        max_epochs=training_cfg["epochs"],

        accumulate_grad_batches=training_cfg.get(
            "gradient_accumulation_steps",
            1,
        ),

        gradient_clip_val=training_cfg.get(
            "max_grad_norm",
            1.0,
        ),

        log_every_n_steps=training_cfg.get(
            "logging_steps",
            10,
        ),

        callbacks=callbacks,

        logger=logger,
    )

    ###########################################################
    # Train
    ###########################################################

    print("=" * 80)
    print("Starting Lightning Training")
    print("=" * 80)

    trainer.fit(
        lightning_model,
        datamodule=datamodule,
    )

    ###########################################################
    # Save
    ###########################################################

    save_dir = training_cfg["output_dir"]

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    trainer.save_checkpoint(
        os.path.join(
            save_dir,
            "last.ckpt",
        )
    )

    tokenizer.save_pretrained(save_dir)

    print("=" * 80)
    print("Training Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
