"""
Main training script.
"""

import argparse
import os
from pprint import pprint

from src.utils.config import load_yaml
from src.training.load_model import load_model
from src.dataset.dataset import load_datasets
from src.training.collator import CFGDataCollator
from src.training.train import build_trainer


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
    print("Training config path:", args.training)
    print("=" * 80)

    print("=" * 80)
    print("Loading Configs")
    print("=" * 80)

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
    print("=" * 80)

    # -----------------------------
    # Validate model config
    # -----------------------------
    required_keys = [
        "model_name",
        "load_in_4bit",
        "device_map",
        "gradient_checkpointing",
        "special_tokens",
        "lora",
    ]

    missing = [k for k in required_keys if k not in model_cfg]

    if missing:
        raise ValueError(
            f"Model config is missing keys: {missing}\n"
            f"Loaded config:\n{model_cfg}"
        )

    print("=" * 80)
    print("Loading Model")
    print("=" * 80)

    model, tokenizer = load_model(model_cfg)

    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    train_dataset, val_dataset, _ = load_datasets(dataset_cfg)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Valid samples : {len(val_dataset)}")

    print("=" * 80)
    print("Creating Data Collator")
    print("=" * 80)

    collator = CFGDataCollator(
        tokenizer=tokenizer,
        dropout_prob=training_cfg.get("dropout_prob", 0.2),
        max_length=dataset_cfg["max_length"],
        conditioning_mode=training_cfg["conditioning"]["mode"],
    )

    os.makedirs(
        training_cfg["output_dir"],
        exist_ok=True,
    )

    print("=" * 80)
    print("Building Trainer")
    print("=" * 80)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        collator=collator,
        training_cfg=training_cfg,
    )

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)

    trainer.train()

    print("=" * 80)
    print("Saving Model")
    print("=" * 80)

    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])

    print("Training Complete!")


if __name__ == "__main__":
    main()