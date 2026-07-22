# ChemFM Classifier-Free Guidance (CFG)

PyTorch Lightning implementation of **Classifier-Free Guidance (CFG)** fine-tuning for **ChemFM (OLMo-7B)** on molecular property-conditioned generation.

This project fine-tunes the ChemFM foundation model using LoRA and 4-bit quantization for controllable molecule generation conditioned on molecular properties such as:

- QED
- LogP
- TPSA
- SAS

---

# Features

- PyTorch Lightning training pipeline
- LoRA fine-tuning (PEFT)
- 4-bit QLoRA (BitsAndBytes)
- Mixed Precision (BF16/FP16)
- Property-conditioned generation
- Random CFG dropout
- Modular configuration using YAML
- TensorBoard logging
- Model checkpointing

---

# Project Structure

```
Olmo_CFG/
│
├── configs/
│   ├── dataset/
│   │   ├── guacamol_10k.yaml
│   │   └── guacamol_50000.yaml
│   │
│   ├── model/
│   │   └── olmo_7b.yaml
│   │
│   └── training/
│       ├── default_10000.yaml
│       └── default_50000.yaml
│
├── data/
│   └── guacamol/
│       ├── train_10000.csv
│       ├── val_10000.csv
│       ├── test_10000.csv
│       ├── train_50000.csv
│       ├── val_50000.csv
│       ├── test_50000.csv
│       ├── test.smiles
│       └── valid.smiles
│
├── src/
│   ├── lightning/
│   │   ├── datamodule.py
│   │   └── lightning_module.py
│   │
│   ├── training/
│   │   ├── collator.py
│   │   ├── dataset.py
│   │   ├── load_dataset.py
│   │   ├── load_model.py
│   │   └── utils.py
│   │
│   └── ...
│
├── train.py
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/gotnochill815-web/Olmo_CFG.git

cd Olmo_CFG
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Each CSV must contain

| Column | Description |
|----------|-------------|
| smiles | Molecular SMILES |
| qed | QED score |
| logp | LogP |
| sas | Synthetic Accessibility Score |
| tpsa | Topological Polar Surface Area |

---

# Model

Current backbone

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

Training uses

- LoRA
- QLoRA (4-bit)
- Gradient Checkpointing

---

# Training

## Train on 10k dataset

```bash
python train.py \
    --dataset configs/dataset/guacamol_10k.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_10000.yaml
```

---

## Train on 50k dataset

```bash
python train.py \
    --dataset configs/dataset/guacamol_50000.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_50000.yaml
```

---

# Configuration

Dataset configuration

```
configs/dataset/
```

Training configuration

```
configs/training/
```

Model configuration

```
configs/model/
```

---

# Training Pipeline

```
CSV Dataset
      │
      ▼
Dataset Loader
      │
      ▼
CFG Data Collator
      │
      ▼
Lightning DataModule
      │
      ▼
Lightning Module
      │
      ▼
LoRA OLMo-7B
      │
      ▼
Optimizer
      │
      ▼
Cosine Scheduler
      │
      ▼
PyTorch Lightning Trainer
```

---

# Lightning Features

Implemented

- LightningModule
- LightningDataModule
- Automatic Mixed Precision
- Gradient Accumulation
- Distributed Training
- Model Checkpointing
- TensorBoard Logging
- Validation Loop

---

# Checkpoints

Saved automatically to

```
checkpoints/
```

Example

```
checkpoints/
    lora_cfg_10000/

checkpoints/
    lora_cfg_50000/
```

---

# Logging

TensorBoard logs

```bash
tensorboard --logdir lightning_logs
```

---

# Supported Training Modes

Current

- 10k GuacaMol
- 50k GuacaMol

Planned

- Full GuacaMol
- ChEMBL
- MOSES

---

# Requirements

- Python 3.10+
- PyTorch
- Lightning
- Transformers
- PEFT
- BitsAndBytes
- RDKit
- pandas
- numpy
