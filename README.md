# ChemFM Classifier-Free Guidance (CFG)

PyTorch Lightning implementation of **Classifier-Free Guidance (CFG)** fine-tuning for **ChemFM (OLMo-7B)** using **QLoRA** for controllable molecular generation.

This repository reproduces classifier-free guidance training for molecular language models by combining property-conditioned generation with random condition dropout, enabling both conditional and unconditional molecule generation from a single model.

---

# Features

- PyTorch Lightning training pipeline
- Classifier-Free Guidance (CFG) training
- Random condition dropout
- Property-conditioned molecular generation
- Single, pair, triple, and all-property conditioning
- QLoRA (4-bit NF4)
- LoRA fine-tuning
- Gradient checkpointing
- Mixed precision training (BF16 / FP16)
- TensorBoard logging
- Automatic checkpointing
- YAML-based configuration

---

# Project Structure

```text
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
│       └── test_50000.csv
│
├── src/
│   ├── dataset/
│   │   └── dataset.py
│   │
│   ├── lightning/
│   │   ├── datamodule.py
│   │   └── lightning_module.py
│   │
│   ├── training/
│   │   ├── collator.py
│   │   ├── load_model.py
│   │   └── utils.py
│   │
│   └── utils/
│       └── config.py
│
├── train.py
├── requirements.txt
└── README.md
```

---

# Installation

## Step 1: Clone the repository

```bash
git clone https://github.com/gotnochill815-web/Olmo_CFG.git
cd Olmo_CFG
```

---

## Step 2: Create a virtual environment (Recommended)

Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The project expects processed GuacaMol CSV files.

Each CSV must contain the following columns.

| Column | Description |
|---------|-------------|
| smiles | Canonical SMILES |
| qed | Quantitative Estimate of Drug-likeness |
| logp | Partition Coefficient |
| tpsa | Topological Polar Surface Area |
| sas | Synthetic Accessibility Score |

Example

| smiles | qed | logp | tpsa | sas |
|--------|-----|------|------|-----|
| CCO | 0.71 | 1.42 | 20.2 | 2.31 |

Place the dataset inside

```text
data/
└── guacamol/
    ├── train_10000.csv
    ├── val_10000.csv
    ├── test_10000.csv
    ├── train_50000.csv
    ├── val_50000.csv
    └── test_50000.csv
```

---

# Model

Base model

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

Training uses

- QLoRA (4-bit NF4)
- LoRA
- AdamW
- Cosine Learning Rate Scheduler
- Gradient Checkpointing
- Mixed Precision Training

---

# Configuration

The repository uses YAML configuration files.

## Dataset

```
configs/dataset/
```

## Model

```
configs/model/
```

## Training

```
configs/training/
```

---

# How to Run

## Step 1

Verify the dataset configuration.

Example

```
configs/dataset/guacamol_10k.yaml
```

contains the correct dataset paths.

---

## Step 2

Verify the model configuration.

```
configs/model/olmo_7b.yaml
```

---

## Step 3

Verify the training configuration.

```
configs/training/default_10000.yaml
```

or

```
configs/training/default_50000.yaml
```

---

## Step 4

Train on the 10k dataset.

```bash
python train.py \
    --dataset configs/dataset/guacamol_10k.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_10000.yaml
```

---

## Step 5

Train on the 50k dataset.

```bash
python train.py \
    --dataset configs/dataset/guacamol_50000.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_50000.yaml
```

---

## Step 6

Monitor training.

```bash
tensorboard --logdir logs
```

Open

```
http://localhost:6006
```

to visualize

- Training loss
- Validation loss
- Learning rate
- Checkpoints

---

## Step 7

After training completes, the repository automatically saves

```
checkpoints/
    last.ckpt
```

TensorBoard logs are stored in

```
logs/
    cfg/
```

---

# Classifier-Free Guidance Training

Unlike standard conditional language modeling, this implementation performs **Classifier-Free Guidance (CFG)** training.

During training, molecular property conditioning is randomly removed with a fixed probability.

```
(1 − dropout_prob)
        ↓
Conditional Training

dropout_prob
        ↓
Unconditional Training
```

Conditional example

```text
<pstart>

<QED> 0.82
<LOGP> 2.11
<TPSA> 41.5
<SAS> 2.34

<molstart>
CCO...
```

Unconditional example

```text
<pstart>

<molstart>
CCO...
```

This allows the model to learn both conditional and unconditional molecule generation, enabling CFG during inference.

---

# Conditioning Modes

The data collator supports multiple conditioning strategies.

## All-property conditioning

- QED
- LOGP
- TPSA
- SAS

## Single-property conditioning

Randomly keeps one property.

## Pair conditioning

Randomly keeps two properties.

## Triple conditioning

Randomly keeps three properties.

## Random conditioning

Randomly selects one of

- Single
- Pair
- Triple
- All

---

# Training Pipeline

```text
CSV Dataset
      │
      ▼
Dataset Loader
      │
      ▼
CFG Data Collator
      │
      ├── Random Property Conditioning
      └── Classifier-Free Dropout
      │
      ▼
Tokenizer
      │
      ▼
Lightning DataModule
      │
      ▼
OLMo-7B + QLoRA
      │
      ▼
AdamW Optimizer
      │
      ▼
Cosine Learning Rate Scheduler
      │
      ▼
PyTorch Lightning Trainer
```

---

# Logging

Launch TensorBoard

```bash
tensorboard --logdir logs
```

---

# Checkpoints

Training automatically stores model checkpoints.

```
checkpoints/
    last.ckpt
```

Training logs

```
logs/
    cfg/
```

---

# Requirements

- Python 3.10+
- PyTorch
- PyTorch Lightning
- Transformers
- PEFT
- BitsAndBytes
- RDKit
- pandas
- numpy

---

# Current Status

Implemented

- PyTorch Lightning training pipeline
- QLoRA fine-tuning
- LoRA adapters
- Gradient checkpointing
- Classifier-Free Guidance (CFG)
- Random property conditioning
- Multi-property conditioning
- TensorBoard logging
- Automatic checkpointing
- YAML configuration system

---

# Future Work

- CFG inference with adjustable guidance scale
- Distributed multi-GPU training
- Additional molecular property conditioning
- Molecule evaluation pipeline
- Sampling utilities
- Hugging Face model export

---

# Acknowledgements

This implementation builds upon:

- ChemFM
- OLMo
- Hugging Face Transformers
- PEFT
- PyTorch Lightning
