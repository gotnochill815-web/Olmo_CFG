# ChemFM Classifier-Free Guidance (CFG)

PyTorch Lightning implementation of **Classifier-Free Guidance (CFG)** fine-tuning for **ChemFM (OLMo-7B)** using **QLoRA** for controllable molecular generation.

This project reproduces the training methodology of Classifier-Free Guidance by combining:

- Property-conditioned molecule generation
- Random condition dropout (CFG training)
- LoRA fine-tuning
- 4-bit QLoRA
- PyTorch Lightning
- Mixed precision training

---

# Features

- ✅ PyTorch Lightning training pipeline
- ✅ True Classifier-Free Guidance (CFG) training
- ✅ Random condition dropout
- ✅ Multi-property conditioning
- ✅ Single / Pair / Triple property conditioning
- ✅ QLoRA (4-bit NF4)
- ✅ LoRA fine-tuning
- ✅ Gradient checkpointing
- ✅ Mixed precision (BF16 / FP16)
- ✅ TensorBoard logging
- ✅ Automatic checkpointing
- ✅ Modular YAML configuration

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

# Dataset Format

Each CSV should contain the following columns.

| Column | Description |
|---------|-------------|
| smiles | Canonical SMILES |
| qed | Quantitative Estimate of Drug-likeness |
| logp | Partition Coefficient |
| tpsa | Topological Polar Surface Area |
| sas | Synthetic Accessibility Score |

Example

| smiles | qed | logp | tpsa | sas |
|--------|------|------|------|------|
| CCO | 0.71 | 1.42 | 20.2 | 2.31 |

---

# Model

Backbone

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

Training uses

- LoRA
- QLoRA (4-bit NF4)
- Gradient Checkpointing
- AdamW
- Cosine LR Scheduler

---

# Classifier-Free Guidance Training

Unlike standard conditional language modeling, this project performs **Classifier-Free Guidance (CFG)** training by randomly removing all conditioning information during training.

For every batch:

- **(1 − dropout_prob)** → Conditional training
- **dropout_prob** → Unconditional training

Example

Conditional sample

```
<pstart>

<QED> 0.82
<LOGP> 2.11
<TPSA> 41.5
<SAS> 2.34

<molstart>
CCO...
```

Unconditional sample

```
<pstart>

<molstart>
CCO...
```

This allows the model to learn both conditional and unconditional molecule generation, enabling CFG inference after training.

---

# Supported Conditioning Modes

The data collator supports multiple conditioning strategies.

### All properties

```
QED
LOGP
TPSA
SAS
```

### Single property

Randomly keeps one property.

### Pair conditioning

Randomly keeps two properties.

### Triple conditioning

Randomly keeps three properties.

### Random conditioning

Randomly chooses one of:

- Single
- Pair
- Triple
- All

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
OLMo-7B + LoRA
      │
      ▼
AdamW
      │
      ▼
Cosine Scheduler
      │
      ▼
PyTorch Lightning Trainer
```

---

# Logging

TensorBoard

```bash
tensorboard --logdir logs
```

---

# Checkpoints

Best checkpoints are automatically saved during training.

Example

```
checkpoints/

    last.ckpt

logs/

    cfg/
```

---

# Configuration

Training configuration

```
configs/training/
```

Dataset configuration

```
configs/dataset/
```

Model configuration

```
configs/model/
```

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

---

# Current Status

### Implemented

- PyTorch Lightning training
- QLoRA fine-tuning
- LoRA adapters
- Gradient checkpointing
- Random property conditioning
- True Classifier-Free Guidance training
- Multi-property conditioning
- Automatic checkpointing
- TensorBoard logging
