# ChemFM-CFG: Property-Conditioned Molecular Generation using Classifier-Free Guidance

Implementation of **Classifier-Free Guidance (CFG)** for property-conditioned molecular generation using the **OLMo-7B molecular language model**.

This project extends the pretrained **OLMo-7B PubChem** model with **QLoRA** fine-tuning and classifier-free guidance to generate molecules conditioned on molecular properties such as **QED, LogP, TPSA, and SAS**.

---

# Features

- Property-conditioned molecular generation
- Classifier-Free Guidance (CFG)
- QLoRA fine-tuning (4-bit quantization)
- OLMo-7B molecular language model
- Custom tokenizer with molecular property tokens
- Config-based training pipeline
- Modular codebase
- Easy experimentation through YAML configuration files

---

# Repository Structure

```
Olmo_CFG/
│
├── configs/
│   ├── dataset/
│   │   └── guacamol_10k.yaml
│   ├── model/
│   │   └── olmo_7b.yaml
│   └── training/
│       └── default_10000.yaml
│
├── data/
│   └── guacamol/
│       ├── train_10000.csv
│       ├── val_10000.csv
│       ├── test_10000.csv
│       ├── valid.smiles
│       └── test.smiles
│
├── src/
│   ├── cfg/
│   ├── dataset/
│   ├── evaluation/
│   ├── generation/
│   ├── training/
│   └── utils/
│
├── tokenizer/
│
├── notebooks/
│
├── train.py
├── generate.py
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

# Requirements

- Python 3.10+
- CUDA-enabled GPU (recommended)
- PyTorch
- Transformers
- PEFT
- BitsAndBytes
- Accelerate
- RDKit

---

# Base Model

This project uses the pretrained model

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

The model will automatically be downloaded from Hugging Face during the first run.

(Optional) Authenticate with Hugging Face for faster downloads:

```python
from huggingface_hub import login
login()
```

or

```bash
export HF_TOKEN=<your_huggingface_token>
```

---

# Dataset

The repository includes the processed GuacaMol dataset used for the 10k experiment.

```
data/
└── guacamol/
    ├── train_10000.csv
    ├── val_10000.csv
    ├── test_10000.csv
    ├── valid.smiles
    └── test.smiles
```

Dataset configuration:

```
configs/dataset/guacamol_10k.yaml
```

---

# Configuration Files

The training pipeline is entirely configuration driven.

## Dataset Configuration

```
configs/dataset/guacamol_10k.yaml
```

Contains:

- dataset paths
- smiles column
- property columns
- maximum sequence length

---

## Model Configuration

```
configs/model/olmo_7b.yaml
```

Contains:

- pretrained model name
- LoRA configuration
- quantization settings
- tokenizer special tokens

---

## Training Configuration

```
configs/training/default_10000.yaml
```

Contains:

- learning rate
- batch size
- gradient accumulation
- number of epochs
- checkpoint directory
- mixed precision settings

---

# Training

Start training using

```bash
python train.py \
    --dataset configs/dataset/guacamol_10k.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_10000.yaml
```

The training pipeline performs:

- Load configuration files
- Load tokenizer
- Load pretrained OLMo model
- Apply QLoRA adapters
- Load dataset
- Create dataloaders
- Train the model
- Save LoRA checkpoints

Checkpoints are saved in

```
checkpoints/lora_cfg_10000/
```

---

# Loading the Model

```python
import yaml

from src.training.load_model import load_model

with open("configs/model/olmo_7b.yaml") as f:
    model_cfg = yaml.safe_load(f)

model, tokenizer = load_model(model_cfg)
```

---

# Property Conditioning

The tokenizer includes the following conditioning tokens

```
<QED>
<LOGP>
<TPSA>
<SAS>
<NULL>
<pstart>
<molstart>
```

These tokens enable conditioning the language model on molecular properties during training and generation.

---

# Generation

After training, generate molecules using

```bash
python generate.py
```

Generation supports property-conditioned molecular generation using Classifier-Free Guidance.

---

# Evaluation

Evaluation utilities are located in

```
src/evaluation/
```

Typical evaluation metrics include:

- Validity
- Uniqueness
- Novelty
- Property distribution
- Molecular quality metrics

---

# Example Workflow

```text
Clone Repository
        │
        ▼
Install Dependencies
        │
        ▼
Download Base Model
        │
        ▼
Load Dataset
        │
        ▼
Apply LoRA
        │
        ▼
Train Model
        │
        ▼
Save Checkpoints
        │
        ▼
Generate Molecules
```

---

