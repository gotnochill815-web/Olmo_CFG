# ChemFM CFG: Conditional Molecular Generation with Classifier-Free Guidance

This repository implements **Classifier-Free Guidance (CFG)** for molecular generation using the **OLMo-7B PubChem** language model.

The project extends a pretrained molecular language model with property-conditioned generation using QLoRA, enabling controlled generation of molecules based on molecular descriptors.

---

## Features

- Classifier-Free Guidance (CFG) for molecular generation
- QLoRA fine-tuning (4-bit quantization)
- Property conditioning using:
  - QED
  - LogP
  - TPSA
  - SAS
- Custom tokenizer with molecular property tokens
- Modular training and inference pipeline
- Evaluation utilities

---

## Repository Structure

```
.
├── configs/
│   ├── dataset/
│   ├── model/
│   └── training/
│
├── data/
│   ├── guacamol/
│   │   ├── test.smiles
│   │   └── valid.smiles
│
├── src/
│   ├── cfg/
│   ├── dataset/
│   ├── evaluation/
│   ├── generation/
│   ├── inference/
│   ├── training/
│   └── utils/
│
├── tokenizer/
├── notebooks/
├── train.py
├── generate.py
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

# Base Model

The project uses

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

which is automatically downloaded through Hugging Face.

If the repository is gated, login first

```python
from huggingface_hub import login
login()
```

---

# Configuration

Model configuration

```
configs/model/olmo_7b.yaml
```

Training configuration

```
configs/training/default.yaml
```

Dataset configuration

```
configs/dataset/
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

# Training

Run

```bash
python train.py
```

Training uses

- QLoRA
- 4-bit quantization
- Gradient checkpointing
- LoRA adapters

Outputs are saved to

```
checkpoints/
```

---

# Generation

Generate molecules using

```bash
python generate.py
```

The generation pipeline supports Classifier-Free Guidance using molecular property prompts.

---

# Property Tokens

The tokenizer is extended with

```
<QED>
<LOGP>
<TPSA>
<SAS>
<NULL>
<pstart>
<molstart>
```

These tokens enable conditioning on molecular properties during generation.

---

# Dataset

This repository does **not** include the full training dataset because of GitHub file size limits.

Training data should be prepared separately.

The repository contains

```
data/
├── guacamol/
│   ├── valid.smiles
│   └── test.smiles
```

---

# Evaluation

Evaluation utilities are available under

```
src/evaluation/
```

Metrics include molecular validity and property evaluation.

---

# Requirements

- Python 3.10+
- PyTorch
- CUDA GPU (recommended)
- Transformers
- PEFT
- BitsAndBytes

---
