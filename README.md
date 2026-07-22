# ChemFM_CFG

Classifier-Free Guidance (CFG) Fine-Tuning for Conditional Molecular Generation using OLMo-7B.

This project extends the ChemFM framework by enabling conditional molecular generation using molecular properties through prompt-based conditioning and LoRA fine-tuning.

---

# Overview

ChemFM_CFG enables conditional molecular generation by conditioning the language model on molecular properties instead of generating molecules unconditionally.

Supported molecular properties:

- QED (Quantitative Estimate of Drug-likeness)
- LogP
- TPSA (Topological Polar Surface Area)
- SAS (Synthetic Accessibility Score)

The project is built on top of the **OLMo-7B PubChem** checkpoint and fine-tunes the model using **LoRA** with **4-bit quantization** for efficient training.

---

# Features

- LoRA Fine-Tuning
- 4-bit Quantization (BitsAndBytes)
- Gradient Checkpointing
- FP16 Mixed Precision Training
- Prompt-based Molecular Property Conditioning
- Multi-property Conditioning
- HuggingFace Trainer Pipeline
- Config-driven Training

---

# Repository Structure

```
ChemFM_CFG/
│
├── configs/
│   ├── dataset/
│   ├── model/
│   └── training/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── cfg/
│   ├── dataset/
│   ├── training/
│   ├── generation/
│   └── utils/
│
├── tokenizer/
├── checkpoints/
├── outputs/
│
├── train.py
├── generate.py
├── benchmark_cfg.py
├── requirements.txt
└── README.md
```

---

# Requirements

Recommended Environment

- Python 3.11+
- CUDA GPU
- PyTorch
- Transformers
- PEFT
- BitsAndBytes
- RDKit

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Hugging Face Authentication

The base model is downloaded from Hugging Face.

Login using

```bash
huggingface-cli login
```

or

```bash
export HF_TOKEN=<your_token>
```

If no token is provided, downloads will still work but may be slower due to anonymous rate limits.

---

# Dataset

Expected directory structure

```
data/
└── processed/
    ├── train_10000.csv
    ├── val_10000.csv
    └── test_10000.csv
```

Each CSV should contain the following columns

| Column | Description |
|---------|-------------|
| smiles | Molecular SMILES |
| qed | QED |
| logp | LogP |
| tpsa | TPSA |
| sas | SAS |

Dataset configuration

```
configs/dataset/guacamol_10k.yaml
```

Example

```yaml
name: guacamol_10k

max_length: 128

smiles_column: smiles

property_columns:
  qed: qed
  logp: logp
  tpsa: tpsa
  sas: sas

train:
  path: data/processed/train_10000.csv

validation:
  path: data/processed/val_10000.csv

test:
  path: data/processed/test_10000.csv
```

---

# Model Configuration

Configuration file

```
configs/model/olmo_7b.yaml
```

Current base model

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

Current configuration

- LoRA Enabled
- 4-bit Quantization
- Gradient Checkpointing
- Automatic Device Mapping

Special tokens

```
<QED>
<LOGP>
<TPSA>
<SAS>
<NULL>
<pstart>
<molstart>
```

---

# Training Configuration

Training configuration

```
configs/training/default_10000.yaml
```

Example

```yaml
batch_size: 4

gradient_accumulation_steps: 4

epochs: 3

learning_rate: 1e-4

warmup_steps: 15

weight_decay: 0.01

fp16: true

bf16: false

conditioning:
    mode: random
```

---

# Conditioning Modes

The training pipeline supports multiple conditioning strategies.

Available modes

- all
- single
- pair
- triple
- random

Current training uses

```yaml
conditioning:
    mode: random
```

For every training sample, one conditioning strategy is randomly selected.

Examples include

### Single Property

```
<pstart>
<QED> 0.82
<molstart>
CCO...
```

### Pair Conditioning

```
<pstart>
<QED> 0.82
<LOGP> 2.45
<molstart>
CCO...
```

### Triple Conditioning

```
<pstart>
<QED> 0.82
<LOGP> 2.45
<TPSA> 78.3
<molstart>
CCO...
```

### All Properties

```
<pstart>
<QED> 0.82
<LOGP> 2.45
<TPSA> 78.3
<SAS> 2.1
<molstart>
CCO...
```

This allows the model to learn flexible molecular conditioning using different combinations of properties.

---

# Training

Start training using

```bash
python train.py \
    --dataset configs/dataset/guacamol_10k.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_10000.yaml
```

---

# Training Output

Fine-tuned LoRA checkpoints are saved to

```
checkpoints/lora_cfg_10000/
```

---

# Verified Training Pipeline

The complete training pipeline has been verified successfully on a fresh Google Colab environment.

Verified components

- Configuration loading
- Dataset loading
- Tokenizer loading
- OLMo model loading
- LoRA initialization
- 4-bit quantization
- Data Collator
- HuggingFace Trainer
- Checkpoint Saving

Example output

```
Train samples : 8000
Valid samples : 1000

trainable params: 19,988,480
all params: 6,907,944,960
trainable%: 0.2894

Training Complete!
```

---

# Notes

Training updates only the LoRA adapter weights.

The full OLMo-7B model remains frozen during fine-tuning.

Training uses

- LoRA
- FP16 Mixed Precision
- Gradient Checkpointing
- 4-bit Quantization
- Prompt-based Molecular Property Conditioning

---

# Current Status

 Training pipeline verified

- Configuration loading
- Dataset loading
- Model loading
- LoRA fine-tuning
- Checkpoint saving

Generation utilities are included in the repository but are still under development and were not part of the verified training workflow.

---

# Citation

If you use this project, please cite the original ChemFM framework and the OLMo PubChem checkpoint used for fine-tuning.

---

# Acknowledgements

This project builds upon:

- ChemFM
- OLMo
- Hugging Face Transformers
- PEFT
- BitsAndBytes
- RDKit

Special thanks to the original authors and contributors of these open-source projects.

---
