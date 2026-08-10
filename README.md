# OLMo-7B Classifier-Free Guided Conditional Molecular Generation

This repository implements **parameter-efficient fine-tuning (QLoRA)** of **OLMo-7B** for **conditional molecular generation** using **Classifier-Free Guidance (CFG)**. The model learns to generate molecular SMILES conditioned on molecular properties while retaining unconditional generation capability through random conditioning dropout.

Built using **PyTorch Lightning**, **Hugging Face Transformers**, **PEFT**, and **BitsAndBytes**, the project supports efficient fine-tuning on modern GPUs and is compatible with both single-GPU and Distributed Data Parallel (DDP) training.

---

# Features

- OLMo-7B fine-tuning with QLoRA (4-bit)
- Classifier-Free Guidance (CFG)
- Conditional molecular generation
- Random conditioning dropout
- Mixed Precision (BF16)
- Gradient Accumulation
- LoRA adapters
- Automatic checkpointing
- TensorBoard logging
- Distributed Data Parallel (DDP) compatible
- dataset for 100k : https://huggingface.co/datasets/prakhya15/guacamol-cfg-100k
- dataset for 500k : https://huggingface.co/datasets/prakhya15/guacamol-cfg-500k

---

# Repository Structure

```text
.
├── configs/
│   ├── dataset/
│   ├── model/
│   └── training/
│
├── lightning/
│   ├── datamodule.py
│   └── lightning_module.py
│
├── models/
│   └── load_model.py
│
├── utils/
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

Create a virtual environment

```bash
python -m venv venv
```

Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Base Model

```
harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos
```

---

# Training Configuration

Example configuration

```yaml
accelerator: gpu

devices: 1

strategy: auto

precision: bf16-mixed

batch_size: 2

gradient_accumulation_steps: 8

epochs: 3

learning_rate: 1e-4

weight_decay: 0.01

warmup_steps: 100

max_grad_norm: 1.0
```

---

# Running Training

## Single GPU

```bash
python train.py \
    --dataset configs/dataset/guacamol_50000.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_50000.yaml
```

---

# Multi-GPU Training (DDP)

This project is compatible with **PyTorch Lightning Distributed Data Parallel (DDP)**.

Update your training configuration:

```yaml
accelerator: gpu

devices: 2

strategy: ddp
```

Launch training using **torchrun**:

```bash
torchrun --nproc_per_node=2 train.py \
    --dataset configs/dataset/guacamol_50000.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_50000.yaml
```

For four GPUs:

```bash
torchrun --nproc_per_node=4 train.py \
    --dataset configs/dataset/guacamol_50000.yaml \
    --model configs/model/olmo_7b.yaml \
    --training configs/training/default_50000.yaml
```
this is the model trained on 50k : https://huggingface.co/prakhya15/Olmo-CFG-50000
---

# Verifying Multi-GPU Training

Verify that the system detects multiple GPUs.

```bash
nvidia-smi -L
```

Example

```text
GPU 0: NVIDIA A100
GPU 1: NVIDIA A100
```

Monitor GPU utilization while training.

```bash
watch -n 1 nvidia-smi
```

A successful DDP run should show:

- Memory allocated on all participating GPUs
- GPU utilization on each device
- Distributed initialization messages from PyTorch Lightning
- Successful checkpoint generation

---

# Logging

Launch TensorBoard

```bash
tensorboard --logdir logs
```

---

# Checkpoints

Model checkpoints are automatically stored in

```text
checkpoints/
```

Example

```text
checkpoints/
└── lora_cfg_50000/
    ├── best-epoch=01.ckpt
    ├── best-epoch=02.ckpt
    └── last.ckpt
```

---

# Classifier-Free Guidance (CFG)

During training, molecular property conditioning is randomly removed with probability

```yaml
dropout_prob: 0.2
```

This enables the model to learn both conditional and unconditional molecular generation within a single model.

---

# Training Pipeline

```text
Dataset
    │
    ▼
Random Property Dropout (CFG)
    │
    ▼
Tokenizer
    │
    ▼
OLMo-7B (4-bit Quantization)
    │
    ▼
LoRA Adapters
    │
    ▼
Loss Computation
    │
    ▼
Backpropagation
    │
    ▼
Checkpoint Saving
```

---

# Frameworks

- PyTorch
- PyTorch Lightning
- Hugging Face Transformers
- PEFT
- BitsAndBytes
- Accelerate

---

# Future Work

- Dynamic CFG scheduling
- Property-guided decoding
- Additional molecular descriptors
- Larger-scale GuacaMol training
- Reinforcement learning fine-tuning

---

# Acknowledgements

- Allen Institute for AI (OLMo)
- Hugging Face
- PyTorch Lightning
- PEFT
- BitsAndBytes
