# C-MoELM: Contrastive Mixture of Experts Language Model with Negative Sample Learning for Fact-Checking

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-C--MoELM-blue)](https://huggingface.co/huynhtin/C-MoELM)
[![Paper](https://img.shields.io/badge/📄%20Paper-Coming%20Soon-lightgrey)](#)

**C-MoELM** is a parameter-efficient sentence representation model for Natural Language Inference (NLI) and Fact-Checking. It integrates a dynamic Top-*k* Mixture-of-Experts (MoE) routing mechanism into Quantized Low-Rank Adaptation (QLoRA), trained with a Fusion Negative Sample Learning strategy that combines semantic hard-negative mining, synthetic negative generation, and a weak-positive formulation for neutral pairs. These signals are jointly optimized with supervised NLI classification, domain adversarial alignment, and load-balancing regularization within a unified multi-objective training framework.

> ⚠️ Transfer task fine-tuning code is available in `src/`.

---

## Repository Structure

```
C-MoELM/
├── README.md
├── src/
│   ├── CMoELoRA_model.py
│   ├── transfer_fact_checking.py    # Transfer task fine-tuning
│   └── requirements.txt
└── data/
    └── README.md                    # Dataset download instructions
```

---

## Performance

C-MoELM outperforms 19 baseline models across all evaluation settings on three fact-checking benchmarks (Macro F1-Score, test set).

| Model | ViNumFCR | ViFactCheck | ViWikiFC | Avg. F1 |
|-------|:--------:|:-----------:|:--------:|:-------:|
| **Fine-tuning PLM** | | | | |
| mBERT | 83.43 | 69.94 | 76.01 | 76.46 |
| XLM-R Large | 90.06 | 88.02 | 85.15 | 87.74 |
| CafeBERT | 89.35 | 87.45 | 85.24 | 87.35 |
| PhoBERT Large | 87.56 | 79.76 | 81.62 | 82.98 |
| ViCLSR | 90.04 | 88.78 | 86.57 | 88.46 |
| InfoXLM | 85.00 | 83.27 | 86.51 | 84.93 |
| NLIMoE | 88.67 | 87.37 | 84.96 | 87.00 |
| **Fine-tuning LLM** | | | | |
| Mistral 7B | 71.57 | 88.63 | 85.36 | 81.85 |
| Llama-3 8B | 92.73 | 88.67 | 89.19 | 90.20 |
| Qwen3-4B | 92.44 | 90.31 | 88.96 | 90.57 |
| Phi-mini-MoE | 60.60 | 70.63 | 74.88 | 68.70 |
| LLaMA-MoE-v2 | 81.29 | 76.68 | 75.66 | 77.88 |
| Qwen1.5-MoE-A2.7B | 89.45 | 86.66 | 85.52 | 87.21 |
| **Prompting LLM** | | | | |
| Mistral 7B | 44.90 | 57.31 | 47.51 | 49.91 |
| Llama-3 8B | 44.69 | 63.10 | 53.07 | 53.62 |
| Qwen3-8B | 36.85 | 76.82 | 75.83 | 63.17 |
| Mixtral 8x7B | 43.01 | 59.81 | 62.77 | 55.20 |
| Llama-4-Maverick | 36.81 | 72.25 | 77.53 | 62.20 |
| Qwen3-30B-A3B | 36.44 | 71.04 | 72.96 | 60.15 |
| **C-MoELM (Ours)** | **93.53** | **91.75** | **90.08** | **91.79** |

---

## Installation

```bash
pip install -r src/requirements.txt
```

> **Requirements:** Python 3.9+, CUDA 11.8+, GPU with at least 24GB VRAM recommended.

---

## Transfer Task Fine-tuning

### 1. Download the pre-trained model

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="huynhtin/C-MoELM", local_dir="./cmoe_checkpoint")
```

### 2. Prepare the dataset

Please refer to [`data/README.md`](data/README.md) for dataset download instructions and expected directory structure.

### 3. Run transfer fine-tuning

```bash
python src/transfer_fact_checking.py \
  --src_dir src/ \
  --cmoe_ckpt_dir ./cmoe_checkpoint \
  --train_file data/vifactcheck/train.jsonl \
  --dev_file data/vifactcheck/dev.jsonl \
  --test_file data/vifactcheck/test.jsonl \
  --output_dir ./transfer_runs/vifactcheck \
  --max_len 256 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --epochs 7 \
  --grad_accum 1 \
  --lr 5e-5 \
  --pooling last2_mean \
  --sep_token "[SEP]" \
  --input_mode concat \
  --tuning_mode light \
  --clf_hidden_mult 1 \
  --clf_num_layers 2 \
  --clf_use_layernorm \
  --w_fc_con 0.0 \
  --w_lb 0.01 \
  --log_every 100 \
  --save_every_epoch
```

> To evaluate a previously trained checkpoint without re-training, add `--eval_from_epoch <epoch_number>` to load and evaluate the saved model at that epoch.

### 4. Adapting to other datasets

| Flag | Description | Default |
|------|-------------|---------|
| `--train_file` | Path to training data | — |
| `--dev_file` | Path to validation data | — |
| `--test_file` | Path to test data | — |
| `--output_dir` | Output directory | — |
| `--max_len` | Max sequence length | `256` |
| `--batch_size` | Training batch size | `16` |
| `--lr` | Learning rate | `5e-5` |
| `--epochs` | Max training epochs | `7` |
| `--input_mode` | Input format: `concat` or `separate` | `concat` |

The following flags can be adjusted based on your dataset size and task complexity:

| Flag | Description | Default |
|------|-------------|---------|
| `--clf_hidden_mult` | Hidden size multiplier for classification head | `1` |
| `--clf_num_layers` | Number of layers in classification head | `2` |
| `--w_fc_con` | Weight for contrastive loss during transfer | `0.0` |
| `--w_lb` | Weight for load balancing loss | `0.01` |

> For smaller datasets, reducing `--clf_num_layers` or `--clf_hidden_mult` may help prevent overfitting.

---

## Pre-trained Model

The pre-trained C-MoELM model is publicly available on HuggingFace:

👉 **[huggingface.co/huynhtin/C-MoELM](https://huggingface.co/huynhtin/C-MoELM)**

---
