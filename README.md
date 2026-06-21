# C-MoELM: Contrastive Mixture of Experts Language Model with Negative Sample Learning for Fact-Checking

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-C--MoELM-blue)](https://huggingface.co/huynhtin/C-MoELM)
[![Paper](https://img.shields.io/badge/📄%20Paper-Coming%20Soon-lightgrey)](#)

**C-MoELM** is a parameter-efficient sentence representation model for Natural Language Inference (NLI) and Fact-Checking. It integrates a dynamic Top-*k* Mixture-of-Experts (MoE) routing mechanism into Quantized Low-Rank Adaptation (QLoRA), trained with a Fusion Negative Sample Learning strategy that combines semantic hard-negative mining, synthetic negative generation, and a weak-positive formulation for neutral pairs. These signals are jointly optimized with supervised NLI classification, domain adversarial alignment, and load-balancing regularization within a unified multi-objective training framework.

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

## Pre-trained Model

The pre-trained C-MoELM model is publicly available on HuggingFace:

👉 **[huggingface.co/huynhtin/C-MoELM](https://huggingface.co/huynhtin/C-MoELM)**

---

## Source Code

> ⚠️ Source code will be released upon paper acceptance.

---
