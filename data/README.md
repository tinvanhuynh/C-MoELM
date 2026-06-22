# Dataset Preparation

C-MoELM is evaluated on three Vietnamese fact-checking benchmarks. The datasets are not redistributed here. Please download them from their official sources or contact the original authors.

---

## Datasets

### ViFactCheck
- **Paper:** [ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnamese]([https://aclanthology.org/2023.findings-acl.27/](https://ojs.aaai.org/index.php/AAAI/article/view/32008))
- **GitHub:** [https://github.com/ndthanggit/ViFactCheck](https://github.com/TTHHA/ViFactCheck)

### ViWikiFC
- **Paper:** [ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Source]([https://aclanthology.org/2024.lrec-main.726/](https://arxiv.org/abs/2405.07615))
- **GitHub:** https://github.com/drunkard72/ViWikiFC

### ViNumFCR
- **Paper:** [ViNumFCR: A Dataset for Numerical Reasoning-based Fact-Checking in Vietnamese]([https://aclanthology.org/2025.naacl-long.364/](https://aclanthology.org/2025.inlg-main.9/))

---

## Expected Directory Structure

After downloading, organize the data as follows:

```
data/
├── vifactcheck/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── viwikifc/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── vinumfcr/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

---

## Data Format

Each `.jsonl` file contains one sample per line. Each sample is expected to have the following fields:

```json
{
  "context": "evidence text",
  "claim": "claim text",
  "label": "supported" | "refuted" | "not_enough_information"
}
```

> Note: Field names may vary slightly across datasets. Please refer to the original dataset documentation and adjust the data loading code in `src/transfer_fact_checking.py` accordingly.
