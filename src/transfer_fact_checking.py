import os
import sys
import json
import math
import time
import csv
import random
import argparse
from dataclasses import fields
from functools import partial
from itertools import cycle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from safetensors.torch import load_file


# ============================================================
# 0) IMPORT CMoE Ver5
# ============================================================

def import_ver5(src_dir: str):
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from CMoELoRA_model import (
        CMoELoRA,
        CMoELoRAConfig,
        set_gates,
        apply_pooling_from_outputs,
        load_jsonl,
        SemanticMinerViCLSR,
        build_same_premise_hardneg_semantic,
        load_processed_dataset,
        NLITripletDataset,
        collate_nli_triplet_dynamic,
        training_step as ver5_nli_training_step,
    )

    return {
        "CMoELoRA": CMoELoRA,
        "CMoELoRAConfig": CMoELoRAConfig,
        "set_gates": set_gates,
        "apply_pooling_from_outputs": apply_pooling_from_outputs,
        "load_jsonl": load_jsonl,
        "SemanticMinerViCLSR": SemanticMinerViCLSR,
        "build_same_premise_hardneg_semantic": build_same_premise_hardneg_semantic,
        "load_processed_dataset": load_processed_dataset,
        "NLITripletDataset": NLITripletDataset,
        "collate_nli_triplet_dynamic": collate_nli_triplet_dynamic,
        "ver5_nli_training_step": ver5_nli_training_step,
    }


# ============================================================
# 1) UTILS
# ============================================================

def seed_all(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return total, trainable, pct


def should_skip_key(k: str) -> bool:
    return (
        ".base_layer.weight.absmax" in k
        or ".base_layer.weight.quant_map" in k
        or ".base_layer.weight.nested_absmax" in k
        or ".base_layer.weight.nested_quant_map" in k
        or ".base_layer.weight.quant_state." in k
    )


def clone_state_dict_to_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not should_skip_key(k)
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_downstream_best_state_safely(model, ckpt_path, tag="best"):
    if not os.path.exists(ckpt_path):
        print(f"[LOAD {tag}] checkpoint not found: {ckpt_path}")
        return False

    print(f"[LOAD {tag}] loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if not should_skip_key(k)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD {tag}] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[LOAD {tag}] first missing:", missing[:10])
    if len(unexpected) > 0:
        print(f"[LOAD {tag}] first unexpected:", unexpected[:10])

    return True


def load_cmoe_checkpoint_qlora_safe(ckpt_dir, CMoELoRA, CMoELoRAConfig, device=None):
    cfg_path = os.path.join(ckpt_dir, "cmoe_config.json")
    weights_path = os.path.join(ckpt_dir, "model.safetensors")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    valid_keys = {f.name for f in fields(CMoELoRAConfig)}
    extra_keys = sorted(set(cfg_dict.keys()) - valid_keys)
    if len(extra_keys) > 0:
        print("[LOAD] Ignoring extra config keys:", extra_keys)

    cfg_core = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    cfg = CMoELoRAConfig(**cfg_core)

    model = CMoELoRA(cfg)
    state = load_file(weights_path)
    filtered_state = {k: v for k, v in state.items() if not should_skip_key(k)}

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print("[LOAD] Loaded CMoE checkpoint with strict=False")
    print("[LOAD] Missing keys   :", len(missing))
    print("[LOAD] Unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("[LOAD] First 10 missing   :", missing[:10])
    if len(unexpected) > 0:
        print("[LOAD] First 10 unexpected:", unexpected[:10])

    if device is not None and not cfg.use_qlora:
        model = model.to(device)

    model.eval()
    return model


# ============================================================
# 2) LABELS / DATASET
# ============================================================

LABEL2ID = {
    "e": 0,  # supported
    "c": 1,  # refuted
    "n": 2,  # nei
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
ID2NAME = {
    0: "supported",
    1: "refuted",
    2: "nei",
}


class FactCheckDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128, sep_token="[SEP]"):
        self.sep_token = sep_token
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                ex = json.loads(line)
                lab = str(ex["label"]).strip()
                if lab not in LABEL2ID:
                    continue

                rows.append({
                    "uid": ex.get("uid", f"{os.path.basename(jsonl_path)}::{idx}"),
                    "premise": str(ex["premise"]),
                    "hypothesis": str(ex["hypothesis"]),
                    "label": LABEL2ID[lab],
                    "label_str": lab,
                })

        self.data = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        text = f"{ex['premise']} {self.sep_token} {ex['hypothesis']}"

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Separate encodings are always returned so the same dataset can be used
        # for both input_mode=concat and input_mode=pair_feature.
        p_enc = self.tokenizer(
            ex["premise"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        h_enc = self.tokenizer(
            ex["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "uid": ex["uid"],
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "input_text": text,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "p_input_ids": p_enc["input_ids"].squeeze(0),
            "p_attention_mask": p_enc["attention_mask"].squeeze(0),
            "h_input_ids": h_enc["input_ids"].squeeze(0),
            "h_attention_mask": h_enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
            "label_str": ex["label_str"],
        }


# ============================================================
# 3) LOSSES / ROUTER STATS
# ============================================================

def supervised_contrastive_loss(emb, labels, temperature=0.07):
    if emb.size(0) <= 1:
        return emb.new_tensor(0.0)

    emb = F.normalize(emb, dim=-1)
    labels = labels.view(-1, 1)

    sim = emb @ emb.t()
    sim = sim / max(float(temperature), 1e-6)

    eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
    logits_mask = 1.0 - eye
    positive_mask = (labels == labels.t()).to(dtype=sim.dtype) * logits_mask

    num_pos = positive_mask.sum(dim=1)
    valid = num_pos > 0
    if valid.sum().item() == 0:
        return emb.new_tensor(0.0)

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    mean_log_pos = (positive_mask * log_prob).sum(dim=1) / num_pos.clamp_min(1.0)
    return -mean_log_pos[valid].mean()


def load_balance_single(gates):
    B, E = gates.shape
    importance = gates.mean(dim=0)
    top1 = torch.argmax(gates, dim=-1)
    load = torch.bincount(top1, minlength=E).float().to(gates.device) / float(B)
    loss = E * torch.sum(importance * load) - 1.0
    return loss, {
        "importance": importance.detach(),
        "load": load.detach(),
        "top1": top1.detach(),
    }


def router_stats(gates):
    gates = gates.detach().float()
    entropy = (-(gates * torch.log(gates + 1e-12)).sum(dim=-1)).mean()
    max_prob = gates.max(dim=-1).values.mean()
    mean = gates.mean(dim=0)
    entropy_of_mean = (-(mean * torch.log(mean + 1e-12)).sum())
    topv = torch.topk(gates, k=min(2, gates.size(-1)), dim=-1).values
    if topv.size(-1) >= 2:
        margin = (topv[:, 0] - topv[:, 1]).mean()
    else:
        margin = max_prob
    return {
        "entropy": entropy,
        "max_prob": max_prob,
        "mean": mean,
        "entropy_of_mean": entropy_of_mean,
        "margin": margin,
    }


# ============================================================
# 4) TARGET-ADAPTIVE MODEL WRAPPER
# ============================================================

class TargetAdaptiveCMoEForFactChecking(nn.Module):
    def __init__(
        self,
        cmoe,
        set_gates_fn,
        apply_pooling_fn,
        num_labels=3,
        clf_dropout=0.1,
        clf_hidden_mult=0.5,
        clf_num_layers=3,
        clf_use_layernorm=False,
        use_target_router_conditioning=True,
        num_target_domains=1,
        input_mode="concat",
        pair_contrastive_source="avg",
    ):
        super().__init__()
        self.cmoe = cmoe
        self.set_gates_fn = set_gates_fn
        self.apply_pooling_fn = apply_pooling_fn
        self.use_target_router_conditioning = bool(use_target_router_conditioning)

        if input_mode not in ["concat", "pair_feature"]:
            raise ValueError("input_mode must be either 'concat' or 'pair_feature'")
        if pair_contrastive_source not in ["avg", "evidence", "claim", "diff"]:
            raise ValueError("pair_contrastive_source must be one of: avg, evidence, claim, diff")

        self.input_mode = input_mode
        self.use_pair_feature_classifier = input_mode == "pair_feature"
        self.pair_contrastive_source = pair_contrastive_source

        hidden_size = cmoe.hidden_size
        emb_dim = cmoe.cfg.emb_dim or hidden_size
        hidden_dim = max(256, int(emb_dim * clf_hidden_mult))

        self.target_dataset_embed = nn.Embedding(num_target_domains, hidden_size)

        self.dropout = nn.Dropout(clf_dropout)
        layers = []
        in_dim = emb_dim * 4 if self.use_pair_feature_classifier else emb_dim

        for _ in range(max(1, clf_num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if clf_use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(clf_dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_labels))
        self.classifier = nn.Sequential(*layers)

        backbone_device = self.cmoe.lm.model.embed_tokens.weight.device
        model_dtype = next(self.cmoe.router.parameters()).dtype

        self.target_dataset_embed = self.target_dataset_embed.to(device=backbone_device, dtype=model_dtype)
        self.classifier = self.classifier.to(device=backbone_device, dtype=model_dtype)

    @torch.no_grad()
    def _pooled_no_lora(self, input_ids, attention_mask):
        self.set_gates_fn(None)
        need_hidden_states = self.cmoe.cfg.pooling in ["last2_mean", "first_last_mean"]
        out = self.cmoe.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        return self.apply_pooling_fn(out, attention_mask, self.cmoe.cfg.pooling)

    def encode_target_conditioned(self, input_ids, attention_mask, pooling=None, target_domain_id=0):
        pooling = pooling or self.cmoe.cfg.pooling

        if not self.use_target_router_conditioning:
            return self.cmoe.encode_with_pooling(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling=pooling,
            )

        pooled0 = self._pooled_no_lora(input_ids, attention_mask)

        bsz = input_ids.size(0)
        target_ids = torch.full(
            (bsz,),
            int(target_domain_id),
            dtype=torch.long,
            device=pooled0.device,
        )
        target_vec = self.target_dataset_embed(target_ids).to(dtype=pooled0.dtype, device=pooled0.device)

        router_input = pooled0 + target_vec
        gates = self.cmoe.router(router_input)

        need_hidden_states = pooling in ["last2_mean", "first_last_mean"]
        self.set_gates_fn(gates)
        out = self.cmoe.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        self.set_gates_fn(None)

        pooled = self.apply_pooling_fn(out, attention_mask, pooling)

        if self.cmoe.proj is not None:
            proj_dtype = next(self.cmoe.proj.parameters()).dtype
            pooled = pooled.to(dtype=proj_dtype)
            pooled = self.cmoe.proj(pooled)

        emb = F.normalize(pooled, dim=-1)
        return emb, gates

    def forward_pair(
        self,
        p_input_ids,
        p_attention_mask,
        h_input_ids,
        h_attention_mask,
        labels=None,
        pooling=None,
        target_domain_id=0,
    ):
        p_emb, p_gates = self.encode_target_conditioned(
            input_ids=p_input_ids,
            attention_mask=p_attention_mask,
            pooling=pooling,
            target_domain_id=target_domain_id,
        )
        h_emb, h_gates = self.encode_target_conditioned(
            input_ids=h_input_ids,
            attention_mask=h_attention_mask,
            pooling=pooling,
            target_domain_id=target_domain_id,
        )

        pair_feat = torch.cat(
            [p_emb, h_emb, torch.abs(p_emb - h_emb), p_emb * h_emb],
            dim=-1,
        )

        target_dtype = next(self.classifier.parameters()).dtype
        target_device = next(self.classifier.parameters()).device
        pair_feat = pair_feat.to(dtype=target_dtype, device=target_device)

        logits = self.classifier(self.dropout(pair_feat))

        loss_ce = None
        if labels is not None:
            loss_ce = F.cross_entropy(logits.float(), labels)

        if self.pair_contrastive_source == "avg":
            emb_for_contrastive = F.normalize(0.5 * (p_emb + h_emb), dim=-1)
        elif self.pair_contrastive_source == "evidence":
            emb_for_contrastive = p_emb
        elif self.pair_contrastive_source == "claim":
            emb_for_contrastive = h_emb
        elif self.pair_contrastive_source == "diff":
            emb_for_contrastive = F.normalize(torch.abs(p_emb - h_emb), dim=-1)
        else:
            raise ValueError(f"Unknown pair_contrastive_source={self.pair_contrastive_source}")

        gates = 0.5 * (p_gates + h_gates)

        return {
            "loss_ce": loss_ce,
            "logits": logits,
            "emb": emb_for_contrastive,
            "gates": gates,
        }

    def forward(self, input_ids, attention_mask, labels=None, pooling=None, target_domain_id=0):
        emb, gates = self.encode_target_conditioned(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling=pooling,
            target_domain_id=target_domain_id,
        )

        target_dtype = next(self.classifier.parameters()).dtype
        target_device = next(self.classifier.parameters()).device
        emb_head = emb.to(dtype=target_dtype, device=target_device)

        logits = self.classifier(self.dropout(emb_head))

        loss_ce = None
        if labels is not None:
            loss_ce = F.cross_entropy(logits.float(), labels)

        return {
            "loss_ce": loss_ce,
            "logits": logits,
            "emb": emb,
            "gates": gates,
        }


# ============================================================
# 5) FREEZE / UNFREEZE
# ============================================================

def freeze_all_cmoe_except_new_modules(model):
    for p in model.cmoe.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.target_dataset_embed.parameters():
        p.requires_grad = True


def set_probe_mode(model):
    freeze_all_cmoe_except_new_modules(model)


def set_light_mode(model):
    """
    light:
    - classifier train
    - target dataset embedding train
    - router train
    - proj train
    - LoRA A/B train
    - optional NLI head/logit_scale/domain head train for replay if present
    """
    freeze_all_cmoe_except_new_modules(model)

    if hasattr(model.cmoe, "router") and model.cmoe.router is not None:
        for p in model.cmoe.router.parameters():
            p.requires_grad = True

    if hasattr(model.cmoe, "proj") and model.cmoe.proj is not None:
        for p in model.cmoe.proj.parameters():
            p.requires_grad = True

    if hasattr(model.cmoe, "nli_head") and model.cmoe.nli_head is not None:
        for p in model.cmoe.nli_head.parameters():
            p.requires_grad = True

    if hasattr(model.cmoe, "domain_head") and model.cmoe.domain_head is not None:
        for p in model.cmoe.domain_head.parameters():
            p.requires_grad = True

    if hasattr(model.cmoe, "logit_scale"):
        model.cmoe.logit_scale.requires_grad = True

    for name, p in model.cmoe.lm.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            p.requires_grad = True


def set_router_only_mode(model):
    freeze_all_cmoe_except_new_modules(model)

    if hasattr(model.cmoe, "router") and model.cmoe.router is not None:
        for p in model.cmoe.router.parameters():
            p.requires_grad = True

    if hasattr(model.cmoe, "proj") and model.cmoe.proj is not None:
        for p in model.cmoe.proj.parameters():
            p.requires_grad = True


def set_topn_mode(model, n):
    set_light_mode(model)

    if n is None or n <= 0:
        print("[INFO] num_unfreeze_layers <= 0, topn behaves like light")
        return

    if hasattr(model.cmoe.lm.model, "layers"):
        total_layers = len(model.cmoe.lm.model.layers)
        n = min(n, total_layers)
        start_idx = max(0, total_layers - n)

        print(f"[INFO] Unfreezing top {n}/{total_layers} backbone layers for transfer")

        for i in range(start_idx, total_layers):
            for _, p in model.cmoe.lm.model.layers[i].named_parameters():
                if p.is_floating_point():
                    p.requires_grad = True

    if hasattr(model.cmoe.lm.model, "norm") and model.cmoe.lm.model.norm is not None:
        for p in model.cmoe.lm.model.norm.parameters():
            if p.is_floating_point():
                p.requires_grad = True


# ============================================================
# 6) NLI REPLAY DATA
# ============================================================

def build_or_load_nli_replay(args, deps, tokenizer):
    if not args.use_nli_replay:
        return None, None

    if not args.vinli_dir or not args.vianli_dir:
        raise ValueError("--use_nli_replay requires --vinli_dir and --vianli_dir")

    load_jsonl = deps["load_jsonl"]
    SemanticMinerViCLSR = deps["SemanticMinerViCLSR"]
    build_same_premise_hardneg_semantic = deps["build_same_premise_hardneg_semantic"]
    load_processed_dataset = deps["load_processed_dataset"]
    NLITripletDataset = deps["NLITripletDataset"]

    vinli_train = load_jsonl(os.path.join(args.vinli_dir, "train.jsonl"), domain_id=0)
    vianli_train = load_jsonl(os.path.join(args.vianli_dir, "train.jsonl"), domain_id=1)
    raw_train = vinli_train + vianli_train

    if len(raw_train) == 0:
        raise ValueError("No NLI replay data loaded.")

    cache_root = args.mined_cache_dir.strip() or os.path.join(args.output_dir, "nli_replay_cache")
    mkdir(cache_root)

    mined_path = os.path.join(
        cache_root,
        f"nli_replay_{args.neutral_mode}_k{args.num_hard_negs}_supnli_seed{args.seed}_viclsr.jsonl",
    )

    if args.reuse_mined_dataset and os.path.exists(mined_path):
        replay_data = load_processed_dataset(mined_path)
    else:
        miner_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[ViCLSR] Loading miner for NLI replay: {args.viclsr_model} on {miner_device}")
        miner = SemanticMinerViCLSR(
            model_name=args.viclsr_model,
            device=miner_device,
            max_len=args.viclsr_max_len,
            batch_size=args.viclsr_batch_size,
        )

        print(f"[ViCLSR] Mining NLI replay hard negatives | neutral_mode={args.neutral_mode} | k={args.num_hard_negs}")
        replay_data = build_same_premise_hardneg_semantic(
            raw_train,
            miner,
            seed=args.seed,
            neutral_mode=args.neutral_mode,
            num_hard_negs=args.num_hard_negs,
            keep_full_nli_for_sup=True,
        )

        with open(mined_path, "w", encoding="utf-8") as f:
            for ex in replay_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"[ViCLSR] Saved replay mined dataset -> {mined_path}")

    replay_ds = NLITripletDataset(replay_data, tokenizer, max_len=args.max_len)
    replay_stats = {
        "vinli_train_raw": len(vinli_train),
        "vianli_train_raw": len(vianli_train),
        "nli_replay_total": len(replay_data),
        "mined_path": mined_path,
        "neutral_mode": args.neutral_mode,
        "num_hard_negs": args.num_hard_negs,
    }
    return replay_ds, replay_stats


# ============================================================
# 7) EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, pooling="mean"):
    model.eval()

    all_uids, all_prem, all_hyp, all_texts = [], [], [], []
    all_labels, all_preds = [], []
    all_gate_means = []
    all_gate_rows = []
    all_logit_rows = []

    num_experts = int(getattr(model.cmoe.cfg, "num_experts", 0))
    top_k = int(getattr(model.cmoe.cfg, "top_k", min(3, max(1, num_experts))))

    for batch in dataloader:
        backbone_device = model.cmoe.lm.model.embed_tokens.weight.device

        input_ids = batch["input_ids"].to(backbone_device)
        attention_mask = batch["attention_mask"].to(backbone_device)
        labels = batch["labels"].to(backbone_device)

        use_amp = backbone_device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            if getattr(model, "use_pair_feature_classifier", False):
                out = model.forward_pair(
                    p_input_ids=batch["p_input_ids"].to(backbone_device),
                    p_attention_mask=batch["p_attention_mask"].to(backbone_device),
                    h_input_ids=batch["h_input_ids"].to(backbone_device),
                    h_attention_mask=batch["h_attention_mask"].to(backbone_device),
                    labels=None,
                    pooling=pooling,
                )
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    pooling=pooling,
                )

        logits = out["logits"].float()
        preds = logits.argmax(dim=-1)

        all_uids.extend(batch["uid"])
        all_prem.extend(batch["premise"])
        all_hyp.extend(batch["hypothesis"])
        all_texts.extend(batch["input_text"])
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_logit_rows.extend(logits.detach().cpu().tolist())

        if out["gates"] is not None:
            gates_cpu = out["gates"].detach().float().cpu()
            all_gate_means.append(gates_cpu.mean(dim=0))
            all_gate_rows.extend(gates_cpu.tolist())
        else:
            all_gate_rows.extend([None] * labels.size(0))

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    gate_mean = None
    if len(all_gate_means) > 0:
        gate_mean = torch.stack(all_gate_means, dim=0).mean(dim=0)

    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=[ID2NAME[i] for i in range(3)],
        digits=4,
        zero_division=0,
    )

    rows = []
    for idx, (uid, prem, hyp, txt, gold, pred) in enumerate(zip(
        all_uids, all_prem, all_hyp, all_texts, all_labels, all_preds
    )):
        row = {
            "uid": uid,
            "premise": prem,
            "hypothesis": hyp,
            "input_text": txt,
            "gold_id": gold,
            "pred_id": pred,
            "gold_label": ID2LABEL[gold],
            "pred_label": ID2LABEL[pred],
            "gold_name": ID2NAME[gold],
            "pred_name": ID2NAME[pred],
            "correct": int(gold == pred),
        }

        if idx < len(all_logit_rows):
            logits_i = all_logit_rows[idx]
            for c in range(len(logits_i)):
                row[f"logit_{ID2NAME.get(c, c)}"] = float(logits_i[c])

        gates_i = all_gate_rows[idx] if idx < len(all_gate_rows) else None
        if gates_i is not None:
            for e, prob in enumerate(gates_i):
                row[f"gate_expert_{e}"] = float(prob)

            order = sorted(range(len(gates_i)), key=lambda e: gates_i[e], reverse=True)
            k_eff = min(top_k, len(order))
            activated = []
            activated_with_prob = []
            for rank in range(k_eff):
                e = order[rank]
                p = float(gates_i[e])
                row[f"expert_rank{rank+1}_id"] = int(e)
                row[f"expert_rank{rank+1}_prob"] = p
                activated.append(str(e))
                activated_with_prob.append(f"{e}:{p:.6f}")

            row["activated_experts_topk"] = "|".join(activated)
            row["activated_experts_topk_with_prob"] = "|".join(activated_with_prob)
            row["top1_expert_id"] = int(order[0]) if len(order) else None
            row["top1_expert_prob"] = float(gates_i[order[0]]) if len(order) else None

            gates_tensor = torch.tensor(gates_i)
            row["gate_entropy"] = float(-(gates_tensor * torch.log(gates_tensor + 1e-12)).sum().item())
            row["gate_max_prob"] = float(max(gates_i))
        else:
            row["activated_experts_topk"] = ""
            row["activated_experts_topk_with_prob"] = ""

        rows.append(row)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "gate_mean": gate_mean,
        "pred_rows": rows,
    }


def save_expert_activation_csv(pred_rows, output_dir, split, tag):
    """
    Save a compact per-sample expert activation file for later analysis.
    """
    df = pd.DataFrame(pred_rows)

    if len(pred_rows) > 0:
        expert_cols = [c for c in df.columns if c.startswith("gate_expert_")]
        rank_cols = [
            c for c in df.columns
            if c.startswith("expert_rank")
            or c in [
                "activated_experts_topk",
                "activated_experts_topk_with_prob",
                "top1_expert_id",
                "top1_expert_prob",
                "gate_entropy",
                "gate_max_prob",
            ]
        ]
        logit_cols = [c for c in df.columns if c.startswith("logit_")]
    else:
        expert_cols, rank_cols, logit_cols = [], [], []

    base_cols = [
        "uid",
        "gold_id", "gold_label", "gold_name",
        "pred_id", "pred_label", "pred_name",
        "correct",
    ]
    text_cols = ["premise", "hypothesis", "input_text"]

    ordered_cols = []
    for c in base_cols + logit_cols + expert_cols + rank_cols + text_cols:
        if c in df.columns:
            ordered_cols.append(c)

    if ordered_cols:
        remaining = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + remaining]

    out_path = os.path.join(output_dir, f"{split}_expert_activations_{tag}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Expert activation CSV -> {out_path}")


# ============================================================
# 8) TRAIN
# ============================================================

def main(args):
    seed_all(args.seed)

    if args.eval_only:
        if not os.path.isdir(args.output_dir):
            raise ValueError(
                f"--eval_only requires an existing --output_dir with saved checkpoints, got: {args.output_dir}"
            )
        print(f"[MODE] eval_only=True -> read-only evaluation from: {args.output_dir}")
    else:
        mkdir(args.output_dir)

    deps = import_ver5(args.src_dir)
    CMoELoRA = deps["CMoELoRA"]
    CMoELoRAConfig = deps["CMoELoRAConfig"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not os.path.isdir(args.cmoe_ckpt_dir):
        raise ValueError(f"--cmoe_ckpt_dir must be a checkpoint directory, got: {args.cmoe_ckpt_dir}")

    required1 = os.path.join(args.cmoe_ckpt_dir, "cmoe_config.json")
    required2 = os.path.join(args.cmoe_ckpt_dir, "model.safetensors")
    if not os.path.exists(required1) or not os.path.exists(required2):
        raise ValueError(
            f"Checkpoint directory missing required files: {args.cmoe_ckpt_dir}\n"
            f"Need at least cmoe_config.json and model.safetensors"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.cmoe_ckpt_dir,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # v_epoch_save: skip training nếu --eval_only HOẶC --eval_from_epoch được set
    skip_training = args.eval_only or (len(args.eval_from_epoch) > 0)

    if not skip_training:
        tokenizer.save_pretrained(args.output_dir)
    else:
        print("[SKIP TRAIN] Skip saving tokenizer files")

    cmoe = load_cmoe_checkpoint_qlora_safe(
        args.cmoe_ckpt_dir,
        CMoELoRA=CMoELoRA,
        CMoELoRAConfig=CMoELoRAConfig,
        device=device,
    )

    if getattr(cmoe.cfg, "use_qlora", False) and args.tuning_mode == "topn":
        print("[WARN] You are using topn with a QLoRA checkpoint. This is hybrid transfer, not pure QLoRA.")

    print("\n===== LOADED CMOE CONFIG =====")
    print(json.dumps(cmoe.cfg.__dict__, ensure_ascii=False, indent=2))
    print("================================\n")

    train_ds = FactCheckDataset(args.train_file, tokenizer, max_len=args.max_len, sep_token=args.sep_token)
    dev_ds = FactCheckDataset(args.dev_file, tokenizer, max_len=args.max_len, sep_token=args.sep_token)
    test_ds = FactCheckDataset(args.test_file, tokenizer, max_len=args.max_len, sep_token=args.sep_token)

    print("Train size:", len(train_ds))
    print("Dev size  :", len(dev_ds))
    print("Test size :", len(test_ds))

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    replay_ds, replay_stats = build_or_load_nli_replay(args, deps, tokenizer)
    replay_dl = None
    if replay_ds is not None:
        replay_dl = DataLoader(
            replay_ds,
            batch_size=args.nli_replay_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(
                deps["collate_nli_triplet_dynamic"],
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        print("[NLI REPLAY]", json.dumps(replay_stats, ensure_ascii=False, indent=2))

    model = TargetAdaptiveCMoEForFactChecking(
        cmoe=cmoe,
        set_gates_fn=deps["set_gates"],
        apply_pooling_fn=deps["apply_pooling_from_outputs"],
        num_labels=3,
        clf_dropout=args.clf_dropout,
        clf_hidden_mult=args.clf_hidden_mult,
        clf_num_layers=args.clf_num_layers,
        clf_use_layernorm=args.clf_use_layernorm,
        use_target_router_conditioning=not args.no_target_router_conditioning,
        num_target_domains=1,
        input_mode=args.input_mode,
        pair_contrastive_source=args.pair_contrastive_source,
    )

    backbone_device = model.cmoe.lm.model.embed_tokens.weight.device
    print("Backbone device:", backbone_device)
    print("Classifier device:", next(model.classifier.parameters()).device)

    if args.tuning_mode == "probe":
        set_probe_mode(model)
    elif args.tuning_mode == "router_only":
        set_router_only_mode(model)
    elif args.tuning_mode == "light":
        set_light_mode(model)
    elif args.tuning_mode == "topn":
        set_topn_mode(model, n=args.num_unfreeze_layers)
    else:
        raise ValueError("tuning_mode must be one of: probe, router_only, light, topn")

    print(f"Tuning mode: {args.tuning_mode}")
    print(f"Input mode : {args.input_mode}")
    print(f"Sep token  : {args.sep_token}")
    if args.input_mode == "pair_feature":
        print(f"Pair contrastive source: {args.pair_contrastive_source}")
    if args.tuning_mode == "topn":
        print(f"Top-N unfreeze layers: {args.num_unfreeze_layers}")

    total_params, trainable_params, pct = count_parameters(model)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,} ({pct:.2f}%)")

    training_config = {
        "method": "Target-Adaptive CMoE-LoRA Transfer",
        "args": vars(args),
        "cmoe_cfg": dict(cmoe.cfg.__dict__),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "pct_trainable_params": pct,
        "train_size": len(train_ds),
        "dev_size": len(dev_ds),
        "test_size": len(test_ds),
        "nli_replay_stats": replay_stats,
        "input_mode": args.input_mode,
        "sep_token": args.sep_token,
        "pair_contrastive_source": args.pair_contrastive_source,
        "target_router_conditioning": not args.no_target_router_conditioning,
    }
    if not skip_training:
        save_json(os.path.join(args.output_dir, "training_config.json"), training_config)
    else:
        print("[SKIP TRAIN] Skip writing training_config.json")

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = math.ceil(len(train_dl) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1_path = os.path.join(args.output_dir, "best_macro_f1.pt")
    best_acc_path = os.path.join(args.output_dir, "best_acc.pt")
    best_loss_path = os.path.join(args.output_dir, "best_train_loss.pt")

    best_dev_macro_f1 = -1.0
    best_dev_acc = -1.0
    best_train_loss = float("inf")  # v_loss: track train loss

    best_state_f1 = None
    best_state_acc = None
    best_state_loss = None  # v_loss

    patience = 0

    csv_f = None
    writer = None
    if not skip_training:
        log_path = os.path.join(args.output_dir, "train_log.csv")
        csv_f = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)

        header = [
            "time_sec", "epoch", "step", "global_step", "opt_step", "lr",
            "loss_total",
            "loss_fc_total", "loss_fc_ce", "loss_fc_con", "loss_fc_lb",
            "loss_nli_replay_weighted", "loss_nli_replay_raw",
            "fc_batch_acc",
            "router_entropy", "router_max_prob", "router_entropy_of_mean", "router_margin",
            "router_temperature",
        ]
        for i in range(cmoe.cfg.num_experts):
            header.append(f"fc_gates_mean_{i}")
        for i in range(cmoe.cfg.num_experts):
            header.append(f"fc_gates_load_{i}")

        writer.writerow(header)
        csv_f.flush()
    else:
        print("[SKIP TRAIN] Skip writing train_log.csv")

    start_time = time.time()
    global_step = 0
    opt_step = 0
    ema_loss = None
    ema_alpha = 0.05

    if not skip_training:
        replay_iter = cycle(replay_dl) if replay_dl is not None else None

        for epoch in range(args.epochs):
            model.train()
            total_loss_epoch = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_dl):
                backbone_device = model.cmoe.lm.model.embed_tokens.weight.device

                input_ids = batch["input_ids"].to(backbone_device)
                attention_mask = batch["attention_mask"].to(backbone_device)
                labels = batch["labels"].to(backbone_device)

                use_amp = backbone_device.type == "cuda"

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    if getattr(model, "use_pair_feature_classifier", False):
                        out = model.forward_pair(
                            p_input_ids=batch["p_input_ids"].to(backbone_device),
                            p_attention_mask=batch["p_attention_mask"].to(backbone_device),
                            h_input_ids=batch["h_input_ids"].to(backbone_device),
                            h_attention_mask=batch["h_attention_mask"].to(backbone_device),
                            labels=labels,
                            pooling=args.pooling,
                        )
                    else:
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            pooling=args.pooling,
                        )

                    loss_fc_ce = out["loss_ce"]
                    loss_fc_con = supervised_contrastive_loss(
                        out["emb"],
                        labels,
                        temperature=args.supcon_temp,
                    )
                    loss_fc_lb, lb_stats = load_balance_single(out["gates"])
                    loss_fc_total = (
                        loss_fc_ce
                        + args.w_fc_con * loss_fc_con
                        + args.w_lb * loss_fc_lb
                    )

                    loss_nli_replay_raw = loss_fc_ce.new_tensor(0.0)
                    loss_nli_replay_weighted = loss_fc_ce.new_tensor(0.0)

                    if replay_iter is not None and args.w_nli_replay > 0:
                        replay_batch = next(replay_iter)
                        replay_batch = {
                            k: v.to(backbone_device)
                            for k, v in replay_batch.items()
                        }

                        nli_out = deps["ver5_nli_training_step"](
                            model.cmoe,
                            replay_batch,
                            neutral_mode=args.neutral_mode,
                            w_neutral_pos=args.w_neutral_pos,
                            w_adv=args.w_adv_replay,
                            w_lb=args.w_lb,
                            w_sup=args.w_nli_sup,
                            step=global_step,
                            total_steps=total_steps,
                            num_synth_negs=args.num_synth_negs,
                            mixup_alpha_min=args.mixup_alpha_min,
                            mixup_alpha_max=args.mixup_alpha_max,
                            hardneg_weight_temp=args.hardneg_weight_temp,
                            hardneg_weight_clamp=args.hardneg_weight_clamp,
                            use_synthetic_hardneg=args.use_synthetic_hardneg,
                            use_hardneg_weighting=args.use_hardneg_weighting,
                        )

                        loss_nli_replay_raw = nli_out["loss"]
                        loss_nli_replay_weighted = args.w_nli_replay * loss_nli_replay_raw

                    loss_total = loss_fc_total + loss_nli_replay_weighted

                loss = loss_total / args.grad_accum
                loss.backward()

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

                cur_loss = float(loss_total.detach().cpu().item())
                ema_loss = cur_loss if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * cur_loss
                total_loss_epoch += cur_loss

                preds = out["logits"].argmax(dim=-1)
                fc_batch_acc = (preds == labels).float().mean().detach()

                rs = router_stats(out["gates"])
                gates_mean = rs["mean"].detach().cpu().tolist()
                gates_load = lb_stats["load"].detach().float().cpu().tolist()
                lr = float(optimizer.param_groups[0]["lr"])
                router_temp = float(getattr(model.cmoe.router, "temperature", 0.0))

                if step % args.log_every == 0:
                    print(
                        f"epoch={epoch+1} step={step}/{len(train_dl)} "
                        f"loss={cur_loss:.4f} ema={ema_loss:.4f} "
                        f"fc_ce={loss_fc_ce.item():.4f} fc_con={loss_fc_con.item():.4f} "
                        f"fc_acc={fc_batch_acc.item():.4f} "
                        f"nli_replay={loss_nli_replay_raw.item():.4f} "
                        f"gmax={rs['max_prob'].item():.3f} gent={rs['entropy'].item():.3f}"
                    )

                row = [
                    round(time.time() - start_time, 3),
                    epoch + 1,
                    step,
                    global_step,
                    opt_step,
                    lr,
                    cur_loss,
                    float(loss_fc_total.detach().cpu().item()),
                    float(loss_fc_ce.detach().cpu().item()),
                    float(loss_fc_con.detach().cpu().item()),
                    float(loss_fc_lb.detach().cpu().item()),
                    float(loss_nli_replay_weighted.detach().cpu().item()),
                    float(loss_nli_replay_raw.detach().cpu().item()),
                    float(fc_batch_acc.detach().cpu().item()),
                    float(rs["entropy"].detach().cpu().item()),
                    float(rs["max_prob"].detach().cpu().item()),
                    float(rs["entropy_of_mean"].detach().cpu().item()),
                    float(rs["margin"].detach().cpu().item()),
                    router_temp,
                ] + [float(x) for x in gates_mean] + [float(x) for x in gates_load]

                writer.writerow(row)
                if step % args.log_every == 0:
                    csv_f.flush()

                global_step += 1

            if len(train_dl) % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            train_loss = total_loss_epoch / max(1, len(train_dl))

            dev_res = evaluate(
                model=model,
                dataloader=dev_dl,
                pooling=args.pooling,
            )

            print("\n" + "=" * 80)
            print(f"Epoch {epoch+1}")
            print(f"train_loss    : {train_loss:.4f}")
            print(f"dev_acc       : {dev_res['acc']:.4f}")
            print(f"dev_macroF1   : {dev_res['macro_f1']:.4f}")
            print(f"dev_weightF1  : {dev_res['weighted_f1']:.4f}")
            if dev_res["gate_mean"] is not None:
                print("dev_gate_mean :", [round(x, 4) for x in dev_res["gate_mean"].tolist()])
            print("=" * 80)

            writer.writerow([
                round(time.time() - start_time, 3),
                epoch + 1,
                "EPOCH_END",
                global_step,
                opt_step,
                float(optimizer.param_groups[0]["lr"]),
                train_loss,
                "", "", "", "", "", "", "", "", "", "", "", "",
            ] + [""] * (cmoe.cfg.num_experts * 2))
            csv_f.flush()

            improved_main_metric = False

            if args.save_best_by in ["f1", "both"]:
                if dev_res["macro_f1"] > best_dev_macro_f1:
                    best_dev_macro_f1 = dev_res["macro_f1"]
                    best_state_f1 = clone_state_dict_to_cpu(model)
                    torch.save(best_state_f1, best_f1_path)
                    print(f"[SAVE] New best macro-F1: {best_dev_macro_f1:.4f}")
                    if args.save_best_by == "f1":
                        improved_main_metric = True

            if args.save_best_by in ["acc", "both"]:
                if dev_res["acc"] > best_dev_acc:
                    best_dev_acc = dev_res["acc"]
                    best_state_acc = clone_state_dict_to_cpu(model)
                    torch.save(best_state_acc, best_acc_path)
                    print(f"[SAVE] New best acc: {best_dev_acc:.4f}")
                    if args.save_best_by == "acc":
                        improved_main_metric = True

            if args.save_best_by == "both":
                if dev_res["macro_f1"] >= best_dev_macro_f1:
                    improved_main_metric = True

            # v_loss: lưu best model dựa trên train_loss
            if args.save_best_by in ["train_loss", "both"]:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_state_loss = clone_state_dict_to_cpu(model)
                    torch.save(best_state_loss, best_loss_path)
                    print(f"[SAVE] New best train_loss: {best_train_loss:.4f}")

            # ---- Per-epoch checkpoint (luôn lưu nếu --save_every_epoch) ----
            if args.save_every_epoch:
                epoch_ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}")
                os.makedirs(epoch_ckpt_dir, exist_ok=True)
                epoch_state = clone_state_dict_to_cpu(model)
                torch.save(epoch_state, os.path.join(epoch_ckpt_dir, "model_state.pt"))
                # Lưu kèm dev metrics để dễ so sánh sau
                save_json(os.path.join(epoch_ckpt_dir, "dev_metrics.json"), {
                    "epoch": epoch + 1,
                    "dev_acc": round(dev_res["acc"], 6),
                    "dev_macro_f1": round(dev_res["macro_f1"], 6),
                    "dev_weighted_f1": round(dev_res["weighted_f1"], 6),
                    "train_loss": round(train_loss, 6),
                })
                print(f"[EPOCH SAVE] checkpoint_epoch{epoch+1} | acc={dev_res['acc']:.4f} f1={dev_res['macro_f1']:.4f}")

            if improved_main_metric:
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    else:
        if args.eval_only:
            print("[EVAL_ONLY] skip training")
        else:
            print("[EVAL_FROM_EPOCH] skip training")

    if csv_f is not None:
        csv_f.flush()
        csv_f.close()

    def run_eval(tag, state_dict=None):
        # v_epoch_save: hỗ trợ tag dạng "epoch_N" để load checkpoint_epochN/model_state.pt
        if tag.startswith("epoch_"):
            epoch_num = tag.split("_")[1]
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch_num}", "model_state.pt")
        elif tag == "best_macro_f1":
            ckpt_path = best_f1_path
        elif tag == "best_acc":
            ckpt_path = best_acc_path
        elif tag == "best_train_loss":
            ckpt_path = best_loss_path
        else:
            raise ValueError(f"Unknown tag: {tag}")

        if args.eval_only or tag.startswith("epoch_"):
            print(f"[EVAL] loading from: {ckpt_path}")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] checkpoint not found: {ckpt_path}")
                return
            loaded = load_downstream_best_state_safely(model, ckpt_path, tag=tag)
            if not loaded:
                print(f"[WARN] Cannot load {tag}")
                return
        else:
            if state_dict is not None:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"[LOAD {tag} FROM MEMORY] missing={len(missing)} unexpected={len(unexpected)}")
            else:
                load_downstream_best_state_safely(model, ckpt_path, tag=tag)

        print("\n" + "=" * 80)
        print(f"EVALUATION ({tag.upper()})")
        print("=" * 80)

        # v_epoch_save: epoch eval lưu vào checkpoint_epochN/, best eval lưu vào output_dir
        is_epoch_tag = tag.startswith("epoch_")
        save_dir = os.path.join(args.output_dir, f"checkpoint_epoch{tag.split('_')[1]}") if is_epoch_tag else args.output_dir
        if is_epoch_tag:
            os.makedirs(save_dir, exist_ok=True)
        should_save = is_epoch_tag or (not args.eval_only)

        print("\n[DEV]")
        dev_res = evaluate(model=model, dataloader=dev_dl, pooling=args.pooling)
        print(f"dev_acc      : {dev_res['acc']:.4f}")
        print(f"dev_macroF1  : {dev_res['macro_f1']:.4f}")
        print(f"dev_weightF1 : {dev_res['weighted_f1']:.4f}")
        if dev_res["gate_mean"] is not None:
            print("dev_gate_mean:", [round(x, 4) for x in dev_res["gate_mean"].tolist()])

        if should_save:
            pd.DataFrame(dev_res["pred_rows"]).to_csv(
                os.path.join(save_dir, f"dev_predictions_{tag}.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            save_expert_activation_csv(dev_res["pred_rows"], save_dir, split="dev", tag=tag)
        else:
            print(f"[EVAL_ONLY] Skip writing dev_predictions_{tag}.csv")
            print(f"[EVAL_ONLY] Skip writing dev_expert_activations_{tag}.csv")

        print("\n[TEST]")
        test_res = evaluate(model=model, dataloader=test_dl, pooling=args.pooling)
        print(f"test_acc      : {test_res['acc']:.4f}")
        print(f"test_macroF1  : {test_res['macro_f1']:.4f}")
        print(f"test_weightF1 : {test_res['weighted_f1']:.4f}")
        if test_res["gate_mean"] is not None:
            print("test_gate_mean:", [round(x, 4) for x in test_res["gate_mean"].tolist()])

        print("\nClassification report (TEST):\n")
        print(test_res["report"])

        if should_save:
            pd.DataFrame(test_res["pred_rows"]).to_csv(
                os.path.join(save_dir, f"test_predictions_{tag}.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            save_expert_activation_csv(test_res["pred_rows"], save_dir, split="test", tag=tag)

            metrics = {
                "selected_by": tag,
                "dev": {
                    "acc": float(dev_res["acc"]),
                    "macro_f1": float(dev_res["macro_f1"]),
                    "weighted_f1": float(dev_res["weighted_f1"]),
                    "gate_mean": None if dev_res["gate_mean"] is None else [float(x) for x in dev_res["gate_mean"].tolist()],
                },
                "test": {
                    "acc": float(test_res["acc"]),
                    "macro_f1": float(test_res["macro_f1"]),
                    "weighted_f1": float(test_res["weighted_f1"]),
                    "gate_mean": None if test_res["gate_mean"] is None else [float(x) for x in test_res["gate_mean"].tolist()],
                    "classification_report": test_res["report"],
                },
                "training_config": training_config,
            }
            save_json(os.path.join(save_dir, f"metrics_{tag}.json"), metrics)
            print(f"[SAVED] → {save_dir}/metrics_{tag}.json")
        else:
            print(f"[EVAL_ONLY] Skip writing test_predictions_{tag}.csv")
            print(f"[EVAL_ONLY] Skip writing metrics_{tag}.json")

    if args.save_best_by == "f1":
        print("\nLoad best_macro_f1 checkpoint...")
        run_eval("best_macro_f1", best_state_f1 if not skip_training else None)

    elif args.save_best_by == "acc":
        print("\nLoad best_acc checkpoint...")
        run_eval("best_acc", best_state_acc if not skip_training else None)

    elif args.save_best_by == "train_loss":
        print("\nLoad best_train_loss checkpoint...")
        run_eval("best_train_loss", best_state_loss if not skip_training else None)

    elif args.save_best_by == "both":
        if args.eval_only:
            print("\nLoad best_macro_f1 checkpoint...")
            run_eval("best_macro_f1", None)
            print("\nLoad best_acc checkpoint...")
            run_eval("best_acc", None)
            print("\nLoad best_train_loss checkpoint...")
            run_eval("best_train_loss", None)
        elif not skip_training:
            print("\nLoad best_macro_f1 checkpoint...")
            run_eval("best_macro_f1", best_state_f1)
            print("\nLoad best_acc checkpoint...")
            run_eval("best_acc", best_state_acc)
            print("\nLoad best_train_loss checkpoint...")
            run_eval("best_train_loss", best_state_loss)

    # eval từng epoch được chỉ định qua --eval_from_epoch
    if args.eval_from_epoch:
        for ep in args.eval_from_epoch:
            print(f"\n{'='*80}")
            print(f"EVAL EPOCH {ep}")
            run_eval(f"epoch_{ep}")

    if args.eval_only:
        print("\n[EVAL_ONLY] Read-only evaluation complete.")
        print("[EVAL_ONLY] No training_config/train_log/prediction/expert_activation/metrics files were overwritten.")
        print("[EVAL_ONLY] Evaluated from:", args.output_dir)
    else:
        print("\nSaved to:", args.output_dir)


# ============================================================
# 9) CLI
# ============================================================

def build_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--src_dir", type=str, default="/content/drive/MyDrive/Projects-All/LLM_MoE_CL/src")

    ap.add_argument("--cmoe_ckpt_dir", type=str, required=True)
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--dev_file", type=str, required=True)
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--clf_dropout", type=float, default=0.2)
    ap.add_argument("--clf_hidden_mult", type=float, default=0.5)
    ap.add_argument("--early_stop_patience", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=100)

    ap.add_argument(
        "--pooling",
        type=str,
        default=None,
        choices=[None, "mean", "last_token", "last2_mean", "first_last_mean"],
    )
    ap.add_argument(
        "--tuning_mode",
        type=str,
        default="light",
        choices=["probe", "router_only", "light", "topn"],
    )
    ap.add_argument("--num_unfreeze_layers", type=int, default=0)
    ap.add_argument(
        "--save_best_by",
        type=str,
        default="both",
        choices=["f1", "acc", "both", "train_loss"],
    )
    ap.add_argument("--save_every_epoch", action="store_true",
                    help="")
    ap.add_argument("--eval_from_epoch", type=int, nargs="+", default=[],
                    help="")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--clf_num_layers", type=int, default=3)
    ap.add_argument("--clf_use_layernorm", action="store_true")

    ap.add_argument("--eval_only", action="store_true")

    # New target-adaptive transfer args.
    ap.add_argument("--no_target_router_conditioning", action="store_true")
    ap.add_argument(
        "--sep_token",
        type=str,
        default="[SEP]",
        help="Separator for concat mode, e.g. [SEP] or </s>.",
    )
    ap.add_argument(
        "--input_mode",
        type=str,
        default="concat",
        choices=["concat", "pair_feature"],
        help="concat: evidence sep claim -> one embedding; pair_feature: encode evidence and claim separately.",
    )
    ap.add_argument(
        "--pair_contrastive_source",
        type=str,
        default="avg",
        choices=["avg", "evidence", "claim", "diff"],
        help="Embedding used for SupCon when input_mode=pair_feature.",
    )
    ap.add_argument("--w_fc_con", type=float, default=0.0)
    ap.add_argument("--supcon_temp", type=float, default=0.07)
    ap.add_argument("--w_lb", type=float, default=0.0)

    # Optional NLI replay.
    ap.add_argument("--use_nli_replay", action="store_true")
    ap.add_argument("--vinli_dir", type=str, default="")
    ap.add_argument("--vianli_dir", type=str, default="")
    ap.add_argument("--nli_replay_batch_size", type=int, default=4)
    ap.add_argument("--w_nli_replay", type=float, default=0.0)
    ap.add_argument("--w_nli_sup", type=float, default=0.2)

    ap.add_argument("--viclsr_model", type=str, default="huynhtin/ViCLSR")
    ap.add_argument("--viclsr_max_len", type=int, default=128)
    ap.add_argument("--viclsr_batch_size", type=int, default=64)
    ap.add_argument("--mined_cache_dir", type=str, default="")
    ap.add_argument("--reuse_mined_dataset", action="store_true")
    ap.add_argument("--no_reuse_mined_dataset", action="store_true")

    ap.add_argument("--neutral_mode", type=str, default="weakpos", choices=["hardneg", "ignore", "weakpos"])
    ap.add_argument("--w_neutral_pos", type=float, default=0.2)
    ap.add_argument("--num_hard_negs", type=int, default=2)
    ap.add_argument("--num_synth_negs", type=int, default=1)
    ap.add_argument("--use_synthetic_hardneg", action="store_true")
    ap.add_argument("--no_use_synthetic_hardneg", action="store_true")
    ap.add_argument("--use_hardneg_weighting", action="store_true")
    ap.add_argument("--no_use_hardneg_weighting", action="store_true")
    ap.add_argument("--hardneg_weight_temp", type=float, default=0.2)
    ap.add_argument("--hardneg_weight_clamp", type=float, default=3.0)
    ap.add_argument("--mixup_alpha_min", type=float, default=0.3)
    ap.add_argument("--mixup_alpha_max", type=float, default=0.7)
    ap.add_argument("--w_adv_replay", type=float, default=0.0)

    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Defaults matching ver5.
    if not args.reuse_mined_dataset and not args.no_reuse_mined_dataset:
        args.reuse_mined_dataset = True
    if args.no_reuse_mined_dataset:
        args.reuse_mined_dataset = False

    if not args.use_synthetic_hardneg and not args.no_use_synthetic_hardneg:
        args.use_synthetic_hardneg = True
    if args.no_use_synthetic_hardneg:
        args.use_synthetic_hardneg = False

    if not args.use_hardneg_weighting and not args.no_use_hardneg_weighting:
        args.use_hardneg_weighting = True
    if args.no_use_hardneg_weighting:
        args.use_hardneg_weighting = False

    main(args)
