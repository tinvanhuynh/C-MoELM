# -*- coding: utf-8 -*-
"""
train_cmoe_lora_vinli_vianli_viclsr_3modes_entailpos.py

CMoE-LoRA contrastive training for Vietnamese NLI (ViNLI + ViANLI) with semantic hard-negative mining
using huynhtin/ViCLSR (XLM-R backbone).

Three experimental cases (toggle with --neutral_mode):
  Case 1: neutral_mode=hardneg
    - POSITIVE is ALWAYS entailment only (SimCSE / CL-NLI common setup)
    - Train only on entailment pairs (premise, hypothesis_entail)
    - Neutral + contradiction are used as negative pool to mine hard negatives (same premise first)

  Case 2: neutral_mode=ignore
    - POSITIVE is ALWAYS entailment only
    - Neutral is ignored entirely for hard negative mining
    - Contradiction is used as hard-negative pool (same premise first; fallback to global contradiction)

  Case 3: neutral_mode=weakpos
    - Train on entailment pairs (strong positive) AND neutral pairs (weak positive)
    - Add a neutral-positive InfoNCE term weighted by --w_neutral_pos
    - Hard negatives are preferably contradictions (fallback: non-entail)

Loss:
  L = L_contrastive + w_lb * L_loadbalance + w_adv * L_adv

Notes:
- DO NOT hardcode HF tokens. Use env var HF_TOKEN if needed.
"""

import os
import json
import math
import random
import argparse
import csv
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training

from functools import partial

# ---------------- Environment knobs ----------------
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional HF login
try:
    from huggingface_hub import login as hf_login
    token = os.environ.get("HF_TOKEN", None)
    if token:
        hf_login(token)
except Exception:
    pass

# ============================================================
# 0) Repro
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) Global gating context (example-level)
# ============================================================
class GateContext:
    gates: Optional[torch.Tensor] = None  # [B,E]


GATE_CTX = GateContext()


def set_gates(gates: Optional[torch.Tensor]):
    GATE_CTX.gates = gates


# ============================================================
# 2) Gradient Reversal Layer (domain adversarial)
# ============================================================
class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverseFn.apply(x, lambd)


# ============================================================
# 3) Pooling
# ============================================================
# def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     """
#     last_hidden: [B,T,H]
#     attention_mask: [B,T]
#     """
#     mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
#     summed = (last_hidden * mask).sum(dim=1)
#     denom = mask.sum(dim=1).clamp(min=1e-6)
#     return summed / denom

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: [B,T,H]
    attention_mask: [B,T]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Decoder-style last-token pooling.
    last_hidden: [B,T,H]
    attention_mask: [B,T]
    """
    last_idx = attention_mask.sum(dim=1) - 1  # [B]
    last_idx = last_idx.clamp(min=0)
    bsz = last_hidden.size(0)
    return last_hidden[torch.arange(bsz, device=last_hidden.device), last_idx]


def apply_pooling_from_outputs(outputs, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """
    pooling in {"mean", "last_token", "last2_mean", "first_last_mean"}
    """
    if pooling == "mean":
        return mean_pool(outputs.last_hidden_state, attention_mask)

    elif pooling == "last_token":
        return last_token_pool(outputs.last_hidden_state, attention_mask)

    elif pooling == "last2_mean":
        # cần output_hidden_states=True
        token_embs = 0.5 * (outputs.hidden_states[-1] + outputs.hidden_states[-2])
        return mean_pool(token_embs, attention_mask)

    elif pooling == "first_last_mean":
        # hidden_states[0] = embedding layer
        # hidden_states[1] = first transformer block output
        # ưu tiên dùng layer đầu tiên thực sự + layer cuối
        token_embs = 0.5 * (outputs.hidden_states[1] + outputs.hidden_states[-1])
        return mean_pool(token_embs, attention_mask)

    else:
        raise ValueError(f"Unknown pooling={pooling}")


# ============================================================
# 4) Router
# ============================================================
class Router(nn.Module):
    """
    Improved router:
    - input LayerNorm (stabilize routing logits)
    - temperature annealing (encourage exploration early, sharper later)
    - router noise on logits (exploration) during training
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_temp: float = 1.5,
        min_temp: float = 0.7,
        noise_std: float = 0.1,
        use_layernorm: bool = True,
        anneal: bool = True,
        use_mlp_router: bool = False,
        router_hidden_mult: float = 0.5,
        router_dropout: float = 0.1,
    ):
        super().__init__()

        self.init_temp = init_temp
        self.min_temp = min_temp
        self.noise_std = noise_std
        self.use_layernorm = use_layernorm
        self.anneal = anneal
        self.temperature = init_temp

        if use_layernorm:
            self.layernorm = nn.LayerNorm(hidden_size)
        else:
            self.layernorm = None

        if use_mlp_router:
            router_hidden = max(1, int(hidden_size * router_hidden_mult))
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, router_hidden),
                nn.GELU(),
                nn.Dropout(router_dropout),
                nn.Linear(router_hidden, num_experts),
            )
        else:
            self.proj = nn.Linear(hidden_size, num_experts)

    def set_temperature(self, step: int, total_steps: int):
        """Linear anneal temp from init_temp -> min_temp."""
        if not self.anneal:
            self.temperature = self.init_temp
            return
        total_steps = max(int(total_steps), 1)
        step = max(min(int(step), total_steps), 0)
        progress = step / total_steps
        self.temperature = self.min_temp + (self.init_temp - self.min_temp) * (1.0 - progress)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        target_dtype = None
        if self.layernorm is not None:
            target_dtype = self.layernorm.weight.dtype
        else:
            target_dtype = next(self.proj.parameters()).dtype

        pooled = pooled.to(dtype=target_dtype)

        if self.layernorm is not None:
            pooled = self.layernorm(pooled)

        logits = self.proj(pooled)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        temp = max(float(self.temperature), 1e-4)
        gates = F.softmax(logits / temp, dim=-1)  # [B,E]
        return gates


# ============================================================
# 5) LoRA-MoE Linear (experts = LoRA A/B per linear layer)
# ============================================================
class LoRAMoELinear(nn.Module):
    """
    Wrap base linear-like module:
      y = base_layer(x) + MoE-LoRA(x)

    Supports:
    - nn.Linear
    - bitsandbytes 4bit linear (as long as it has in_features/out_features and callable forward)
    """

    def __init__(
        self,
        base_linear: nn.Module,
        num_experts: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        top_k: int = 2,
    ):
        super().__init__()

        if not hasattr(base_linear, "in_features") or not hasattr(base_linear, "out_features"):
            raise TypeError(f"Unsupported base_linear type: {type(base_linear)}")

        self.base_layer = base_linear
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.num_experts = num_experts
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)


        # infer device for LoRA params
        ref_param = None
        for p in base_linear.parameters():
            ref_param = p
            break
        if ref_param is None:
            raise RuntimeError(f"No parameters found in base layer: {type(base_linear)}")

        base_device = ref_param.device

        # use explicit dtype for trainable LoRA params
        lora_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.A = nn.Parameter(
            torch.zeros(num_experts, r, self.in_features, device=base_device, dtype=lora_dtype)
        )
        self.B = nn.Parameter(
            torch.zeros(num_experts, self.out_features, r, device=base_device, dtype=lora_dtype)
        )

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # freeze base layer params
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)  # [B,T,out]
        gates = GATE_CTX.gates
        if gates is None:
            return base

        Bsz, T, _ = x.shape
        E = self.num_experts
        x_d = self.dropout(x).to(dtype=self.A.dtype)

        if self.top_k >= E:
            xa = torch.einsum("bti,eri->bter", x_d, self.A)
            xab = torch.einsum("bter,eor->bteo", xa, self.B)
            mixed = torch.einsum("bteo,be->bto", xab, gates.to(dtype=xab.dtype)) * self.scaling
            return base + mixed.to(dtype=base.dtype)

        topv, topi = torch.topk(gates, k=self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = base
        for k in range(self.top_k):
            idx = topi[:, k]
            w = topv[:, k].view(Bsz, 1, 1).to(dtype=self.A.dtype)
            A_k = self.A[idx]
            B_k = self.B[idx]

            xa = torch.einsum("bti,bri->btr", x_d, A_k)
            xab = torch.einsum("btr,bor->bto", xa, B_k)
            out = out + (w * (xab * self.scaling)).to(dtype=out.dtype)

        return out


TARGET_KEYWORDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def should_patch(local_name: str) -> bool:
    return any(k in local_name for k in TARGET_KEYWORDS)


def patch_lora_moe(module: nn.Module, num_experts: int, r: int, alpha: float, dropout: float, top_k: int):
    """
    Recursively replace target projection modules with LoRAMoELinear.
    Patch by name, not by strict nn.Linear type.
    """
    for name, child in list(module.named_children()):
        if should_patch(name) and hasattr(child, "in_features") and hasattr(child, "out_features"):
            setattr(
                module,
                name,
                LoRAMoELinear(
                    child,
                    num_experts=num_experts,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    top_k=top_k,
                ),
            )
        else:
            patch_lora_moe(child, num_experts, r, alpha, dropout, top_k)


# ============================================================
# 6) CMoE-LoRA wrapper + HF-style save/load
# ============================================================
@dataclass
class CMoELoRAConfig:
    base_model: str
    num_experts: int = 4
    top_k: int = 2
    r: int = 8
    alpha: float = 16.0
    lora_dropout: float = 0.05
    grl_lambda: float = 1.0
    temperature: float = 0.07

    # QLoRA
    use_qlora: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None

    # Router
    router_init_temp: float = 1.5
    router_min_temp: float = 0.7
    router_noise_std: float = 0.1
    router_use_layernorm: bool = True
    router_anneal: bool = True

    # Router architecture
    use_mlp_router: bool = False
    router_hidden_mult: float = 0.5
    router_dropout: float = 0.1

    # Embedding / pooling
    emb_dim: Optional[int] = None
    pooling: str = "mean"     # mean | last_token | last2_mean | first_last_mean

    # NEW: partial unfreeze
    unfreeze_top_layers: int = 0
    train_final_norm: bool = False

    # NEW: stronger projection head
    use_mlp_proj: bool = False
    proj_hidden_mult: float = 1.0
    proj_dropout: float = 0.1

    # NEW: supervised NLI auxiliary head
    use_sup_nli: bool = False
    sup_dropout: float = 0.1



class CMoELoRA(nn.Module):
    def __init__(self, cfg: CMoELoRAConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_qlora and (cfg.unfreeze_top_layers > 0 or cfg.train_final_norm):
            print("[WARN] Running hybrid mode: QLoRA + partial backbone unfreeze")

        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg.bnb_4bit_compute_dtype]

        model_kwargs = {
            "torch_dtype": compute_dtype,
        }

        if cfg.attn_implementation is not None:
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        if cfg.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config

        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            **model_kwargs,
        )

        if cfg.use_qlora:
            self.lm = prepare_model_for_kbit_training(
                self.lm,
                use_gradient_checkpointing=False
            )

        self.hidden_size = self.lm.config.hidden_size

        patch_lora_moe(
            self.lm,
            cfg.num_experts,
            cfg.r,
            cfg.alpha,
            cfg.lora_dropout,
            cfg.top_k,
        )
        # -------------------------------------------------
        # Trainable policy
        # 1) freeze all LM params
        # 2) always train LoRA A/B
        # 3) optionally unfreeze top N transformer layers
        # 4) optionally unfreeze final norm
        # -------------------------------------------------
        for _, p in self.lm.named_parameters():
            p.requires_grad = False

        # always train LoRA experts
        for n, p in self.lm.named_parameters():
            if n.endswith(".A") or n.endswith(".B"):
                p.requires_grad = True

        # optionally unfreeze top transformer layers
        num_unfreeze = int(getattr(cfg, "unfreeze_top_layers", 0))
        if num_unfreeze > 0 and hasattr(self.lm.model, "layers"):
            total_layers = len(self.lm.model.layers)
            start_idx = max(0, total_layers - num_unfreeze)
            for layer_idx in range(start_idx, total_layers):
                for name, p in self.lm.model.layers[layer_idx].named_parameters():
                    # only unfreeze floating-point params
                    if p.is_floating_point():
                        p.requires_grad = True

        # optionally unfreeze final norm
        if getattr(cfg, "train_final_norm", False) and hasattr(self.lm.model, "norm"):
            for p in self.lm.model.norm.parameters():
                if p.is_floating_point():
                    p.requires_grad = True

        self.router = Router(
            self.hidden_size,
            cfg.num_experts,
            init_temp=cfg.router_init_temp,
            min_temp=cfg.router_min_temp,
            noise_std=cfg.router_noise_std,
            use_layernorm=cfg.router_use_layernorm,
            anneal=cfg.router_anneal,
            use_mlp_router=cfg.use_mlp_router,
            router_hidden_mult=cfg.router_hidden_mult,
            router_dropout=cfg.router_dropout,
        )
        out_dim = cfg.emb_dim or self.hidden_size
        self.proj = None

        # NEW: stronger projection head
        if cfg.use_mlp_proj:
            proj_hidden = int(self.hidden_size * cfg.proj_hidden_mult)
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_size, proj_hidden),
                nn.GELU(),
                nn.Dropout(cfg.proj_dropout),
                nn.Linear(proj_hidden, out_dim),
            )
        elif out_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, out_dim)

        self.domain_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, 2),  # 0: ViNLI, 1: ViANLI
        )

        # NEW: supervised NLI auxiliary head
        self.nli_head = None
        if cfg.use_sup_nli:
            pair_dim = out_dim * 4
            self.nli_head = nn.Sequential(
                nn.Linear(pair_dim, out_dim),
                nn.GELU(),
                nn.Dropout(cfg.sup_dropout),
                nn.Linear(out_dim, 3),
            )

        # Align dtype + device with backbone
        model_dtype = compute_dtype
        backbone_device = self.lm.model.embed_tokens.weight.device

        self.router = self.router.to(device=backbone_device, dtype=model_dtype)
        self.domain_head = self.domain_head.to(device=backbone_device, dtype=model_dtype)
        if self.proj is not None:
            self.proj = self.proj.to(device=backbone_device, dtype=model_dtype)
        if self.nli_head is not None:
            self.nli_head = self.nli_head.to(device=backbone_device, dtype=model_dtype)

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / cfg.temperature), device=backbone_device, dtype=model_dtype)
        )


    @torch.no_grad()
    def _pooled_no_lora(self, input_ids, attention_mask) -> torch.Tensor:
        """
        First pass: get router features with gates=None (LoRA inactive).
        Khi train thật sự với --pooling khác mean, router cũng dùng cùng pooling đó.
        """
        set_gates(None)
        need_hidden_states = (self.cfg.pooling in ["last2_mean", "first_last_mean"])
        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        return apply_pooling_from_outputs(out, attention_mask, self.cfg.pooling)

    def encode(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main training/inference path.
        Dùng pooling theo self.cfg.pooling.
        """
        return self.encode_with_pooling(input_ids, attention_mask, pooling=self.cfg.pooling)

    def encode_with_pooling(self, input_ids, attention_mask, pooling: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eval-only flexible extraction:
        - Nếu pooling == self.cfg.pooling: khớp đúng thiết kế train
        - Nếu pooling != self.cfg.pooling: useful cho ablation nhanh trên cùng checkpoint

        Lưu ý:
        - Router pass 1 vẫn dùng self.cfg.pooling nếu bạn train model đó.
        - Ở eval ablation, bạn có thể đổi pooling ở pass 2 để benchmark extraction.
        """
        # Router features: giữ đúng config train của model
        pooled0 = self._pooled_no_lora(input_ids, attention_mask)  # [B,H]
        gates = self.router(pooled0)                               # [B,E]

        need_hidden_states = (pooling in ["last2_mean", "first_last_mean"])

        set_gates(gates)
        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        set_gates(None)

        pooled = apply_pooling_from_outputs(out, attention_mask, pooling)

        if self.proj is not None:
            proj_dtype = next(self.proj.parameters()).dtype
            pooled = pooled.to(dtype=proj_dtype)
            pooled = self.proj(pooled)

        emb = F.normalize(pooled, dim=-1)
        return emb, gates
    def encode_superbatch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pooling is None:
            pooling = self.cfg.pooling

        # pass 1: no LoRA / no gates
        pooled0 = self._pooled_no_lora(input_ids, attention_mask)
        gates = self.router(pooled0)

        need_hidden_states = pooling in ["last2_mean", "first_last_mean"]

        # pass 2: apply gates
        set_gates(gates)
        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        set_gates(None)

        pooled = apply_pooling_from_outputs(out, attention_mask, pooling)

        if self.proj is not None:
            proj_dtype = next(self.proj.parameters()).dtype
            pooled = pooled.to(dtype=proj_dtype)
            pooled = self.proj(pooled)

        emb = F.normalize(pooled, dim=-1)
        return emb, gates


    # ---------- HF-style save/load ----------
    def save_pretrained(self, save_directory: str, tokenizer=None, extra_config: Optional[Dict] = None):
        os.makedirs(save_directory, exist_ok=True)

        cfg_to_save = asdict(self.cfg)
        if extra_config is not None:
            cfg_to_save["training_meta"] = extra_config

        with open(os.path.join(save_directory, "cmoe_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_to_save, f, ensure_ascii=False, indent=2)

        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        safetensors_save(state, os.path.join(save_directory, "model.safetensors"))

        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        cfg_path = os.path.join(load_directory, "cmoe_config.json")
        weights_path = os.path.join(load_directory, "model.safetensors")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing {cfg_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing {weights_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg_core = {
            k: v for k, v in cfg_dict.items()
            if k in CMoELoRAConfig.__dataclass_fields__
        }
        cfg = CMoELoRAConfig(**cfg_core)

        model = cls(cfg)
        state = safetensors_load(weights_path)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State mismatch. missing={missing}, unexpected={unexpected}")

        if device is not None and not cfg.use_qlora:
            model.to(device)
        model.eval()
        return model


# ============================================================
# 7) Data loading + ViCLSR mining
# ============================================================
LABEL_MAP = {"e": 0, "n": 1, "c": 2}  # entail, neutral, contradiction


def load_jsonl(path: str, domain_id: int) -> List[Dict]:
    """Expect jsonl lines: {"premise":..., "hypothesis":..., "label":"e|n|c"}"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if "premise" not in ex or "hypothesis" not in ex or "label" not in ex:
                continue
            lab = ex["label"]
            if lab not in LABEL_MAP:
                continue
            data.append({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": LABEL_MAP[lab],
                "domain": domain_id,
            })
    return data


class SemanticMinerViCLSR:
    """ViCLSR encoder (XLM-R). mean pooling + L2 norm. Used ONLY for mining hard negatives."""

    def __init__(self, model_name="huynhtin/ViCLSR", device="cuda", max_len=128, batch_size=64):
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(model_name).to(device)
        self.enc.eval()

        self.cache: Dict[str, torch.Tensor] = {}  # text -> CPU float32 [D] normalized

    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), self.batch_size):
            bt = texts[i:i+self.batch_size]
            batch = self.tok(bt, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            hs = self.enc(**batch).last_hidden_state  # [B,T,H]
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1)
            outs.append(pooled.detach().cpu().float())
        return torch.cat(outs, dim=0)

    def precompute(self, texts: List[str]):
        texts = [t for t in texts if t not in self.cache]
        if not texts:
            return
        embs = self._embed_batch(texts)
        for t, e in zip(texts, embs):
            self.cache[t] = e

    def pick_hard_neg(self, query_text: str, candidates: List[str]) -> Tuple[str, float]:
        """Backward-compatible: return single hardest candidate."""
        topk = self.pick_topk_hard_negs(query_text, candidates, k=1)
        if len(topk) == 0:
            return query_text, 0.0
        return topk[0]

    def pick_topk_hard_negs(self, query_text: str, candidates: List[str], k: int = 4) -> List[Tuple[str, float]]:
        """
        Return top-k candidates with highest cosine similarity to query_text in ViCLSR space.
        Output: [(text, sim), ...] sorted descending by sim.
        """
        if len(candidates) == 0:
            return []

        uniq_candidates = list(dict.fromkeys(candidates))
        self.precompute([query_text] + uniq_candidates)

        q = self.cache[query_text]  # [D]
        c = torch.stack([self.cache[t] for t in uniq_candidates], dim=0)  # [N,D]
        sims = c @ q

        topk = min(k, len(uniq_candidates))
        vals, idxs = torch.topk(sims, k=topk, largest=True)
        return [(uniq_candidates[int(i)], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]


def build_same_premise_hardneg_semantic(
    examples: List[Dict],
    miner: SemanticMinerViCLSR,
    seed: int = 13,
    neutral_mode: str = "hardneg",
    num_hard_negs: int = 4,
    keep_full_nli_for_sup: bool = False,
) -> List[Dict]:
    """
    Attach ex["hard_negatives"] and ex["hard_negative_scores"] for each TRAIN example we keep.

    hardneg:
      - Keep ONLY entailment examples
      - Negatives: same-premise non-entail (neutral + contradiction)

    ignore:
      - Keep ONLY entailment examples
      - Negatives: contradiction only

    weakpos:
      - Keep entailment + neutral examples
      - Drop contradictions from positive training pairs
      - Hard negatives use contradiction as primary pool
      - Neutral is NOT used as main negative pool in weakpos
    """
    rnd = random.Random(seed)
    by_premise = defaultdict(list)
    global_non_entail: List[str] = []
    global_contra: List[str] = []
    global_all: List[str] = []

    for ex in examples:
        by_premise[ex["premise"]].append(ex)
        global_all.append(ex["hypothesis"])
        if ex["label"] != LABEL_MAP["e"]:
            global_non_entail.append(ex["hypothesis"])
        if ex["label"] == LABEL_MAP["c"]:
            global_contra.append(ex["hypothesis"])

    all_h = list({ex["hypothesis"] for ex in examples})
    miner.precompute(all_h)

    kept = []
    for ex in examples:
        # keep positives
        if neutral_mode in ["hardneg", "ignore"]:
            # contrastive: entailment only
            # supervised: optionally keep full NLI labels
            if (ex["label"] != LABEL_MAP["e"]) and (not keep_full_nli_for_sup):
                continue

        elif neutral_mode == "weakpos":
            # contrastive: entailment + neutral
            # supervised: optionally keep contradiction too
            if (ex["label"] == LABEL_MAP["c"]) and (not keep_full_nli_for_sup):
                continue

        else:
            raise ValueError(f"Unknown neutral_mode={neutral_mode}")

        group = by_premise[ex["premise"]]

        if neutral_mode == "hardneg":
            cands = [g for g in group if g["label"] != LABEL_MAP["e"] and g["hypothesis"] != ex["hypothesis"]]
            cand_texts = [c["hypothesis"] for c in cands]
            topk = miner.pick_topk_hard_negs(ex["hypothesis"], cand_texts, k=num_hard_negs)

            if len(topk) < num_hard_negs:
                remain = num_hard_negs - len(topk)
                existing = {t for t, _ in topk}
                fallback_pool = [t for t in global_non_entail if t not in existing and t != ex["hypothesis"]]
                if len(fallback_pool) == 0:
                    fallback_pool = [t for t in global_all if t not in existing and t != ex["hypothesis"]]
                # semantic fallback using ViCLSR instead of random
                fallback_candidates = [t for t in fallback_pool if t != ex["hypothesis"]]

                if len(fallback_candidates) > 0:
                    fallback_topk = miner.pick_topk_hard_negs(
                        ex["hypothesis"],
                        fallback_candidates,
                        k=remain
                    )
                    topk.extend(fallback_topk)

        elif neutral_mode == "ignore":
            cands = [g for g in group if g["label"] == LABEL_MAP["c"] and g["hypothesis"] != ex["hypothesis"]]
            cand_texts = [c["hypothesis"] for c in cands]
            topk = miner.pick_topk_hard_negs(ex["hypothesis"], cand_texts, k=num_hard_negs)

            if len(topk) < num_hard_negs:
                remain = num_hard_negs - len(topk)
                existing = {t for t, _ in topk}
                fallback_pool = [t for t in global_contra if t not in existing and t != ex["hypothesis"]]
                if len(fallback_pool) == 0:
                    fallback_pool = [t for t in global_non_entail if t not in existing and t != ex["hypothesis"]]
                if len(fallback_pool) == 0:
                    fallback_pool = [t for t in global_all if t not in existing and t != ex["hypothesis"]]
                # semantic fallback using ViCLSR instead of random
                fallback_candidates = [t for t in fallback_pool if t != ex["hypothesis"]]

                if len(fallback_candidates) > 0:
                    fallback_topk = miner.pick_topk_hard_negs(
                        ex["hypothesis"],
                        fallback_candidates,
                        k=remain
                    )
                    topk.extend(fallback_topk)

        else:  # weakpos
            # IMPORTANT:
            # neutral remains weak positive and must NOT return as hard negative
            # hard negative pool is contradiction-only

            cands = [
                g for g in group
                if g["label"] == LABEL_MAP["c"] and g["hypothesis"] != ex["hypothesis"]
            ]
            cand_texts = [c["hypothesis"] for c in cands]
            topk = miner.pick_topk_hard_negs(ex["hypothesis"], cand_texts, k=num_hard_negs)

            if len(topk) < num_hard_negs:
                remain = num_hard_negs - len(topk)
                existing = {t for t, _ in topk}

                # contradiction-only semantic fallback
                fallback_pool = [
                    t for t in global_contra
                    if t not in existing and t != ex["hypothesis"]
                ]

                if len(fallback_pool) > 0:
                    fallback_topk = miner.pick_topk_hard_negs(
                        ex["hypothesis"],
                        fallback_pool,
                        k=remain
                    )
                    topk.extend(fallback_topk)

        topk = topk[:num_hard_negs]
        # ensure fixed K hard negatives for batching
        while len(topk) < num_hard_negs:
            if len(topk) > 0:
                topk.append(topk[-1])  # duplicate last negative
            else:
                # extremely rare fallback
                topk.append((ex["hypothesis"], 0.0))

        ex2 = dict(ex)
        ex2["hard_negatives"] = [t for t, _ in topk]
        ex2["hard_negative_scores"] = [float(s) for _, s in topk]

        # contrastive participation depends on mode
        if neutral_mode in ["hardneg", "ignore"]:
            # only entailment contributes to contrastive
            ex2["use_contrastive"] = 1 if ex["label"] == LABEL_MAP["e"] else 0

        elif neutral_mode == "weakpos":
            # entailment + neutral contribute to contrastive
            ex2["use_contrastive"] = 1 if ex["label"] in [LABEL_MAP["e"], LABEL_MAP["n"]] else 0

        else:
            ex2["use_contrastive"] = 0

        kept.append(ex2)

    return kept

def load_processed_dataset(path: str) -> List[Dict]:
    """
    Load previously mined dataset from jsonl cache.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"\nLoaded processed dataset ← {path}")
    return data

class NLITripletDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len: int):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def _encode(self, text: str):
        return self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        p = self._encode(ex["premise"])
        h = self._encode(ex["hypothesis"])

        neg_ids = []
        neg_masks = []
        for neg_text in ex["hard_negatives"]:
            n = self._encode(neg_text)
            neg_ids.append(n["input_ids"].squeeze(0))
            neg_masks.append(n["attention_mask"].squeeze(0))

        max_neg_len = max(x.size(0) for x in neg_ids)

        def pad_1d_local(x, target_len, pad_value):
            out = x.new_full((target_len,), pad_value)
            out[: x.size(0)] = x
            return out

        def pad_mask_1d_local(x, target_len):
            out = x.new_zeros((target_len,))
            out[: x.size(0)] = x
            return out

        neg_ids = [pad_1d_local(x, max_neg_len, self.tok.pad_token_id) for x in neg_ids]
        neg_masks = [pad_mask_1d_local(x, max_neg_len) for x in neg_masks]

        n_input_ids = torch.stack(neg_ids, dim=0)   # [K, T_local]
        n_mask = torch.stack(neg_masks, dim=0)      # [K, T_local]

        neg_scores = torch.tensor(
            ex.get("hard_negative_scores", [0.0] * len(ex["hard_negatives"])),
            dtype=torch.float
        )

        return {
            "p_input_ids": p["input_ids"].squeeze(0),
            "p_mask": p["attention_mask"].squeeze(0),
            "h_input_ids": h["input_ids"].squeeze(0),
            "h_mask": h["attention_mask"].squeeze(0),
            "n_input_ids": n_input_ids,     # [K,T]
            "n_mask": n_mask,               # [K,T]
            "neg_scores": neg_scores,       # [K]
            "pair_label": torch.tensor(ex["label"], dtype=torch.long),
            "domain_label": torch.tensor(ex["domain"], dtype=torch.long),
            "use_contrastive": torch.tensor(ex.get("use_contrastive", 1), dtype=torch.long),
        }

def collate_nli_triplet_dynamic(batch, pad_token_id: int):
    """
    Dynamic padding with ONE shared max length for p/h/n in the batch.
    This is required because training_step concatenates p, h, n along dim=0.
    """
    K = batch[0]["n_input_ids"].size(0)

    max_len = 1
    for ex in batch:
        max_len = max(max_len, ex["p_input_ids"].size(0))
        max_len = max(max_len, ex["h_input_ids"].size(0))
        max_len = max(max_len, ex["n_input_ids"].size(1))  # [K, T]

    def pad_1d(x, target_len, pad_value):
        out = x.new_full((target_len,), pad_value)
        out[: x.size(0)] = x
        return out

    def pad_mask_1d(x, target_len):
        out = x.new_zeros((target_len,))
        out[: x.size(0)] = x
        return out

    p_input_ids, p_mask = [], []
    h_input_ids, h_mask = [], []
    n_input_ids, n_mask = [], []
    neg_scores = []
    pair_label = []
    domain_label = []
    use_contrastive = []

    for ex in batch:
        p_input_ids.append(pad_1d(ex["p_input_ids"], max_len, pad_token_id))
        p_mask.append(pad_mask_1d(ex["p_mask"], max_len))

        h_input_ids.append(pad_1d(ex["h_input_ids"], max_len, pad_token_id))
        h_mask.append(pad_mask_1d(ex["h_mask"], max_len))

        neg_ids_i = []
        neg_mask_i = []
        for k in range(K):
            neg_ids_i.append(pad_1d(ex["n_input_ids"][k], max_len, pad_token_id))
            neg_mask_i.append(pad_mask_1d(ex["n_mask"][k], max_len))

        n_input_ids.append(torch.stack(neg_ids_i, dim=0))   # [K, max_len]
        n_mask.append(torch.stack(neg_mask_i, dim=0))       # [K, max_len]

        neg_scores.append(ex["neg_scores"])
        pair_label.append(ex["pair_label"])
        domain_label.append(ex["domain_label"])
        use_contrastive.append(ex["use_contrastive"])

    return {
        "p_input_ids": torch.stack(p_input_ids, dim=0),   # [B, max_len]
        "p_mask": torch.stack(p_mask, dim=0),
        "h_input_ids": torch.stack(h_input_ids, dim=0),   # [B, max_len]
        "h_mask": torch.stack(h_mask, dim=0),
        "n_input_ids": torch.stack(n_input_ids, dim=0),   # [B, K, max_len]
        "n_mask": torch.stack(n_mask, dim=0),
        "neg_scores": torch.stack(neg_scores, dim=0),
        "pair_label": torch.stack(pair_label, dim=0),
        "domain_label": torch.stack(domain_label, dim=0),
        "use_contrastive": torch.stack(use_contrastive, dim=0),
    }

# ============================================================
# 8) Losses
# ============================================================
def synthetic_hard_negatives(
    hard_neg_embs: torch.Tensor,   # [B,K,D]
    num_synth: int = 2,
    alpha_min: float = 0.3,
    alpha_max: float = 0.7,
) -> torch.Tensor:
    """
    Pairwise mixup from real hard negatives.
    Return: [B,M,D]
    """
    B, K, D = hard_neg_embs.shape
    if num_synth <= 0 or K < 2:
        return hard_neg_embs.new_zeros((B, 0, D))

    device = hard_neg_embs.device
    dtype = hard_neg_embs.dtype
    synths = []

    for _ in range(num_synth):
        i = torch.randint(low=0, high=K, size=(B,), device=device)
        j = torch.randint(low=0, high=K, size=(B,), device=device)

        same = (i == j)
        if same.any():
            j[same] = (j[same] + 1) % K

        alpha = torch.empty(B, 1, device=device, dtype=dtype).uniform_(alpha_min, alpha_max)

        ni = hard_neg_embs[torch.arange(B, device=device), i]  # [B,D]
        nj = hard_neg_embs[torch.arange(B, device=device), j]  # [B,D]

        mix = alpha * ni + (1.0 - alpha) * nj
        mix = F.normalize(mix, dim=-1).to(dtype=dtype)
        synths.append(mix)

    return torch.stack(synths, dim=1).to(dtype=dtype)  # [B,M,D]


def info_nce_inbatch_with_multihardneg_masked(
    a_emb: torch.Tensor,
    p_emb: torch.Tensor,
    hard_neg_embs: torch.Tensor,
    hard_neg_scores: torch.Tensor,
    synth_neg_embs: Optional[torch.Tensor],
    mask: torch.Tensor,
    logit_scale: torch.Tensor,
    hardneg_weight_temp: float = 0.2,
    hardneg_weight_clamp: float = 3.0,
    use_hardneg_weighting: bool = True,
) -> torch.Tensor:
    """
    InfoNCE with:
      - in-batch negatives
      - K real hard negatives
      - optional M synthetic hard negatives
      - hardness-aware weighting
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=a_emb.device, dtype=a_emb.dtype)

    B = a_emb.size(0)
    hard_neg_embs = hard_neg_embs.to(dtype=a_emb.dtype)
    hard_neg_scores = hard_neg_scores.to(device=a_emb.device, dtype=a_emb.dtype)
    if synth_neg_embs is not None:
        synth_neg_embs = synth_neg_embs.to(device=a_emb.device, dtype=a_emb.dtype)
    logit_scale = logit_scale.to(dtype=a_emb.dtype)

    # [B,B]
    sims = logit_scale * (a_emb @ p_emb.t())

    # [B,K]
    sim_ah = logit_scale * torch.einsum("bd,bkd->bk", a_emb, hard_neg_embs)

    if use_hardneg_weighting:
        real_logits = hard_neg_scores / max(hardneg_weight_temp, 1e-6)
        if hardneg_weight_clamp > 0:
            real_logits = torch.clamp(real_logits, min=-hardneg_weight_clamp, max=hardneg_weight_clamp)
        w_real = F.softmax(real_logits, dim=-1)
        sim_ah = sim_ah + torch.log(w_real + 1e-12)

    extra_logits = [sim_ah]

    if synth_neg_embs is not None and synth_neg_embs.size(1) > 0:
        # [B,M]
        sim_as = logit_scale * torch.einsum("bd,bmd->bm", a_emb, synth_neg_embs)

        if use_hardneg_weighting:
            syn_logits = sim_as.detach() / max(hardneg_weight_temp, 1e-6)
            if hardneg_weight_clamp > 0:
                syn_logits = torch.clamp(syn_logits, min=-hardneg_weight_clamp, max=hardneg_weight_clamp)
            w_syn = F.softmax(syn_logits, dim=-1)
            sim_as = sim_as + torch.log(w_syn + 1e-12)
        extra_logits.append(sim_as)

    logits = torch.cat([sims] + extra_logits, dim=1)  # [B, B+K(+M)]
    targets = torch.arange(B, device=logits.device)

    ce = F.cross_entropy(logits, targets, reduction="none")[mask]
    return ce.mean()


def load_balance_switch(gates_a: torch.Tensor, gates_b: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Switch-style balancing loss."""
    gates = 0.5 * (gates_a + gates_b)  # [B,E]
    B, E = gates.shape
    importance = gates.mean(dim=0)  # [E]
    top1 = torch.argmax(gates, dim=-1)  # [B]
    load = torch.bincount(top1, minlength=E).float() / float(B)  # [E]
    loss = E * torch.sum(importance * load) - 1.0
    return loss, {"importance": importance.detach(), "load": load.detach(), "top1": top1.detach()}


def training_step(
    model: CMoELoRA,
    batch: Dict[str, torch.Tensor],
    neutral_mode: str,
    w_neutral_pos: float,
    w_adv: float,
    w_lb: float,
    w_sup: float,
    step: int,
    total_steps: int,
    num_synth_negs: int,
    mixup_alpha_min: float,
    mixup_alpha_max: float,
    hardneg_weight_temp: float,
    hardneg_weight_clamp: float,
    use_synthetic_hardneg: bool,
    use_hardneg_weighting: bool,
) -> Dict[str, torch.Tensor]:
    """One training step returning losses + gating stats."""

    B, K, T = batch["n_input_ids"].shape

    p_ids = batch["p_input_ids"]          # [B,T]
    p_mask = batch["p_mask"]              # [B,T]
    h_ids = batch["h_input_ids"]          # [B,T]
    h_mask = batch["h_mask"]              # [B,T]

    n_ids = batch["n_input_ids"].view(B * K, T)   # [B*K,T]
    n_mask = batch["n_mask"].view(B * K, T)       # [B*K,T]

    # -------- super-batch concat --------
    all_ids = torch.cat([p_ids, h_ids, n_ids], dim=0)      # [B + B + B*K, T]
    all_mask = torch.cat([p_mask, h_mask, n_mask], dim=0)

    all_emb, all_gates = model.encode_superbatch(all_ids, all_mask)

    # -------- split back --------
    p_emb = all_emb[:B]
    h_emb = all_emb[B:2 * B]
    n_emb_flat = all_emb[2 * B:]
    n_emb = n_emb_flat.view(B, K, -1)

    gates_p = all_gates[:B]
    gates_h = all_gates[B:2 * B]

    neg_scores = batch["neg_scores"]

    if use_synthetic_hardneg:
        synth_emb = synthetic_hard_negatives(
            n_emb,
            num_synth=num_synth_negs,
            alpha_min=mixup_alpha_min,
            alpha_max=mixup_alpha_max,
        )
    else:
        synth_emb = None

    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    pair_label = batch["pair_label"]
    use_contrastive = batch["use_contrastive"].bool()
    num_contrastive = use_contrastive.sum().detach()
    num_sup_only = (~use_contrastive).sum().detach()

    num_contrastive_entail = ((pair_label == LABEL_MAP["e"]) & use_contrastive).sum().detach()
    num_contrastive_neutral = ((pair_label == LABEL_MAP["n"]) & use_contrastive).sum().detach()
    num_sup_contra = ((pair_label == LABEL_MAP["c"]) & (~use_contrastive)).sum().detach()

    if neutral_mode in ["hardneg", "ignore"]:
        mask_e = (pair_label == LABEL_MAP["e"]) & use_contrastive
        loss_ph = info_nce_inbatch_with_multihardneg_masked(
            p_emb, h_emb, n_emb, neg_scores, synth_emb, mask_e, logit_scale,
            hardneg_weight_temp=hardneg_weight_temp,
            hardneg_weight_clamp=hardneg_weight_clamp,
            use_hardneg_weighting=use_hardneg_weighting,
        )
        loss_hp = info_nce_inbatch_with_multihardneg_masked(
            h_emb, p_emb, n_emb, neg_scores, synth_emb, mask_e, logit_scale,
            hardneg_weight_temp=hardneg_weight_temp,
            hardneg_weight_clamp=hardneg_weight_clamp,
            use_hardneg_weighting=use_hardneg_weighting,
        )
        loss_con_e = 0.5 * (loss_ph + loss_hp)
        loss_con_n = torch.tensor(0.0, device=p_emb.device, dtype=p_emb.dtype)
        loss_con = loss_con_e

    elif neutral_mode == "weakpos":
        mask_e = (pair_label == LABEL_MAP["e"]) & use_contrastive
        mask_n = (pair_label == LABEL_MAP["n"]) & use_contrastive

        loss_e_ph = info_nce_inbatch_with_multihardneg_masked(
            p_emb, h_emb, n_emb, neg_scores, synth_emb, mask_e, logit_scale,
            hardneg_weight_temp=hardneg_weight_temp,
            hardneg_weight_clamp=hardneg_weight_clamp,
            use_hardneg_weighting=use_hardneg_weighting,
        )
        loss_e_hp = info_nce_inbatch_with_multihardneg_masked(
            h_emb, p_emb, n_emb, neg_scores, synth_emb, mask_e, logit_scale,
            hardneg_weight_temp=hardneg_weight_temp,
            hardneg_weight_clamp=hardneg_weight_clamp,
            use_hardneg_weighting=use_hardneg_weighting,
        )
        loss_con_e = 0.5 * (loss_e_ph + loss_e_hp)

        loss_n_ph = info_nce_inbatch_with_multihardneg_masked(
            p_emb, h_emb, n_emb, neg_scores, synth_emb, mask_n, logit_scale,
            hardneg_weight_temp=hardneg_weight_temp,
            hardneg_weight_clamp=hardneg_weight_clamp,
            use_hardneg_weighting=use_hardneg_weighting,
        )
        loss_n_hp = info_nce_inbatch_with_multihardneg_masked(
            h_emb, p_emb, n_emb, neg_scores, synth_emb, mask_n, logit_scale,
            hardneg_weight_temp=hardneg_weight_temp,
            hardneg_weight_clamp=hardneg_weight_clamp,
            use_hardneg_weighting=use_hardneg_weighting,
        )
        loss_con_n = 0.5 * (loss_n_ph + loss_n_hp)

        loss_con = loss_con_e + (w_neutral_pos * loss_con_n)

    else:
        raise ValueError(f"Unknown neutral_mode={neutral_mode}")
    
    # -------------------------------------------------
    # NEW: supervised NLI auxiliary loss
    # -------------------------------------------------
    if model.nli_head is not None:
        pair_feat = torch.cat([
            p_emb,
            h_emb,
            torch.abs(p_emb - h_emb),
            p_emb * h_emb,
        ], dim=-1)
        nli_logits = model.nli_head(pair_feat)
        loss_sup = F.cross_entropy(nli_logits.float(), pair_label)
    else:
        loss_sup = torch.tensor(0.0, device=p_emb.device, dtype=p_emb.dtype)

    fused = 0.5 * (p_emb + h_emb)
    fused_grl = grad_reverse(fused, model.cfg.grl_lambda)
    dom_logits = model.domain_head(fused_grl)
    loss_adv = F.cross_entropy(dom_logits, batch["domain_label"])

    loss_lb, lb_stats = load_balance_switch(gates_p, gates_h)
    loss_total = loss_con + (w_sup * loss_sup) + (w_adv * loss_adv) + (w_lb * loss_lb)

    gates_batch = torch.cat([gates_p, gates_h], dim=0)  # [2B,E]
    entropy_per_ex = -(gates_batch * torch.log(gates_batch + 1e-12)).sum(dim=-1)
    router_entropy_mean = entropy_per_ex.mean().detach()
    router_max_prob_mean = gates_batch.max(dim=-1).values.mean().detach()
    gates_mean = gates_batch.mean(dim=0).detach()
    router_entropy_of_mean = (-(gates_mean * torch.log(gates_mean + 1e-12)).sum()).detach()
    gates_load = lb_stats["load"].detach()

    mean_neg_score = neg_scores.float().mean().detach()

    # analysis metrics for paper
    mean_real_hardneg_sim = torch.einsum("bd,bkd->bk", p_emb, n_emb).mean().detach()

    if synth_emb is not None and synth_emb.size(1) > 0:
        mean_synth_hardneg_sim = torch.einsum("bd,bmd->bm", p_emb, synth_emb).mean().detach()
    else:
        mean_synth_hardneg_sim = torch.tensor(0.0, device=p_emb.device, dtype=p_emb.dtype)

    return {
        "loss": loss_total,
        "loss_con": loss_con.detach(),
        "loss_con_e": loss_con_e.detach(),
        "loss_con_n": loss_con_n.detach(),
        "loss_sup": loss_sup.detach(),
        "loss_adv": loss_adv.detach(),
        "loss_lb": loss_lb.detach(),

        "router_entropy_mean": router_entropy_mean,
        "router_max_prob_mean": router_max_prob_mean,
        "router_entropy_of_mean": router_entropy_of_mean,
        "gates_mean": gates_mean,
        "gates_load": gates_load,

        "mean_neg_score": mean_neg_score,
        "mean_real_hardneg_sim": mean_real_hardneg_sim,
        "mean_synth_hardneg_sim": mean_synth_hardneg_sim,
        "num_contrastive": num_contrastive,
        "num_sup_only": num_sup_only,
        "num_contrastive_entail": num_contrastive_entail,
        "num_contrastive_neutral": num_contrastive_neutral,
        "num_sup_contra": num_sup_contra,
    }
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return total, trainable, pct

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vinli_dir", type=str, required=True, help="folder containing train.jsonl/dev.jsonl/test.jsonl")
    ap.add_argument("--vianli_dir", type=str, required=True, help="folder containing train.jsonl/dev.jsonl/test.jsonl")
    ap.add_argument(
        "--data_mode",
        type=str,
        default="both",
        choices=["both", "vinli_only", "vianli_only"],
        help="Which training corpus to use"
    )
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B")

    ap.add_argument("--output_dir", type=str, default="cmoe_runs/final_hf")

    ap.add_argument("--max_len", type=int, default=128, help="Qwen tokenizer max_len")
    ap.add_argument("--viclsr_model", type=str, default="huynhtin/ViCLSR")
    ap.add_argument("--viclsr_max_len", type=int, default=128, help="ViCLSR mining max_len")
    ap.add_argument("--viclsr_batch_size", type=int, default=64)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--router_init_temp", type=float, default=1.5)
    ap.add_argument("--router_min_temp", type=float, default=0.7)
    ap.add_argument("--router_noise_std", type=float, default=0.1)
    ap.add_argument("--router_use_layernorm", action="store_true", help="enable LayerNorm on router input")
    ap.add_argument("--no_router_use_layernorm", action="store_true", help="disable LayerNorm on router input")
    ap.add_argument("--router_anneal", action="store_true", help="enable temperature annealing")
    ap.add_argument("--no_router_anneal", action="store_true", help="disable temperature annealing")

    ap.add_argument("--num_experts", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--use_qlora", action="store_true")
    ap.add_argument("--no_use_qlora", action="store_true")
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    ap.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    ap.add_argument("--no_bnb_4bit_use_double_quant", action="store_true")
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default=None)

    ap.add_argument("--w_adv", type=float, default=0.01)
    ap.add_argument("--w_lb", type=float, default=0.03)
    ap.add_argument("--grl_lambda", type=float, default=1.0)

    ap.add_argument(
        "--neutral_mode",
        type=str,
        default="hardneg",
        choices=["hardneg", "ignore", "weakpos"],
        help=(
            "hardneg: entail-only positives; negatives are non-entail (neutral+contra) in same premise. "
            "ignore: entail-only positives; negatives use contradiction only (neutral ignored). "
            "weakpos: add neutral weak-positive loss term."
        ),
    )
    ap.add_argument("--w_neutral_pos", type=float, default=0.3, help="only used when neutral_mode=weakpos")

    ap.add_argument("--num_hard_negs", type=int, default=4, help="Top-k real hard negatives per example")
    ap.add_argument("--num_synth_negs", type=int, default=2, help="Number of synthetic hard negatives per example")
    ap.add_argument("--hardneg_weight_temp", type=float, default=0.2, help="Temperature for hardness weighting")
    ap.add_argument("--hardneg_weight_clamp", type=float, default=3.0, help="Clamp max weight to avoid over-hard negatives")
    ap.add_argument("--mixup_alpha_min", type=float, default=0.3)
    ap.add_argument("--mixup_alpha_max", type=float, default=0.7)
    ap.add_argument("--use_synthetic_hardneg", action="store_true")
    ap.add_argument("--no_use_synthetic_hardneg", action="store_true")
    ap.add_argument("--use_hardneg_weighting", action="store_true", help="Enable hardness-aware weighting")
    ap.add_argument("--no_use_hardneg_weighting", action="store_true", help="Disable hardness-aware weighting")

    ap.add_argument("--reuse_mined_dataset", action="store_true", help="Reuse cached mined dataset if available")
    ap.add_argument("--no_reuse_mined_dataset", action="store_true", help="Force rebuild mined dataset")

    ap.add_argument("--mined_cache_dir", type=str, default="", help="Optional shared cache dir for mined dataset")

    # NEW: backbone unfreeze
    ap.add_argument("--unfreeze_top_layers", type=int, default=0, help="Unfreeze top N transformer layers")
    ap.add_argument("--train_final_norm", action="store_true", help="Train final LM norm")
    ap.add_argument("--no_train_final_norm", action="store_true", help="Do not train final LM norm")

    # NEW: projection head
    ap.add_argument("--use_mlp_proj", action="store_true", help="Use 2-layer MLP projection head")
    ap.add_argument("--no_use_mlp_proj", action="store_true", help="Disable MLP projection head")
    ap.add_argument("--proj_hidden_mult", type=float, default=1.0)
    ap.add_argument("--proj_dropout", type=float, default=0.1)

    # NEW: supervised auxiliary loss
    ap.add_argument("--use_sup_nli", action="store_true", help="Enable supervised NLI auxiliary head/loss")
    ap.add_argument("--no_use_sup_nli", action="store_true", help="Disable supervised NLI auxiliary head/loss")
    ap.add_argument("--w_sup", type=float, default=0.5, help="Weight for supervised NLI auxiliary loss")
    ap.add_argument("--sup_dropout", type=float, default=0.1)

    # router architecture
    ap.add_argument("--use_mlp_router", action="store_true")
    ap.add_argument("--no_use_mlp_router", action="store_true")
    ap.add_argument("--router_hidden_mult", type=float, default=0.5)
    ap.add_argument("--router_dropout", type=float, default=0.1)

    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing (may error with global gating). Default OFF.")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "last_token", "last2_mean", "first_last_mean"], help="Sentence embedding pooling used during training/inference." )
    


    args = ap.parse_args()

    if not args.use_qlora and not args.no_use_qlora:
        args.use_qlora = False
    if args.no_use_qlora:
        args.use_qlora = False

    if not args.bnb_4bit_use_double_quant and not args.no_bnb_4bit_use_double_quant:
        args.bnb_4bit_use_double_quant = True
    if args.no_bnb_4bit_use_double_quant:
        args.bnb_4bit_use_double_quant = False

    if not args.use_mlp_router and not args.no_use_mlp_router:
        args.use_mlp_router = False

    if args.no_use_mlp_router:
        args.use_mlp_router = False

    if not args.train_final_norm and not args.no_train_final_norm:
        args.train_final_norm = False
    if args.no_train_final_norm:
        args.train_final_norm = False

    if not args.use_mlp_proj and not args.no_use_mlp_proj:
        args.use_mlp_proj = False
    if args.no_use_mlp_proj:
        args.use_mlp_proj = False

    if not args.use_sup_nli and not args.no_use_sup_nli:
        args.use_sup_nli = False
    if args.no_use_sup_nli:
        args.use_sup_nli = False

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

    # Router flags: default True, allow explicit disable
    if not args.router_use_layernorm and not args.no_router_use_layernorm:
        args.router_use_layernorm = True
    if args.no_router_use_layernorm:
        args.router_use_layernorm = False

    if not args.router_anneal and not args.no_router_anneal:
        args.router_anneal = True
    if args.no_router_anneal:
        args.router_anneal = False

    if args.data_mode != "both":
        print(f"[INFO] data_mode={args.data_mode} => disable adversarial domain loss (set w_adv=0.0)")
        args.w_adv = 0.0

    print(f"[TRAIN CONFIG] data_mode={args.data_mode}, w_adv={args.w_adv}")

    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miner_device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id


    vinli_train = []
    vianli_train = []

    if args.data_mode in ["both", "vinli_only"]:
        vinli_train = load_jsonl(
            os.path.join(args.vinli_dir, "train.jsonl"),
            domain_id=0
        )

    if args.data_mode in ["both", "vianli_only"]:
        vianli_train = load_jsonl(
            os.path.join(args.vianli_dir, "train.jsonl"),
            domain_id=1
        )

    raw_train = vinli_train + vianli_train

    if len(raw_train) == 0:
        raise ValueError("No training data loaded. Check --data_mode and dataset paths.")

    print(f"[DATA] mode={args.data_mode}")
    print(f"[DATA] ViNLI train size : {len(vinli_train)}")
    print(f"[DATA] ViANLI train size: {len(vianli_train)}")
    print(f"[DATA] Total train size : {len(raw_train)}")
    print(f"[TRAIN CONFIG] data_mode={args.data_mode}, w_adv={args.w_adv}")

    # ---------------- Mined dataset cache ----------------
    if args.mined_cache_dir.strip():
        cache_root = args.mined_cache_dir
    else:
        cache_root = os.path.join(args.output_dir, "mined_dataset")

    os.makedirs(cache_root, exist_ok=True)

    sup_tag = "supnli" if args.use_sup_nli else "nosupnli"

    mined_path = os.path.join(
        cache_root,
        f"train_{args.neutral_mode}_k{args.num_hard_negs}_{sup_tag}_viclsr.jsonl"
    )

    if args.reuse_mined_dataset and os.path.exists(mined_path):
        train_data = load_processed_dataset(mined_path)
    else:
        print(f"\n[ViCLSR] Loading miner: {args.viclsr_model} on {miner_device} (max_len={args.viclsr_max_len})")
        miner = SemanticMinerViCLSR(
            model_name=args.viclsr_model,
            device=miner_device,
            max_len=args.viclsr_max_len,
            batch_size=args.viclsr_batch_size,
        )

        print(f"[ViCLSR] Mining hard negatives | neutral_mode={args.neutral_mode} | k={args.num_hard_negs} ...")
        train_data = build_same_premise_hardneg_semantic(
            raw_train,
            miner,
            seed=args.seed,
            neutral_mode=args.neutral_mode,
            num_hard_negs=args.num_hard_negs,
            keep_full_nli_for_sup=args.use_sup_nli,
        )
        print(f"[ViCLSR] Done. Kept {len(train_data)} training pairs (raw={len(raw_train)}).\n")

        with open(mined_path, "w", encoding="utf-8") as f:
            for ex in train_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"Saved processed dataset → {mined_path}")

    train_ds = NLITripletDataset(train_data, tok, max_len=args.max_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_nli_triplet_dynamic, pad_token_id=tok.pad_token_id),
    )
    cfg = CMoELoRAConfig(
        base_model=args.base_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        r=args.lora_r,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        grl_lambda=args.grl_lambda,
        temperature=0.07,

        use_qlora=args.use_qlora,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        attn_implementation=args.attn_implementation,

        router_init_temp=args.router_init_temp,
        router_min_temp=args.router_min_temp,
        router_noise_std=args.router_noise_std,
        router_use_layernorm=args.router_use_layernorm,
        router_anneal=args.router_anneal,

        emb_dim=None,
        pooling=args.pooling,

        unfreeze_top_layers=args.unfreeze_top_layers,
        train_final_norm=args.train_final_norm,

        use_mlp_proj=args.use_mlp_proj,
        proj_hidden_mult=args.proj_hidden_mult,
        proj_dropout=args.proj_dropout,

        use_sup_nli=args.use_sup_nli,
        sup_dropout=args.sup_dropout,
        use_mlp_router=args.use_mlp_router,
        router_hidden_mult=args.router_hidden_mult,
        router_dropout=args.router_dropout,
    )

    model = CMoELoRA(cfg)
    if not args.use_qlora:
        model = model.to(device)
    model.train()
    total_params, trainable_params, pct_trainable = count_parameters(model)

    if args.gradient_checkpointing:
        try:
            model.lm.model.gradient_checkpointing_enable()
            model.lm.config.use_cache = False
            print("[Info] Enabled gradient checkpointing; set use_cache=False")
        except Exception as e:
            print(f"[Warn] Could not enable gradient checkpointing: {e}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (len(train_dl) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    print("\n===== TRAIN INFO =====")
    print(f"Train pairs kept: {len(train_ds)} (mode={args.neutral_mode})")
    print(f"Mined dataset cache: {mined_path}")
    print(f"Reuse mined dataset: {args.reuse_mined_dataset}")
    print(f"use_synthetic_hardneg={args.use_synthetic_hardneg}")
    print(f"use_hardneg_weighting={args.use_hardneg_weighting}")
    print(f"num_hard_negs={args.num_hard_negs} | num_synth_negs={args.num_synth_negs}")
    print(f"unfreeze_top_layers={args.unfreeze_top_layers} | train_final_norm={args.train_final_norm}")
    print(f"use_mlp_proj={args.use_mlp_proj} | proj_hidden_mult={args.proj_hidden_mult} | proj_dropout={args.proj_dropout}")
    print(f"use_sup_nli={args.use_sup_nli} | w_sup={args.w_sup} | sup_dropout={args.sup_dropout}")
    print(f"router_type={'mlp' if args.use_mlp_router else 'linear'}")
    print(f"router_hidden_mult={args.router_hidden_mult}")
    print(f"router_dropout={args.router_dropout}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({pct_trainable:.2f}%)")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum}")
    print(f"Num batches/epoch: {len(train_dl)}")
    print(f"Epochs: {args.epochs}")
    print(f"Total optimizer steps: {total_steps} | warmup: {warmup_steps}")
    print(f"Loss: L = L_contrastive + w_sup*L_sup + w_lb*L_lb + w_adv*L_adv")
    print(f"w_lb={args.w_lb} | w_adv={args.w_adv} | grl_lambda={args.grl_lambda}")
    if args.neutral_mode == "weakpos":
        print(f"w_neutral_pos={args.w_neutral_pos}")
    print("======================\n")

    log_path = os.path.join(args.output_dir, "train_log.csv")
    csv_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)

    header = [
        "time_sec", "epoch", "step", "opt_step",
        "loss_total", "loss_con", "loss_con_e", "loss_con_n",
        "loss_adv", "loss_lb","loss_sup",
        "ema_loss", "lr", "logit_scale","router_temperature",

        "router_entropy_mean",
        "router_max_prob_mean",
        "router_entropy_of_mean",

        "num_entail",
        "num_neutral",
        "num_contra",
        "num_contrastive",
        "num_sup_only",
        "num_contrastive_entail",
        "num_contrastive_neutral",
        "num_sup_contra",

        "mean_neg_score",
        "mean_real_hardneg_sim",
        "mean_synth_hardneg_sim",

        "neutral_mode",
        "num_hard_negs",
        "num_synth_negs",
        "use_synthetic_hardneg",
        "use_hardneg_weighting",
        "hardneg_weight_temp",
        "hardneg_weight_clamp",
        "unfreeze_top_layers",
        "train_final_norm",
        "use_mlp_proj",
        "use_sup_nli",
        "w_sup",
        "router_type",
        "router_hidden_mult",
        "router_dropout",
        "total_params",
        "trainable_params",
        "pct_trainable_params",
    ]
    for i in range(args.num_experts):
        header.append(f"gates_mean_{i}")
    for i in range(args.num_experts):
        header.append(f"gates_load_{i}")

    writer.writerow(header)
    csv_f.flush()

    start_time = time.time()
    ema_loss = None
    ema_alpha = 0.05

    step = 0
    opt_step = 0
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}", leave=True)
        for batch in pbar:

            device = model.lm.model.embed_tokens.weight.device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Router temperature annealing (anneal by optimizer-step to respect grad_accum)
            if hasattr(model, "router") and getattr(args, "router_anneal", False):
                approx_opt_step = (step // max(1, args.grad_accum))
                model.router.set_temperature(step=approx_opt_step, total_steps=total_steps)

            out = training_step(
            model,
            batch,
            neutral_mode=args.neutral_mode,
            w_neutral_pos=args.w_neutral_pos,
            w_adv=args.w_adv,
            w_lb=args.w_lb,
            w_sup=args.w_sup,
            step=step,
            total_steps=total_steps,
            num_synth_negs=args.num_synth_negs,
            mixup_alpha_min=args.mixup_alpha_min,
            mixup_alpha_max=args.mixup_alpha_max,
            hardneg_weight_temp=args.hardneg_weight_temp,
            hardneg_weight_clamp=args.hardneg_weight_clamp,
            use_synthetic_hardneg=args.use_synthetic_hardneg,
            use_hardneg_weighting=args.use_hardneg_weighting,
        )

            cur_loss = float(out["loss"].item())
            ema_loss = cur_loss if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * cur_loss

            # ===== Router metrics & expert stats (from training_step output) =====
            router_entropy_mean = float(out["router_entropy_mean"].detach().cpu().item())
            router_max_prob_mean = float(out["router_max_prob_mean"].detach().cpu().item())
            entropy_of_mean = float(out["router_entropy_of_mean"].detach().cpu().item())

            gm = out["gates_mean"].float().detach().cpu()  # [E]
            gl = out["gates_load"].float().detach().cpu()  # [E]
            gm_np = gm.tolist()
            gl_np = gl.tolist()

            lr = float(opt.param_groups[0]["lr"])
            logit_scale = float(model.logit_scale.exp().detach().cpu().clamp(max=100.0).item())
            router_temperature = float(model.router.temperature)

            pl = batch["pair_label"].detach()
            num_entail = int((pl == LABEL_MAP["e"]).sum().item())
            num_neutral = int((pl == LABEL_MAP["n"]).sum().item())
            num_contra = int((pl == LABEL_MAP["c"]).sum().item())

            loss = out["loss"] / max(1, args.grad_accum)
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

            pbar.set_postfix(
                step=f"{step}",
                opt_step=f"{opt_step}/{total_steps}",
                loss=f"{cur_loss:.4f}",
                ema=f"{ema_loss:.4f}",
                con=f"{float(out['loss_con'].item()):.4f}",
                conE=f"{float(out['loss_con_e'].item()):.4f}",
                conN=f"{float(out['loss_con_n'].item()):.4f}",
                adv=f"{float(out['loss_adv'].item()):.4f}",
                lb=f"{float(out['loss_lb'].item()):.4f}",
                gmax=f"{router_max_prob_mean:.3f}",
                gent=f"{router_entropy_mean:.3f}",
                e=num_entail,
                n=num_neutral,
                c=num_contra,
                sup=f"{float(out['loss_sup'].item()):.4f}",
            )

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} opt_step={opt_step}/{total_steps} "
                    f"loss={cur_loss:.4f} ema={ema_loss:.4f} "
                    f"con={out['loss_con'].item():.4f} conE={out['loss_con_e'].item():.4f} conN={out['loss_con_n'].item():.4f} "
                    f"adv={out['loss_adv'].item():.4f} lb={out['loss_lb'].item():.4f} "
                    f"gmax={router_max_prob_mean:.3f} gent={router_entropy_mean:.3f} "
                    f"labels(e/n/c)=({num_entail}/{num_neutral}/{num_contra}) "
                    f"gates_mean={[f'{x:.3f}' for x in gm_np]} "
                    f"gates_load={[f'{x:.3f}' for x in gl_np]} "
                    f"sup={out['loss_sup'].item():.4f} "
                    
                )
            row = [
                round(time.time() - start_time, 3),
                epoch,
                step,
                opt_step,

                cur_loss,
                float(out["loss_con"].item()),
                float(out["loss_con_e"].item()),
                float(out["loss_con_n"].item()),
                float(out["loss_adv"].item()),
                float(out["loss_lb"].item()),
                float(out["loss_sup"].item()),

                float(ema_loss),
                lr,
                logit_scale,
                router_temperature,


                router_entropy_mean,
                router_max_prob_mean,
                entropy_of_mean,

                num_entail,
                num_neutral,
                num_contra,
                int(out["num_contrastive"].detach().cpu().item()),
                int(out["num_sup_only"].detach().cpu().item()),
                int(out["num_contrastive_entail"].detach().cpu().item()),
                int(out["num_contrastive_neutral"].detach().cpu().item()),
                int(out["num_sup_contra"].detach().cpu().item()),

                float(out["mean_neg_score"].detach().cpu().item()),
                float(out["mean_real_hardneg_sim"].detach().cpu().item()),
                float(out["mean_synth_hardneg_sim"].detach().cpu().item()),

                args.neutral_mode,
                args.num_hard_negs,
                args.num_synth_negs,
                int(args.use_synthetic_hardneg),
                int(args.use_hardneg_weighting),
                args.hardneg_weight_temp,
                args.hardneg_weight_clamp,
                args.unfreeze_top_layers,
                int(args.train_final_norm),
                int(args.use_mlp_proj),
                int(args.use_sup_nli),
                args.w_sup,
                "mlp" if args.use_mlp_router else "linear",
                args.router_hidden_mult,
                args.router_dropout,
                total_params,
                trainable_params,
                pct_trainable,
            ] + [float(x) for x in gm_np] + [float(x) for x in gl_np]
            writer.writerow(row)

            if step % args.log_every == 0:
                csv_f.flush()

            step += 1

    csv_f.flush()
    csv_f.close()
    print(f"\nSaved train log to: {log_path}")
    extra_config = dict(vars(args))
    extra_config["total_params"] = total_params
    extra_config["trainable_params"] = trainable_params
    extra_config["pct_trainable_params"] = pct_trainable

    model.save_pretrained(args.output_dir, tokenizer=tok, extra_config=extra_config)
    print(f"\nSaved HF-style custom checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()