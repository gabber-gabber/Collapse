from __future__ import annotations

import os

from .types import DefenseSpec, FineTuneMode, ModelSpec


# Local fine-tuned victim checkpoints (Qwen GLUE seed42 runs) live outside the
# repo. Set CCS_VICTIM_MODEL_ROOT to the directory that contains the
# `qwen25_0p5b/{mnli,qnli,sst2}/seed42` and `qwen25_1p5b/...` subtrees, e.g.
#     export CCS_VICTIM_MODEL_ROOT=/path/to/glue_outputs
# When unset, the relative path is used as-is (suitable for an artifact
# reviewer who places the checkpoints next to the working directory).
_VICTIM_ROOT = os.environ.get("CCS_VICTIM_MODEL_ROOT", "")


def _local_victim(rel_path: str) -> str:
    return os.path.join(_VICTIM_ROOT, rel_path) if _VICTIM_ROOT else rel_path


MODEL_SPECS = {
    "bert_base": ModelSpec(
        key="bert_base",
        family="bert",
        victim_checkpoint="yoshitomo-matsubara/bert-base-uncased-mnli",
        public_checkpoint="bert-base-uncased",
        target_suffixes=(
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
            "attention.output.dense.weight",
            "intermediate.dense.weight",
            "output.dense.weight",
        ),
    ),
    "qwen_base": ModelSpec(
        key="qwen_base",
        family="qwen",
        victim_checkpoint=_local_victim("qwen25_0p5b/mnli/seed42"),
        public_checkpoint="Qwen/Qwen2.5-0.5B",
        target_suffixes=(
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ),
    ),
    "qwen_large": ModelSpec(
        key="qwen_large",
        family="qwen",
        victim_checkpoint=_local_victim("qwen25_1p5b/mnli/seed42"),
        public_checkpoint="Qwen/Qwen2.5-1.5B",
        target_suffixes=(
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ),
    ),
    "vit_base": ModelSpec(
        key="vit_base",
        family="vit",
        victim_checkpoint="avanishd/vit-base-patch16-224-in21k-finetuned-cifar100",
        public_checkpoint="google/vit-base-patch16-224-in21k",
        target_suffixes=(
            "attention.attention.query.weight",
            "attention.attention.key.weight",
            "attention.attention.value.weight",
            "attention.output.dense.weight",
            "intermediate.dense.weight",
            "output.dense.weight",
        ),
    ),
    "qwen_base_gsm8k": ModelSpec(
        key="qwen_base_gsm8k",
        family="qwen",
        victim_checkpoint="LahiruWije/Qwen2.5-0.5B-Instruct-GPRO-GSM8K",
        public_checkpoint="Qwen/Qwen2.5-0.5B",
        target_suffixes=(
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ),
    ),
    "qwen_large_gsm8k": ModelSpec(
        key="qwen_large_gsm8k",
        family="qwen",
        victim_checkpoint="YWZBrandon/openai-gsm8k_Qwen-Qwen2.5-1.5B_full_sft_2e-6",
        public_checkpoint="Qwen/Qwen2.5-1.5B",
        target_suffixes=(
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ),
    ),
}


DEFENSE_SPECS = {
    "arrowcloak": DefenseSpec(
        key="arrowcloak",
        observations=10,
        needs_alignment=True,
        needs_mask_recovery=True,
        fine_tune_mode=FineTuneMode.DIRECTION_ONLY,
        mask_rank=1,
        alignment_strength="default",
        seed_slot=5,
        attack_prior_source="blackbox",
        attack_loader="train",
        attack_lr=5e-4,
        attack_epochs=1,
        weight_scope="linear",
        postprocess="reset_tsqp_lengths",
    ),
    "loro": DefenseSpec(
        key="loro",
        observations=4,
        needs_alignment=False,
        needs_mask_recovery=True,
        fine_tune_mode=FineTuneMode.FULL,
        seed_slot=1,
    ),
    "nnsplitter": DefenseSpec(
        key="nnsplitter",
        observations=1,
        needs_alignment=False,
        needs_mask_recovery=False,
        fine_tune_mode=FineTuneMode.MASKED_ONLY,
        seed_slot=0,
    ),
    "translinkguard": DefenseSpec(
        key="translinkguard",
        observations=1,
        needs_alignment=True,
        needs_mask_recovery=False,
        fine_tune_mode=FineTuneMode.FULL,
        seed_slot=3,
        attack_loader="recover",
        attack_lr=1e-5,
        attack_epochs=3,
    ),
    "tsqp": DefenseSpec(
        key="tsqp",
        observations=1,
        needs_alignment=False,
        needs_mask_recovery=False,
        fine_tune_mode=FineTuneMode.LENGTH_ONLY,
        seed_slot=2,
    ),
    "oprior": DefenseSpec(
        key="oprior",
        observations=10,
        needs_alignment=True,
        needs_mask_recovery=True,
        fine_tune_mode=FineTuneMode.LENGTH_AND_MASKED,
        mask_rank=2,
        alignment_strength="strong",
        seed_slot=5,
        attack_prior_source="blackbox",
        attack_loader="train",
        attack_lr=5e-4,
        attack_epochs=1,
        weight_scope="linear",
        postprocess="reset_tsqp_lengths",
        sparse_ratio=0.01,
        sparse_strength=0.6,
        static_sparse_seed_offset=4242,
    ),
    "ourdefense": DefenseSpec(
        key="ourdefense",
        observations=4,
        needs_alignment=True,
        needs_mask_recovery=True,
        fine_tune_mode=FineTuneMode.LENGTH_ONLY,
        alignment_strength="strong",
    ),
}


EXPERIMENT_MATRIX = {
    "bert_base": ("mnli", "qnli", "sst2"),
    "qwen_base": ("mnli", "qnli", "sst2"),
    "qwen_large": ("mnli", "qnli", "sst2"),
    "vit_base": ("cifar10", "cifar100"),
    "qwen_base_gsm8k": ("gsm8k",),
    "qwen_large_gsm8k": ("gsm8k",),
}


DATASET_MODEL_DEFAULTS = {
    ("bert_base", "mnli"): {
        "victim_checkpoint": "yoshitomo-matsubara/bert-base-uncased-mnli",
        "public_checkpoint": "bert-base-uncased",
        "attack_lr": 1e-5,
    },
    ("bert_base", "qnli"): {
        "victim_checkpoint": "textattack/bert-base-uncased-QNLI",
        "public_checkpoint": "bert-base-uncased",
        "attack_lr": 1e-5,
    },
    ("bert_base", "sst2"): {
        "victim_checkpoint": "textattack/bert-base-uncased-SST-2",
        "public_checkpoint": "bert-base-uncased",
        "attack_lr": 1e-5,
    },
    ("qwen_base", "mnli"): {
        "victim_checkpoint": _local_victim("qwen25_0p5b/mnli/seed42"),
        "public_checkpoint": "Qwen/Qwen2.5-0.5B",
        "attack_lr": 1e-5,
    },
    ("qwen_base", "qnli"): {
        "victim_checkpoint": _local_victim("qwen25_0p5b/qnli/seed42"),
        "public_checkpoint": "Qwen/Qwen2.5-0.5B",
        "attack_lr": 1e-5,
    },
    ("qwen_base", "sst2"): {
        "victim_checkpoint": _local_victim("qwen25_0p5b/sst2/seed42"),
        "public_checkpoint": "Qwen/Qwen2.5-0.5B",
        "attack_lr": 1e-5,
    },
    ("qwen_large", "mnli"): {
        "victim_checkpoint": _local_victim("qwen25_1p5b/mnli/seed42"),
        "public_checkpoint": "Qwen/Qwen2.5-1.5B",
        "attack_lr": 1e-5,
    },
    ("qwen_large", "qnli"): {
        "victim_checkpoint": _local_victim("qwen25_1p5b/qnli/seed42"),
        "public_checkpoint": "Qwen/Qwen2.5-1.5B",
        "attack_lr": 1e-5,
    },
    ("qwen_large", "sst2"): {
        "victim_checkpoint": _local_victim("qwen25_1p5b/sst2/seed42"),
        "public_checkpoint": "Qwen/Qwen2.5-1.5B",
        "attack_lr": 1e-5,
    },
    ("vit_base", "cifar10"): {
        "victim_checkpoint": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "public_checkpoint": "google/vit-base-patch16-224-in21k",
        "attack_lr": 1e-5,
    },
    ("vit_base", "cifar100"): {
        "victim_checkpoint": "avanishd/vit-base-patch16-224-in21k-finetuned-cifar100",
        "public_checkpoint": "google/vit-base-patch16-224-in21k",
        "attack_lr": 1e-5,
    },
    ("qwen_base_gsm8k", "gsm8k"): {
        "victim_checkpoint": "LahiruWije/Qwen2.5-0.5B-Instruct-GPRO-GSM8K",
        "public_checkpoint": "Qwen/Qwen2.5-0.5B",
        "attack_lr": 1e-5,
    },
    ("qwen_large_gsm8k", "gsm8k"): {
        "victim_checkpoint": "YWZBrandon/openai-gsm8k_Qwen-Qwen2.5-1.5B_full_sft_2e-6",
        "public_checkpoint": "Qwen/Qwen2.5-1.5B",
        "attack_lr": 1e-5,
    },
}


DEFENSE_EXPERIMENT_OVERRIDES = {
    ("qwen_base", "mnli", "arrowcloak"): {"batch_size": 8},
    ("qwen_base", "mnli", "oprior"): {"batch_size": 8},
    ("qwen_base", "qnli", "arrowcloak"): {"batch_size": 8},
    ("qwen_base", "qnli", "oprior"): {"batch_size": 8},
    ("qwen_base", "sst2", "arrowcloak"): {"batch_size": 8},
    ("qwen_base", "sst2", "oprior"): {"batch_size": 8},
    ("qwen_large", "mnli", "arrowcloak"): {"batch_size": 8},
    ("qwen_large", "mnli", "oprior"): {"batch_size": 8},
    ("qwen_large", "qnli", "arrowcloak"): {"batch_size": 8},
    ("qwen_large", "qnli", "oprior"): {"batch_size": 8},
    ("qwen_large", "sst2", "arrowcloak"): {"batch_size": 8},
    ("qwen_large", "sst2", "oprior"): {"batch_size": 8},
    ("qwen_base_gsm8k", "gsm8k", "arrowcloak"): {"batch_size": 4},
    ("qwen_base_gsm8k", "gsm8k", "oprior"): {"batch_size": 4},
    ("qwen_large_gsm8k", "gsm8k", "arrowcloak"): {"batch_size": 2},
    ("qwen_large_gsm8k", "gsm8k", "oprior"): {"batch_size": 2},
    ("qwen_large_gsm8k", "gsm8k", "nnsplitter"): {"batch_size": 2},
    ("qwen_large_gsm8k", "gsm8k", "loro"): {"batch_size": 2},
    ("qwen_large_gsm8k", "gsm8k", "tsqp"): {"batch_size": 2},
    ("qwen_large_gsm8k", "gsm8k", "translinkguard"): {"batch_size": 2},
}


DEFENSE_SPEC_OVERRIDES = {
    ("bert_base", "mnli", "oprior"): {
        "mask_rank": 1,
        "attack_lr": 5e-5,
    },
    ("bert_base", "mnli", "translinkguard"): {
        "attack_lr": 1e-6,
        "attack_epochs": 1,
    },
    ("bert_base", "qnli", "oprior"): {
        "mask_rank": 1,
        "attack_lr": 5e-5,
    },
    ("bert_base", "sst2", "oprior"): {
        "mask_rank": 1,
        "attack_lr": 5e-5,
    },
    ("qwen_base", "mnli", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_base", "qnli", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_base", "sst2", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_large", "mnli", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_large", "qnli", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_large", "sst2", "oprior"): {
        "mask_rank": 1,
    },
    ("vit_base", "cifar100", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_base_gsm8k", "gsm8k", "oprior"): {
        "mask_rank": 1,
    },
    ("qwen_large_gsm8k", "gsm8k", "oprior"): {
        "mask_rank": 1,
    },
}
