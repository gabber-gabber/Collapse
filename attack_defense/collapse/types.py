from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Protocol

import torch


TensorMap = Dict[str, torch.Tensor]


class FineTuneMode(str, Enum):
    FULL = "full"
    LENGTH_ONLY = "length_only"
    MASKED_ONLY = "masked_only"
    LENGTH_AND_MASKED = "length_and_masked"
    DIRECTION_ONLY = "direction_only"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    victim_checkpoint: str
    public_checkpoint: str
    target_suffixes: tuple[str, ...]
    permutation_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class DefenseSpec:
    key: str
    observations: int
    needs_alignment: bool
    needs_mask_recovery: bool
    fine_tune_mode: FineTuneMode
    mask_rank: int = 1
    alignment_strength: str = "default"
    seed_slot: int = 0
    attack_prior_source: str = "public"
    attack_loader: str = "recover"
    attack_lr: Optional[float] = None
    attack_epochs: Optional[int] = None
    weight_scope: str = "target"
    postprocess: str = "none"
    sparse_ratio: Optional[float] = None
    sparse_strength: Optional[float] = None
    static_sparse_seed_offset: int = 0


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    family: str
    hf_name: str
    hf_subset: Optional[str]
    num_labels: int
    batch_size: int
    train_samples: int
    eval_samples: int


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    model_key: str
    dataset_key: str
    victim_checkpoint: str
    public_checkpoint: str
    seed: int = 42
    attack_views: int = 4
    blackbox_epochs: int = 1
    blackbox_lr: float = 2e-5
    attack_epochs: int = 1
    attack_lr: float = 1e-5
    recover_ratio: float = 0.01
    train_samples: Optional[int] = None
    eval_samples: Optional[int] = None
    batch_size: Optional[int] = None
    output_root: str = "results"
    log_root: str = "logs"


@dataclass
class MatrixArtifacts:
    obfuscated: torch.Tensor
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class RecoveryArtifacts:
    recovered: torch.Tensor
    permutation: Optional[torch.Tensor] = None
    sparse_mask: Optional[torch.Tensor] = None
    sparse_value: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    whitebox_acc: float
    blackbox_acc: float
    obfuscated_acc: float
    attack_acc: float
    recovered_layers: int
    extra: Dict[str, object] = field(default_factory=dict)


    def as_dict(self) -> Dict[str, object]:
        return {
            "whitebox_acc": self.whitebox_acc,
            "blackbox_acc": self.blackbox_acc,
            "obfuscated_acc": self.obfuscated_acc,
            "attack_acc": self.attack_acc,
            "recovered_layers": self.recovered_layers,
            **self.extra,
        }


class TaskAdapter(Protocol):
    spec: ModelSpec

    def load_victim_model(self): ...

    def load_public_model(self): ...

    def collect_target_weights(self, model) -> TensorMap: ...

    def collect_linear_weights(self, model) -> TensorMap: ...

    def set_target_weights(self, model, weights: Mapping[str, torch.Tensor]) -> None: ...

    def set_linear_weights(self, model, weights: Mapping[str, torch.Tensor]) -> None: ...

    def evaluate(self, model) -> float: ...

    def run_blackbox_finetune(self, model) -> None: ...

    def run_attack_finetune(
        self,
        model,
        mode: FineTuneMode,
        sparse_masks: Optional[Mapping[str, torch.Tensor]] = None,
        direction_by_layer: Optional[Mapping[str, torch.Tensor]] = None,
        loader_kind: str = "recover",
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
    ) -> None: ...

    def baseline_report(self) -> Dict[str, float]: ...


ObfuscateFn = Callable[[torch.Tensor, int], MatrixArtifacts]
RecoverFn = Callable[..., RecoveryArtifacts]
MatrixObfuscator = Callable[[TensorMap, int], Dict[str, MatrixArtifacts]]
MatrixAttacker = Callable[[List[TensorMap], TensorMap, ModelSpec], Dict[str, RecoveryArtifacts]]


class LegacyReference(Protocol):
    def whitebox_accuracy(self, target_key: str) -> float: ...

    def blackbox_accuracy(self, target_key: str) -> float: ...
