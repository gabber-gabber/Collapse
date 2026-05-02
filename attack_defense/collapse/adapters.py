from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForSequenceClassification, AutoTokenizer

from .datasets import (
    DATASET_SPECS,
    GSM8K_MAX_NEW_TOKENS,
    build_cifar_dataloaders,
    build_glue_dataloaders,
    build_gsm8k_dataloaders,
    gsm8k_extract_pred,
)
from .finetune import apply_constraints_to_tensor_map
from .types import ExperimentSpec, FineTuneMode, ModelSpec, TaskAdapter


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseHFAdapter(TaskAdapter):
    def __init__(self, spec: ModelSpec, experiment: ExperimentSpec, prefer_device: str = "cuda") -> None:
        self.spec = spec
        self.experiment = experiment
        self.dataset_spec = DATASET_SPECS[experiment.dataset_key]
        self.device = get_device(prefer_device)
        self.train_loader = None
        self.eval_loader = None
        set_seed(experiment.seed)

    def _iter_linear_modules(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                yield name, module

    def collect_target_weights(self, model) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, module in self._iter_linear_modules(model):
            weight_name = f"{name}.weight"
            if not weight_name.endswith(self.spec.target_suffixes):
                continue
            out[weight_name] = module.weight.data.detach().cpu().clone()
        return out

    def collect_linear_weights(self, model) -> Dict[str, torch.Tensor]:
        return {f"{name}.weight": module.weight.data.detach().cpu().clone() for name, module in self._iter_linear_modules(model)}

    def set_target_weights(self, model, weights: Mapping[str, torch.Tensor]) -> None:
        for name, module in self._iter_linear_modules(model):
            weight_name = f"{name}.weight"
            if weight_name not in weights:
                continue
            weight = weights[weight_name]
            if tuple(weight.shape) != tuple(module.weight.data.shape):
                continue
            module.weight.data = weight.to(device=module.weight.data.device, dtype=module.weight.data.dtype)

    def set_linear_weights(self, model, weights: Mapping[str, torch.Tensor]) -> None:
        for name, module in self._iter_linear_modules(model):
            weight_name = f"{name}.weight"
            if weight_name not in weights:
                continue
            weight = weights[weight_name]
            if tuple(weight.shape) != tuple(module.weight.data.shape):
                continue
            module.weight.data = weight.to(device=module.weight.data.device, dtype=module.weight.data.dtype)

    def evaluate(self, model) -> float:
        model.eval()
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                labels = batch["labels"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_count += labels.numel()
        return 100.0 * total_correct / max(total_count, 1)

    def _train_model(self, model, loader: DataLoader, lr: float, epochs: int) -> None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            return
        optimizer = AdamW(trainable, lr=lr)
        for _ in range(epochs):
            model.train()
            for batch in loader:
                labels = batch["labels"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs, labels=labels)
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    def _train_model_direction_only_generic(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
        trainable_filter=None,
        grad_clip: float | None = None,
    ) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for name, module in self._iter_linear_modules(model):
            weight_name = f"{name}.weight"
            if weight_name in direction_by_layer and (trainable_filter is None or trainable_filter(weight_name)):
                module.weight.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            return

        optimizer = AdamW(trainable, lr=lr)
        dirs_dev = {k: v.to(self.device) for k, v in direction_by_layer.items()}

        for _ in range(epochs):
            model.train()
            for batch in loader:
                labels = batch["labels"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs, labels=labels)
                outputs.loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
                optimizer.step()
                with torch.no_grad():
                    for name, module in self._iter_linear_modules(model):
                        weight_name = f"{name}.weight"
                        if weight_name not in dirs_dev:
                            continue
                        if trainable_filter is not None and not trainable_filter(weight_name):
                            continue
                        d = dirs_dev[weight_name]
                        s = torch.sum(module.weight.data * d, dim=0)
                        module.weight.data = d * s.unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)

    def _train_model_length_and_masked_generic(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
        sparse_masks: Mapping[str, torch.Tensor],
        trainable_filter=None,
        grad_clip: float | None = None,
    ) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for name, module in self._iter_linear_modules(model):
            weight_name = f"{name}.weight"
            if weight_name in direction_by_layer and (trainable_filter is None or trainable_filter(weight_name)):
                module.weight.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            return

        optimizer = AdamW(trainable, lr=lr)
        dirs_dev = {k: v.to(self.device) for k, v in direction_by_layer.items()}
        masks_dev = {k: v.to(self.device) for k, v in sparse_masks.items()}

        for _ in range(epochs):
            model.train()
            for batch in loader:
                labels = batch["labels"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs, labels=labels)
                outputs.loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
                optimizer.step()
                with torch.no_grad():
                    for name, module in self._iter_linear_modules(model):
                        weight_name = f"{name}.weight"
                        if weight_name not in dirs_dev:
                            continue
                        if trainable_filter is not None and not trainable_filter(weight_name):
                            continue
                        d = dirs_dev[weight_name]
                        mask = masks_dev.get(weight_name, torch.zeros_like(d, dtype=torch.bool))
                        w = module.weight.data
                        keep = (~mask).to(dtype=w.dtype)
                        num = torch.sum((w * d) * keep, dim=0)
                        den = torch.sum((d * d) * keep, dim=0).clamp_min(1e-8)
                        s = num / den
                        projected = d * s.unsqueeze(0)
                        module.weight.data = torch.where(mask, w, projected)
                optimizer.zero_grad(set_to_none=True)

    def _train_model_direction_only_bert(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
    ) -> None:
        self._train_model_direction_only_generic(model, loader, lr, epochs, direction_by_layer)

    def _train_model_direction_only_vit(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
    ) -> None:
        self._train_model_direction_only_generic(model, loader, lr, epochs, direction_by_layer)

    def _qwen_trainable_weight(self, weight_name: str) -> bool:
        return ".self_attn.k_proj.weight" not in weight_name and ".self_attn.v_proj.weight" not in weight_name

    def _train_model_direction_only_qwen(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
    ) -> None:
        effective_lr = min(lr, 1e-4)
        self._train_model_direction_only_scale_only_qwen(model, loader, effective_lr, epochs, direction_by_layer)

    def _train_model_direction_only_scale_only_qwen(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
    ) -> None:
        for param in model.parameters():
            param.requires_grad = False

        patched: list[tuple[torch.nn.Linear, object, torch.Tensor, torch.nn.Parameter]] = []
        trainable_scales: list[torch.nn.Parameter] = []

        def _make_forward(direction: torch.Tensor, scale: torch.nn.Parameter, bias):
            def _forward(input):
                return F.linear(input, direction * scale.unsqueeze(0), bias)
            return _forward

        try:
            for name, module in self._iter_linear_modules(model):
                weight_name = f"{name}.weight"
                if weight_name not in direction_by_layer or not self._qwen_trainable_weight(weight_name):
                    continue
                direction = direction_by_layer[weight_name].to(device=self.device, dtype=module.weight.data.dtype)
                init_scale = torch.sum(module.weight.data * direction, dim=0).detach()
                scale = torch.nn.Parameter(init_scale.clone())
                original_forward = module.forward
                module.forward = _make_forward(direction, scale, module.bias)
                patched.append((module, original_forward, direction, scale))
                trainable_scales.append(scale)

            if not trainable_scales:
                return

            optimizer = AdamW(trainable_scales, lr=lr, weight_decay=0.0)
            for _ in range(epochs):
                model.train()
                for batch in loader:
                    labels = batch["labels"].to(self.device)
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    outputs = model(**inputs, labels=labels)
                    outputs.loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_scales, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        finally:
            with torch.no_grad():
                for module, original_forward, direction, scale in patched:
                    module.weight.data = direction * scale.unsqueeze(0)
                    module.forward = original_forward

    def _train_model_length_and_masked_bert(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
        sparse_masks: Mapping[str, torch.Tensor],
    ) -> None:
        self._train_model_length_and_masked_generic(model, loader, lr, epochs, direction_by_layer, sparse_masks)

    def _train_model_length_and_masked_vit(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
        sparse_masks: Mapping[str, torch.Tensor],
    ) -> None:
        self._train_model_length_and_masked_generic(model, loader, lr, epochs, direction_by_layer, sparse_masks)

    def _train_model_length_and_masked_qwen(
        self,
        model,
        loader: DataLoader,
        lr: float,
        epochs: int,
        direction_by_layer: Mapping[str, torch.Tensor],
        sparse_masks: Mapping[str, torch.Tensor],
    ) -> None:
        effective_lr = min(lr, 5e-5)
        self._train_model_length_and_masked_generic(
            model,
            loader,
            effective_lr,
            epochs,
            direction_by_layer,
            sparse_masks,
            trainable_filter=self._qwen_trainable_weight,
            grad_clip=1.0,
        )

    def run_blackbox_finetune(self, model) -> None:
        for p in model.parameters():
            p.requires_grad = True
        self._train_model(model, self.train_loader, self.experiment.blackbox_lr, self.experiment.blackbox_epochs)

    def run_attack_finetune(
        self,
        model,
        mode: FineTuneMode,
        sparse_masks: Optional[Mapping[str, torch.Tensor]] = None,
        direction_by_layer: Optional[Mapping[str, torch.Tensor]] = None,
        loader_kind: str = "recover",
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
    ) -> None:
        original = self.collect_linear_weights(model)
        loader = self.train_loader if loader_kind == "train" else self._build_recover_loader()
        lr = self.experiment.attack_lr if lr is None else lr
        epochs = self.experiment.attack_epochs if epochs is None else epochs

        if mode == FineTuneMode.DIRECTION_ONLY:
            if not direction_by_layer:
                return
            if self.spec.family == "bert":
                self._train_model_direction_only_bert(model, loader, lr, epochs, direction_by_layer)
            elif self.spec.family == "qwen":
                self._train_model_direction_only_qwen(model, loader, lr, epochs, direction_by_layer)
            elif self.spec.family == "vit":
                self._train_model_direction_only_vit(model, loader, lr, epochs, direction_by_layer)
            else:
                self._train_model_direction_only_generic(model, loader, lr, epochs, direction_by_layer)
            return
        if mode == FineTuneMode.LENGTH_AND_MASKED and direction_by_layer:
            if self.spec.family == "bert":
                self._train_model_length_and_masked_bert(model, loader, lr, epochs, direction_by_layer, sparse_masks or {})
            elif self.spec.family == "qwen":
                self._train_model_length_and_masked_qwen(model, loader, lr, epochs, direction_by_layer, sparse_masks or {})
            elif self.spec.family == "vit":
                self._train_model_length_and_masked_vit(model, loader, lr, epochs, direction_by_layer, sparse_masks or {})
            else:
                self._train_model_length_and_masked_generic(model, loader, lr, epochs, direction_by_layer, sparse_masks or {})
            return

        for p in model.parameters():
            p.requires_grad = True
        self._train_model(model, loader, lr, epochs)
        updated = self.collect_target_weights(model)
        constrained = apply_constraints_to_tensor_map(original, updated, mode, sparse_masks=sparse_masks)
        self.set_linear_weights(model, constrained)

    def _build_recover_loader(self) -> DataLoader:
        dataset = self.train_loader.dataset
        total = len(dataset)
        take = max(1, int(total * max(1e-4, self.experiment.recover_ratio)))
        generator = torch.Generator()
        generator.manual_seed(self.experiment.seed + 7)
        indices = torch.randperm(total, generator=generator)[:take].tolist()
        subset = dataset.select(indices) if hasattr(dataset, "select") else Subset(dataset, indices)
        return DataLoader(subset, batch_size=self.train_loader.batch_size, shuffle=True, collate_fn=self.train_loader.collate_fn)

    def baseline_report(self) -> Dict[str, float]:
        victim = self.load_victim_model()
        public = self.load_public_model()
        whitebox = self.evaluate(victim)
        self.run_blackbox_finetune(public)
        blackbox = self.evaluate(public)
        return {"whitebox_acc": whitebox, "blackbox_acc": blackbox}


class TextClassificationAdapter(BaseHFAdapter):
    def __init__(self, spec: ModelSpec, experiment: ExperimentSpec, prefer_device: str = "cuda") -> None:
        super().__init__(spec, experiment, prefer_device=prefer_device)
        self.tokenizer = AutoTokenizer.from_pretrained(experiment.victim_checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.train_loader, self.eval_loader = build_glue_dataloaders(
            dataset_key=experiment.dataset_key,
            tokenizer=self.tokenizer,
            train_samples=experiment.train_samples or self.dataset_spec.train_samples,
            eval_samples=experiment.eval_samples or self.dataset_spec.eval_samples,
            batch_size=experiment.batch_size or self.dataset_spec.batch_size,
            seed=experiment.seed,
            target_label2id=self._target_label2id(),
        )

    def _target_label2id(self) -> Dict[str, int]:
        if self.experiment.dataset_key == "mnli":
            return {"entailment": 0, "neutral": 1, "contradiction": 2}
        if self.experiment.dataset_key in {"qnli", "sst2"}:
            return {"entailment": 0, "not_entailment": 1, "negative": 0, "positive": 1}
        return {}

    def load_victim_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.experiment.victim_checkpoint).to(self.device)
        if self.tokenizer.pad_token_id is not None and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def load_public_model(self):
        kwargs = {"num_labels": self.dataset_spec.num_labels}
        label2id = self._target_label2id()
        if label2id:
            kwargs["label2id"] = label2id
            kwargs["id2label"] = {int(v): k for k, v in label2id.items()}
        model = AutoModelForSequenceClassification.from_pretrained(self.experiment.public_checkpoint, **kwargs).to(self.device)
        if self.tokenizer.pad_token_id is not None and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = self.tokenizer.pad_token_id
        return model


class ImageClassificationAdapter(BaseHFAdapter):
    def __init__(self, spec: ModelSpec, experiment: ExperimentSpec, prefer_device: str = "cuda") -> None:
        super().__init__(spec, experiment, prefer_device=prefer_device)
        self.image_processor = AutoImageProcessor.from_pretrained(experiment.victim_checkpoint)
        self.train_loader, self.eval_loader = build_cifar_dataloaders(
            dataset_key=experiment.dataset_key,
            image_processor=self.image_processor,
            train_samples=experiment.train_samples or self.dataset_spec.train_samples,
            eval_samples=experiment.eval_samples or self.dataset_spec.eval_samples,
            batch_size=experiment.batch_size or self.dataset_spec.batch_size,
            seed=experiment.seed,
            target_label2id={},
        )

    def load_victim_model(self):
        return AutoModelForImageClassification.from_pretrained(self.experiment.victim_checkpoint).to(self.device)

    def load_public_model(self):
        return AutoModelForImageClassification.from_pretrained(
            self.experiment.public_checkpoint,
            num_labels=self.dataset_spec.num_labels,
        ).to(self.device)


class CausalLMGSM8KAdapter(BaseHFAdapter):
    def __init__(self, spec: ModelSpec, experiment: ExperimentSpec, prefer_device: str = "cuda") -> None:
        super().__init__(spec, experiment, prefer_device=prefer_device)
        self.tokenizer = AutoTokenizer.from_pretrained(experiment.victim_checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_loader, self.eval_loader = build_gsm8k_dataloaders(
            tokenizer=self.tokenizer,
            train_samples=experiment.train_samples or self.dataset_spec.train_samples,
            eval_samples=experiment.eval_samples or self.dataset_spec.eval_samples,
            batch_size=experiment.batch_size or self.dataset_spec.batch_size,
            seed=experiment.seed,
        )

    def _finalize_lm_model(self, model):
        if self.tokenizer.pad_token_id is not None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.use_cache = False
        return model.to(self.device)

    def load_victim_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.experiment.victim_checkpoint, torch_dtype=torch.float32)
        return self._finalize_lm_model(model)

    def load_public_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.experiment.public_checkpoint, torch_dtype=torch.float32)
        return self._finalize_lm_model(model)

    def evaluate(self, model) -> float:
        model.eval()
        prev_use_cache = getattr(model.config, "use_cache", False)
        model.config.use_cache = True
        total_correct = 0
        total_count = 0
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        try:
            with torch.no_grad():
                for batch in self.eval_loader:
                    prompt_ids = batch["prompt_ids"].to(self.device)
                    attn = batch["prompt_attention_mask"].to(self.device)
                    golds = batch["gold"]
                    gen = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attn,
                        max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )
                    new_tokens = gen[:, prompt_ids.shape[1]:]
                    decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    for pred_text, gold in zip(decoded, golds):
                        pred = gsm8k_extract_pred(pred_text)
                        if gold and pred == gold:
                            total_correct += 1
                        total_count += 1
        finally:
            model.config.use_cache = prev_use_cache
        return 100.0 * total_correct / max(total_count, 1)


def build_adapter(model_spec: ModelSpec, experiment: ExperimentSpec, prefer_device: str = "cuda") -> TaskAdapter:
    family = DATASET_SPECS[experiment.dataset_key].family
    if family == "text":
        return TextClassificationAdapter(model_spec, experiment, prefer_device=prefer_device)
    if family == "image":
        return ImageClassificationAdapter(model_spec, experiment, prefer_device=prefer_device)
    if family == "causal_lm":
        return CausalLMGSM8KAdapter(model_spec, experiment, prefer_device=prefer_device)
    raise NotImplementedError(f"Dataset family not implemented yet: {experiment.dataset_key}")
