from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import datasets as hf_datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .types import DatasetSpec


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "mnli": DatasetSpec(
        key="mnli",
        family="text",
        hf_name="glue",
        hf_subset="mnli",
        num_labels=3,
        batch_size=32,
        train_samples=3072,
        eval_samples=1024,
    ),
    "qnli": DatasetSpec(
        key="qnli",
        family="text",
        hf_name="glue",
        hf_subset="qnli",
        num_labels=2,
        batch_size=32,
        train_samples=768,
        eval_samples=512,
    ),
    "sst2": DatasetSpec(
        key="sst2",
        family="text",
        hf_name="glue",
        hf_subset="sst2",
        num_labels=2,
        batch_size=32,
        train_samples=512,
        eval_samples=512,
    ),
    "cifar10": DatasetSpec(
        key="cifar10",
        family="image",
        hf_name="cifar10",
        hf_subset=None,
        num_labels=10,
        batch_size=32,
        train_samples=10000,
        eval_samples=2000,
    ),
    "cifar100": DatasetSpec(
        key="cifar100",
        family="image",
        hf_name="cifar100",
        hf_subset=None,
        num_labels=100,
        batch_size=32,
        train_samples=2000,
        eval_samples=5000,
    ),
    "gsm8k": DatasetSpec(
        key="gsm8k",
        family="causal_lm",
        hf_name="gsm8k",
        hf_subset="main",
        num_labels=0,
        batch_size=4,
        train_samples=512,
        eval_samples=200,
    ),
}


GSM8K_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_GSM8K_LAST_NUMBER_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")
GSM8K_MAX_PROMPT_LEN = 384
GSM8K_MAX_SEQ_LEN = 512
GSM8K_MAX_NEW_TOKENS = 256


def gsm8k_extract_gold(answer_text: str) -> str:
    match = GSM8K_ANSWER_RE.search(answer_text or "")
    if not match:
        return ""
    return match.group(1).replace(",", "").rstrip(".")


def gsm8k_extract_pred(generated_text: str) -> str:
    if not generated_text:
        return ""
    match = GSM8K_ANSWER_RE.search(generated_text)
    if match:
        return match.group(1).replace(",", "").rstrip(".")
    numbers = _GSM8K_LAST_NUMBER_RE.findall(generated_text)
    if not numbers:
        return ""
    return numbers[-1].replace(",", "").rstrip(".")


def _gsm8k_format_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _gsm8k_format_full(question: str, answer: str) -> str:
    return f"Question: {question.strip()}\nAnswer: {answer.strip()}"


def build_gsm8k_dataloaders(
    tokenizer,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    ds = load_dataset("gsm8k", "main")
    hf_datasets.disable_caching()

    train_n = min(train_samples, len(ds["train"]))
    eval_n = min(eval_samples, len(ds["test"]))
    train = ds["train"].shuffle(seed=seed).select(range(train_n))
    eval_ds = ds["test"].select(range(eval_n))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

    def _train_preprocess(batch):
        input_ids_list, attn_list, labels_list = [], [], []
        for q, a in zip(batch["question"], batch["answer"]):
            prompt = _gsm8k_format_prompt(q)
            full = _gsm8k_format_full(q, a)
            prompt_ids = tokenizer(prompt, truncation=True, max_length=GSM8K_MAX_PROMPT_LEN, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, truncation=True, max_length=GSM8K_MAX_SEQ_LEN - 1, add_special_tokens=False)["input_ids"]
            full_ids = full_ids + [eos_id]
            if len(full_ids) > GSM8K_MAX_SEQ_LEN:
                full_ids = full_ids[:GSM8K_MAX_SEQ_LEN]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            if len(labels) < len(full_ids):
                labels = labels + [-100] * (len(full_ids) - len(labels))
            labels = labels[: len(full_ids)]
            attn = [1] * len(full_ids)
            input_ids_list.append(full_ids)
            attn_list.append(attn)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": labels_list}

    def _eval_preprocess(batch):
        prompt_ids_list, attn_list, gold_list = [], [], []
        for q, a in zip(batch["question"], batch["answer"]):
            prompt_ids = tokenizer(_gsm8k_format_prompt(q), truncation=True, max_length=GSM8K_MAX_PROMPT_LEN, add_special_tokens=False)["input_ids"]
            prompt_ids_list.append(prompt_ids)
            attn_list.append([1] * len(prompt_ids))
            gold_list.append(gsm8k_extract_gold(a))
        return {"prompt_ids": prompt_ids_list, "prompt_attention_mask": attn_list, "gold": gold_list}

    train = train.map(_train_preprocess, batched=True, remove_columns=train.column_names)
    eval_ds = eval_ds.map(_eval_preprocess, batched=True, remove_columns=eval_ds.column_names)

    def _pad_right(seqs, pad_value, max_len=None):
        if max_len is None:
            max_len = max(len(s) for s in seqs)
        return [list(s) + [pad_value] * (max_len - len(s)) for s in seqs]

    def _pad_left(seqs, pad_value, max_len=None):
        if max_len is None:
            max_len = max(len(s) for s in seqs)
        return [[pad_value] * (max_len - len(s)) + list(s) for s in seqs]

    def _train_collate(features):
        input_ids = [f["input_ids"] for f in features]
        attn = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = max(len(x) for x in input_ids)
        return {
            "input_ids": torch.tensor(_pad_right(input_ids, pad_id, max_len), dtype=torch.long),
            "attention_mask": torch.tensor(_pad_right(attn, 0, max_len), dtype=torch.long),
            "labels": torch.tensor(_pad_right(labels, -100, max_len), dtype=torch.long),
        }

    def _eval_collate(features):
        prompt_ids = [f["prompt_ids"] for f in features]
        attn = [f["prompt_attention_mask"] for f in features]
        golds = [f["gold"] for f in features]
        max_len = max(len(x) for x in prompt_ids)
        return {
            "prompt_ids": torch.tensor(_pad_left(prompt_ids, pad_id, max_len), dtype=torch.long),
            "prompt_attention_mask": torch.tensor(_pad_left(attn, 0, max_len), dtype=torch.long),
            "gold": golds,
        }

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=_train_collate)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=_eval_collate)
    return train_loader, eval_loader


def _collate_and_fix_label(collator, features):
    batch = collator(features)
    if "label" in batch:
        batch["labels"] = batch.pop("label")
    return batch


def build_glue_dataloaders(
    dataset_key: str,
    tokenizer,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    seed: int,
    target_label2id: Dict[str, int],
) -> Tuple[DataLoader, DataLoader]:
    spec = DATASET_SPECS[dataset_key]
    ds = load_dataset(spec.hf_name, spec.hf_subset)
    hf_datasets.disable_caching()

    if dataset_key == "mnli":
        train = ds["train"].shuffle(seed=seed).select(range(train_samples))
        eval_ds = ds["validation_matched"].select(range(eval_samples))
        label_names = ds["train"].features["label"].names

        def preprocess(batch):
            encoded = tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)
            remap = [target_label2id.get(label_names[int(x)].lower(), int(x)) for x in batch["label"]]
            encoded["label"] = remap
            return encoded

    elif dataset_key == "qnli":
        train = ds["train"].shuffle(seed=seed).select(range(train_samples))
        eval_ds = ds["validation"].select(range(eval_samples))
        label_names = ds["train"].features["label"].names

        def preprocess(batch):
            encoded = tokenizer(batch["question"], batch["sentence"], truncation=True, max_length=128)
            remap = [target_label2id.get(label_names[int(x)].lower(), int(x)) for x in batch["label"]]
            encoded["label"] = remap
            return encoded

    elif dataset_key == "sst2":
        train = ds["train"].shuffle(seed=seed).select(range(train_samples))
        eval_ds = ds["validation"].select(range(eval_samples))
        label_names = ds["train"].features["label"].names

        def preprocess(batch):
            encoded = tokenizer(batch["sentence"], truncation=True, max_length=128)
            remap = [target_label2id.get(label_names[int(x)].lower(), int(x)) for x in batch["label"]]
            encoded["label"] = remap
            return encoded

    else:
        raise KeyError(f"Unsupported GLUE dataset: {dataset_key}")

    train = train.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train.column_names:
        columns.insert(2, "token_type_ids")
    train.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=lambda x: _collate_and_fix_label(collator, x))
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: _collate_and_fix_label(collator, x))
    return train_loader, eval_loader


def build_cifar_dataloaders(
    dataset_key: str,
    image_processor,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    seed: int,
    target_label2id: Dict[str, int],
) -> Tuple[DataLoader, DataLoader]:
    spec = DATASET_SPECS[dataset_key]
    ds = load_dataset(spec.hf_name)
    hf_datasets.disable_caching()

    train_n = min(train_samples, len(ds["train"]))
    eval_n = min(eval_samples, len(ds["test"]))
    train_ds = ds["train"].shuffle(seed=seed).select(range(train_n))
    eval_ds = ds["test"].select(range(eval_n))

    label_col = "label" if "label" in ds["train"].features else "fine_label"
    label_names = ds["train"].features[label_col].names
    remap = {src_id: int(target_label2id.get(str(src_name).lower(), src_id)) for src_id, src_name in enumerate(label_names)}

    def collate_fn(features):
        images = [feature["img"].convert("RGB") for feature in features]
        labels = [remap[int(feature[label_col])] for feature in features]
        batch = image_processor(images=images, return_tensors="pt")
        batch["labels"] = __import__("torch").tensor(labels, dtype=__import__("torch").long)
        return batch

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, eval_loader
