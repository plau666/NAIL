"""Shared utilities for gsm/ experiments: dataset, collator, eval, GSM8K answer extraction.

Simplified from real_math/offline_bc.py — no few-shot prompting (we always use
Gemma-IT with its chat template).
"""

import json
import re

import torch
import wandb
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None


def extract_last_number(text: str):
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return matches[-1].replace(",", "") if matches else None


def normalize_answer(s: str) -> str:
    return s.strip().replace("$", "").replace("%", "").replace(",", "").rstrip(".")


def extract_answer(text: str):
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    num = extract_last_number(text)
    return normalize_answer(num) if num is not None else None


def extract_gt_answer(answer_text) -> str:
    s = str(answer_text)
    return normalize_answer(s.split("####")[-1].strip())


def answers_match(pred: str, gt: str) -> bool:
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except (ValueError, TypeError):
        return pred == gt


# ---------------------------------------------------------------------------
# Prompt construction (Gemma-IT chat template, always)
# ---------------------------------------------------------------------------

def build_prompt(question: str, tokenizer, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Dataset for SFT on (prompt, completion) pairs — loss masked on prompt tokens
# ---------------------------------------------------------------------------

class RolloutSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048,
                 prompt_field="prompt", completion_field="completion",
                 system_prompt=SYSTEM_PROMPT):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.prompt_field = prompt_field
        self.completion_field = completion_field
        self.examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = ex[self.prompt_field]
        completion = ex[self.completion_field]

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_messages = prompt_messages + [{"role": "assistant", "content": completion}]
        full_str = self.tokenizer.apply_chat_template(full_messages, tokenize=False)

        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_str, add_special_tokens=False)["input_ids"]

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        assert len(full_ids) == len(labels)
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


class SFTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# GSM8K generation-based accuracy eval
# ---------------------------------------------------------------------------

def evaluate_on_gsm8k(model, tokenizer, eval_data_path, max_examples=None,
                      max_new_tokens=512, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    examples = []
    with open(eval_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if max_examples:
        examples = examples[:max_examples]

    prompts, gt_answers = [], []
    for ex in examples:
        question = ex.get("prompt") or ex.get("question")
        prompts.append(build_prompt(question, tokenizer))
        gt_answers.append(extract_gt_answer(ex.get("answer", "")))

    correct = total = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j, output in enumerate(outputs):
            prompt_len = encoded["input_ids"].shape[1]
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            pred = extract_answer(generated)
            if pred is not None and answers_match(pred, gt_answers[i + j]):
                correct += 1
            total += 1

    tokenizer.padding_side = original_padding_side
    model.train()
    return correct / total if total > 0 else 0.0, correct, total


def compute_eval_loss(model, tokenizer, eval_data_path, max_length=1024,
                      batch_size=4, prompt_field="question", completion_field="answer",
                      system_prompt=SYSTEM_PROMPT):
    """LM cross-entropy loss on (prompt, completion) pairs, masked to completion tokens.

    Returns (mean_loss_per_completion_token, n_examples).
    Used by forward / reverse custom training loops to get a periodic eval_loss.
    """
    ds = RolloutSFTDataset(eval_data_path, tokenizer, max_length=max_length,
                            prompt_field=prompt_field, completion_field=completion_field,
                            system_prompt=system_prompt)
    collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    device = next(model.parameters()).device
    total_loss_sum = 0.0  # sum over all completion tokens
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # CE loss with ignore_index=-100 averaged over completion tokens
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_tok = (shift_labels != -100).sum().item()
            total_loss_sum += loss.item()
            total_tokens += n_tok

    model.train()
    mean_loss = total_loss_sum / max(1, total_tokens)
    return mean_loss, len(ds)


class GSM8KAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_data_path, eval_steps=250,
                 max_eval_examples=200, max_new_tokens=512, eval_batch_size=32,
                 skip_initial_eval=False):
        self.tokenizer = tokenizer
        self.eval_data_path = eval_data_path
        self.eval_steps = eval_steps
        self.max_eval_examples = max_eval_examples
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        self.skip_initial_eval = skip_initial_eval

    def _run_eval(self, model):
        return evaluate_on_gsm8k(
            model, self.tokenizer, self.eval_data_path,
            max_examples=self.max_eval_examples,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.eval_batch_size,
        )

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if self.skip_initial_eval:
            print("[Step 0] Skipping initial eval")
            return
        print("\n[Step 0] Running GSM8K baseline accuracy eval...")
        acc, n_c, n_t = self._run_eval(model)
        print(f"  Baseline accuracy: {n_c}/{n_t} = {acc * 100:.1f}%")
        if wandb.run is not None:
            wandb.log({"eval/gsm8k_accuracy": acc, "train/global_step": 0}, step=0)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        print(f"\n[Step {state.global_step}] Running GSM8K eval...")
        acc, n_c, n_t = self._run_eval(model)
        print(f"  accuracy: {n_c}/{n_t} = {acc * 100:.1f}%")
        if wandb.run is not None:
            wandb.log({
                "eval/gsm8k_accuracy": acc,
                "train/global_step": state.global_step,
            }, step=state.global_step)
