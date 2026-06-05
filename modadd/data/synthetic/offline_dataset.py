import os
import json

import torch

from data.synthetic.target_spans import (
    print_target_span_diagnostic,
    target_ids_from_y_row,
)

# Loader for fixed rendered datasets used by LogLossBC/SFT. Student-prefix
# methods read the prompt bank and query the teacher on current student
# prefixes instead.
_CACHE = {}
_STATE = {}


def _effective_n(total_n, subset_size, split):
    if subset_size is None or subset_size <= 0 or split != "train":
        return total_n
    subset_size = int(subset_size)
    if subset_size > total_n:
        raise ValueError(
            f"requested train subset_size={subset_size} exceeds available "
            f"{split} rows={total_n}"
        )
    return subset_size


def _load_meta(data_dir):
    key = (data_dir, "meta")
    if key not in _CACHE:
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def _target_len_from_meta(meta):
    cot_len = int(meta["cot_len"])
    target_len = int(meta.get("target_len", cot_len))
    if target_len != cot_len:
        raise ValueError(
            f"offline dataset target_len {target_len} does not match cot_len {cot_len}"
        )
    return target_len


def _validate_teacher_probs(meta, x, y, teacher_probs, data_dir):
    if meta.get("train_target_type") != "teacher_probs":
        raise ValueError(
            f"offline_target_type='teacher_probs' requested for {data_dir}, "
            f"but dataset meta train_target_type={meta.get('train_target_type')!r}"
        )
    if teacher_probs is None:
        raise ValueError(
            f"Dataset {data_dir} is missing train_teacher_probs.pt required for "
            "offline_target_type='teacher_probs'"
        )
    if teacher_probs.ndim != 3:
        raise ValueError(
            f"Expected train_teacher_probs.pt to have rank 3, got shape "
            f"{tuple(teacher_probs.shape)}"
        )
    if teacher_probs.size(0) != x.size(0):
        raise ValueError(
            f"train_teacher_probs rows {teacher_probs.size(0)} do not match "
            f"train_x rows {x.size(0)}"
        )
    target_len = _target_len_from_meta(meta)
    if teacher_probs.size(1) != target_len:
        raise ValueError(
            f"train_teacher_probs target_len {teacher_probs.size(1)} does not "
            f"match meta target_len {target_len}"
        )
    if x.size(1) != y.size(1):
        raise ValueError(
            f"train_x sequence length {x.size(1)} does not match train_y "
            f"sequence length {y.size(1)}"
        )
    expected_seq_len = int(meta["prompt_len"]) + target_len - 1
    if x.size(1) != expected_seq_len:
        raise ValueError(
            f"train_x sequence length {x.size(1)} does not match expected "
            f"prompt_len+target_len-1={expected_seq_len}"
        )


def _load_split(data_dir, split, target_type="tokens"):
    key = (data_dir, split, target_type)
    if key not in _CACHE:
        x = torch.load(os.path.join(data_dir, f"{split}_x.pt"), map_location="cpu")
        y = torch.load(os.path.join(data_dir, f"{split}_y.pt"), map_location="cpu")
        teacher_probs = None
        if split == "train" and target_type == "teacher_probs":
            meta = _load_meta(data_dir)
            teacher_probs_path = os.path.join(data_dir, "train_teacher_probs.pt")
            teacher_probs = torch.load(teacher_probs_path, map_location="cpu")
            _validate_teacher_probs(meta, x, y, teacher_probs, data_dir)
        _CACHE[key] = (x, y, teacher_probs)
    return _CACHE[key]


def print_offline_target_span_diagnostic(data_dir, *, method_name, target_type="tokens"):
    meta = _load_meta(data_dir)
    x_all, y_all, teacher_probs_all = _load_split(
        data_dir,
        "train",
        target_type=target_type,
    )
    if x_all.size(0) == 0:
        raise ValueError(f"offline dataset {data_dir} has no train rows")

    target_ids = target_ids_from_y_row(y_all[0])
    actual_target_len = int(target_ids.numel())
    if teacher_probs_all is not None:
        prob_target_len = int(teacher_probs_all.size(1))
        if prob_target_len != actual_target_len:
            raise ValueError(
                f"teacher_probs target_len {prob_target_len} does not match "
                f"Y continuation length {actual_target_len}"
            )

    print_target_span_diagnostic(
        method_name=method_name,
        task=str(meta.get("task", "s5")),
        p=int(meta.get("p", 5)),
        prompt_len=int(meta["prompt_len"]),
        cot_len=_target_len_from_meta(meta),
        final_answer_len=int(meta.get("final_answer_len", meta.get("answer_len", 0))),
        actual_target_len=actual_target_len,
        total_sequence_len=int(x_all.size(1)),
        prompt_ids=x_all[0, : int(meta["prompt_len"])],
        target_ids=target_ids,
        target_description=f"offline {target_type} continuation from train_y suffix",
    )


def _move(x, y, device, teacher_probs=None):
    x = x.long()
    y = y.long()
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        if teacher_probs is not None:
            teacher_probs = teacher_probs.pin_memory().to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
    else:
        x = x.to(device)
        y = y.to(device)
        if teacher_probs is not None:
            teacher_probs = teacher_probs.to(device=device, dtype=torch.float32)
    if teacher_probs is None:
        return x, y
    return x, y, teacher_probs


def reset_train_epoch(data_dir, shuffle=True, seed=None, subset_size=None, start_pos=0):
    x_all, _, _ = _load_split(data_dir, "train")
    n = _effective_n(x_all.size(0), subset_size, "train")
    start_pos = int(start_pos)
    if start_pos < 0 or start_pos > n:
        raise ValueError(
            f"requested train start_pos={start_pos} must be between 0 and available "
            f"train rows={n}"
        )

    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        perm = torch.randperm(n, generator=g)
    else:
        perm = torch.arange(n)

    _STATE[data_dir] = {
        "perm": perm,
        "pos": start_pos,
        "n": n,
    }


def get_train_epoch_state(data_dir):
    if data_dir not in _STATE:
        return None
    st = _STATE[data_dir]
    return {
        "perm": st["perm"].clone(),
        "pos": int(st["pos"]),
        "n": int(st["n"]),
    }


def set_train_epoch_state(data_dir, state):
    _STATE[data_dir] = {
        "perm": state["perm"].to(device="cpu", dtype=torch.long).clone(),
        "pos": int(state["pos"]),
        "n": int(state["n"]),
    }


def get_batch(split, batch_size, device, data_dir, subset_size=None, target_type="tokens"):
    x_all, y_all, teacher_probs_all = _load_split(data_dir, split, target_type=target_type)
    n = _effective_n(x_all.size(0), subset_size, split)
    idx = torch.randint(n, (batch_size,))
    teacher_probs = None
    if teacher_probs_all is not None:
        teacher_probs = teacher_probs_all[idx]
    return _move(x_all[idx], y_all[idx], device, teacher_probs=teacher_probs)


def get_train_batch_once(batch_size, device, data_dir, subset_size=None, target_type="tokens"):
    if data_dir not in _STATE:
        reset_train_epoch(data_dir, shuffle=True, subset_size=subset_size)

    x_all, y_all, teacher_probs_all = _load_split(data_dir, "train", target_type=target_type)
    st = _STATE[data_dir]

    if st["pos"] >= st["n"]:
        raise StopIteration

    idx = st["perm"][st["pos"]: st["pos"] + batch_size]
    st["pos"] += idx.numel()

    teacher_probs = None
    if teacher_probs_all is not None:
        teacher_probs = teacher_probs_all[idx]
    return _move(x_all[idx], y_all[idx], device, teacher_probs=teacher_probs)


def iter_eval_batches(split, batch_size, device, data_dir, subset_size=None, target_type="tokens"):
    x_all, y_all, teacher_probs_all = _load_split(data_dir, split, target_type=target_type)
    n = _effective_n(x_all.size(0), subset_size, split)
    for start in range(0, n, batch_size):
        idx = slice(start, min(start + batch_size, n))
        teacher_probs = None
        if teacher_probs_all is not None:
            teacher_probs = teacher_probs_all[idx]
        yield _move(x_all[idx], y_all[idx], device, teacher_probs=teacher_probs)
