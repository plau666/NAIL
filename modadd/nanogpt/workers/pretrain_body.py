"""
Low-level trainer backend used by `python -m nanogpt.run`.

Public launches should generally go through Hydra presets, e.g.:
$ python -m nanogpt.run experiment=modadd_cot
$ python -m nanogpt.run experiment=modadd_noisy_bc

This training script can also be run both on a single gpu in debug mode,
and in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python -m nanogpt.run experiment=modadd_cot optim.batch_size=32 runtime.compile=false

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 -m nanogpt.run experiment=modadd_cot

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 -m nanogpt.run experiment=modadd_cot
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 -m nanogpt.run experiment=modadd_cot
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
import re
from contextlib import nullcontext

import numpy as np
import random
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from nanogpt.trainers.wandb import maybe_init_wandb

# -----------------------------------------------------------------------------
# default config values; Hydra experiment presets are the supported interface
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch', 'resume', or 'warm_start'
init_from_ckpt = ''
continue_from_subset_size = 0
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'NAIL'
wandb_run_name = 'modadd-cot'
wandb_run_id = ''
wandb_init_timeout = 300
# data
dataset = 'modadd_cot'
modadd_p = 7
modadd_m = 21
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available(
) and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster

modadd_eval_metrics = False
modadd_eval_clean_train_loss = False
eval_n = 256
eval_batch_size = 256
eval_seed = 123
save_every = 0  # 0 disables numbered checkpoints
offline_single_epoch = False
offline_eval_full = True
offline_train_subset_size = 0
offline_train_shuffle = False
offline_target_type = 'tokens'
final_eval_on_exit = False
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith(
    '_') and isinstance(v, (int, float, bool, str))]
if 'INJECTED_CONFIG' not in globals():
    raise RuntimeError(
        "nanogpt.workers.pretrain_body is an internal module. "
        "Use `python -m nanogpt.run ...`."
    )
for key, value in INJECTED_CONFIG.items():
    if key in globals():
        globals()[key] = value
    else:
        raise ValueError(f"Unknown injected config key: {key}")
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    if 'cuda' in device:
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    # this process will do logging, checkpointing etc.
    master_process = ddp_rank == 0
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * \
    ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
random.seed(1337 + seed_offset)
np.random.seed(1337 + seed_offset)
torch.manual_seed(1337 + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)


def synthetic_task_name(dataset_name):
    # Synthetic offline datasets are the rendered noisy-expert trajectory
    # datasets used by LogLossBC/SFT in the paper. Clean expert training uses
    # the non-offline task names (`modadd_cot`, `modadd_base`).
    if dataset_name in ('modadd_cot', 'modadd_base') or dataset_name.startswith('modadd_clean_offline') or dataset_name.startswith('modadd_noisy_offline'):
        return 'modadd'
    return None


def is_modadd_offline_dataset(name):
    return name.startswith('modadd_clean_offline') or name.startswith('modadd_noisy_offline')


def is_synthetic_offline_dataset(name):
    return is_modadd_offline_dataset(name)


def normalize_offline_dataset_name(name):
    return re.sub(r'_n\d+', '_n*', name)


def synthetic_eval_metrics_enabled(task_name):
    return task_name == 'modadd' and modadd_eval_metrics


def synthetic_clean_train_loss_enabled(task_name):
    return task_name == 'modadd' and modadd_eval_clean_train_loss


def synthetic_report_cot_exact(task_name):
    return False


synthetic_task = synthetic_task_name(dataset)

# synthetic task hooks
if is_synthetic_offline_dataset(dataset):
    from data.common.offline_dataset import get_batch as synthetic_offline_get_batch
    from data.common.offline_dataset import (
        get_train_batch_once,
        get_train_epoch_state,
        iter_eval_batches,
        print_offline_target_span_diagnostic,
        reset_train_epoch,
        set_train_epoch_state,
    )
    from data.common.offline_losses import offline_teacher_prob_loss_from_logits
if synthetic_task == 'modadd':
    from data.modular_addition.task import (
        estimate_saved_clean_train_loss as modadd_estimate_saved_clean_train_loss,
        evaluate_saved_clean_modadd_metrics,
        vocab_size as modadd_vocab_size,
    )
    if dataset in ('modadd_cot', 'modadd_base'):
        from data.modular_addition.task import evaluate_clean_modadd_metrics
        from data.modular_addition.task import get_batch as modadd_get_batch

# poor man's data loader
data_dir = os.path.join('data', dataset)
if is_synthetic_offline_dataset(dataset):
    subset_msg = offline_train_subset_size if offline_train_subset_size > 0 else "full"
    print(f"offline dataset dir: {data_dir}, requested train subset: {subset_msg}")

offline_dataset_meta = None
if is_synthetic_offline_dataset(dataset):
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            offline_dataset_meta = json.load(f)

if offline_target_type not in ('tokens', 'teacher_probs'):
    raise ValueError(
        f"offline_target_type must be 'tokens' or 'teacher_probs', got "
        f"{offline_target_type!r}"
    )
if offline_target_type == 'teacher_probs':
    # Full-distribution LogLossBC: instead of hard rendered tokens, train on
    # saved teacher next-token probabilities. The paper's main LogLossBC runs
    # use hard token targets, but this supports the full-information ablation.
    if not is_modadd_offline_dataset(dataset):
        raise ValueError(
            "offline_target_type='teacher_probs' is currently only supported "
            "for modular-addition offline datasets"
        )
    if offline_dataset_meta is None:
        raise ValueError(
            f"offline_target_type='teacher_probs' requires {data_dir}/meta.json"
        )
    if offline_dataset_meta.get("train_target_type") != "teacher_probs":
        raise ValueError(
            f"Dataset {dataset} has train_target_type="
            f"{offline_dataset_meta.get('train_target_type')!r}, expected "
            "'teacher_probs'"
        )
    dataset_teacher_law = offline_dataset_meta.get("teacher_law")
    if dataset_teacher_law not in ("distributional_noise", "random_suffix_after_error"):
        raise ValueError(
            f"Dataset {dataset} has teacher_law="
            f"{dataset_teacher_law!r}, expected 'distributional_noise' "
            "or 'random_suffix_after_error' for "
            "offline full-distribution BC"
        )
    expected_decode_mode = (
        "random_suffix_after_error_sample"
        if dataset_teacher_law == "random_suffix_after_error"
        else "sample_then_corrupt"
    )
    if offline_dataset_meta.get("train_decode_mode") != expected_decode_mode:
        raise ValueError(
            f"Dataset {dataset} has train_decode_mode="
            f"{offline_dataset_meta.get('train_decode_mode')!r}, expected "
            f"{expected_decode_mode!r} for offline full-distribution BC"
        )

if is_synthetic_offline_dataset(dataset) and master_process:
    print_offline_target_span_diagnostic(
        data_dir,
        method_name=f"offline_bc/{offline_target_type}",
        target_type=offline_target_type,
    )
    if offline_dataset_meta is not None:
        config['prompt_len'] = int(offline_dataset_meta.get('prompt_len', 0))
        config['cot_len'] = int(offline_dataset_meta.get('cot_len', 0))
        config['target_len'] = int(
            offline_dataset_meta.get('target_len', offline_dataset_meta.get('cot_len', 0))
        )
        config['final_answer_len'] = int(
            offline_dataset_meta.get(
                'final_answer_len',
                offline_dataset_meta.get('answer_len', 0),
            )
        )
        config['answer_len'] = int(
            offline_dataset_meta.get('answer_len', config['final_answer_len'])
        )
        config['target_span'] = offline_dataset_meta.get(
            'target_span',
            'cot_with_final_answer_suffix',
        )

resolved_modadd_p = modadd_p
resolved_modadd_m = modadd_m
if is_modadd_offline_dataset(dataset):
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            modadd_meta = json.load(f)
        resolved_modadd_p = int(modadd_meta.get('p', resolved_modadd_p))
        resolved_modadd_m = int(modadd_meta.get('m', resolved_modadd_m))
modadd_p = resolved_modadd_p
modadd_m = resolved_modadd_m
config['modadd_p'] = modadd_p
config['modadd_m'] = modadd_m
config['resolved_modadd_p'] = resolved_modadd_p
config['resolved_modadd_m'] = resolved_modadd_m
modadd_mode = 'base' if dataset == 'modadd_base' else 'cot'
config['modadd_mode'] = modadd_mode

continue_from_subset_size = int(continue_from_subset_size)
config['continue_from_subset_size'] = continue_from_subset_size
config['init_from_ckpt'] = init_from_ckpt
if continue_from_subset_size < 0:
    raise ValueError(
        f"continue_from_subset_size must be non-negative, got {continue_from_subset_size}"
    )
if init_from == 'warm_start':
    if not init_from_ckpt:
        raise ValueError("init_from='warm_start' requires init_from_ckpt to be set")
    if continue_from_subset_size > 0:
        if not is_synthetic_offline_dataset(dataset):
            raise ValueError(
                "continue_from_subset_size is only supported for synthetic offline datasets"
            )
        if not offline_single_epoch:
            raise ValueError(
                "continue_from_subset_size requires offline_single_epoch=True"
            )
        if offline_train_shuffle:
            raise ValueError(
                "continue_from_subset_size currently requires offline_train_shuffle=False"
            )
elif init_from_ckpt:
    raise ValueError("init_from_ckpt is only supported when init_from='warm_start'")


def get_batch(split, target_type=None):
    if dataset in ('modadd_cot', 'modadd_base'):
        return modadd_get_batch(
            batch_size=batch_size,
            device=device,
            p=resolved_modadd_p,
            m=resolved_modadd_m,
            mode=modadd_mode,
        )
    elif is_synthetic_offline_dataset(dataset):
        if target_type is None:
            target_type = offline_target_type if split == 'train' else 'tokens'
        return synthetic_offline_get_batch(
            split=split,
            batch_size=batch_size,
            device=device,
            data_dir=data_dir,
            subset_size=offline_train_subset_size,
            target_type=target_type,
        )
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'),
                         dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'),
                         dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def unpack_batch(batch):
    if len(batch) == 2:
        X, Y = batch
        teacher_probs = None
    elif len(batch) == 3:
        X, Y, teacher_probs = batch
    else:
        raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
    return X, Y, teacher_probs


def offline_teacher_prob_loss(model_ref, X, Y, teacher_probs):
    # Soft-target behavior cloning on saved teacher distributions. This is the
    # offline analogue of `teacher_signal=full` in the student-prefix methods.
    logits, _ = model_ref(X, return_full_logits=True)
    _, loss, stats = offline_teacher_prob_loss_from_logits(logits, Y, teacher_probs)
    return logits, loss, stats


def compute_batch_loss(model_ref, X, Y, teacher_probs=None, split='train'):
    if teacher_probs is not None:
        if split != 'train':
            raise ValueError("teacher_probs batches are only supported for train split")
        return offline_teacher_prob_loss(model_ref, X, Y, teacher_probs)
    logits, loss = model_ref(X, Y)
    return logits, loss, None


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_vocab_size = None
is_synthetic = synthetic_task is not None
if synthetic_task == 'modadd':
    meta_vocab_size = modadd_vocab_size(resolved_modadd_p)
    print(f"using modular-addition vocab_size = {meta_vocab_size} for {dataset} (p={resolved_modadd_p}, m={resolved_modadd_m})")
else:
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to padded vocab_size 50304 because dataset metadata "
            "did not specify a vocabulary size")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get('config', {})
    for key in [
        'dataset',
        'modadd_p',
        'modadd_m',
        'offline_single_epoch',
        'offline_train_subset_size',
        'offline_train_shuffle',
        'offline_target_type',
    ]:
        if key in checkpoint_config and checkpoint_config[key] != globals()[key]:
            raise ValueError(
                f"Resume mismatch for {key}: checkpoint has {checkpoint_config[key]!r}, "
                f"current config requests {globals()[key]!r}"
            )
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Normalize state dicts saved from compiled models.
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'warm_start':
    print(f"Warm-starting training from {init_from_ckpt}")
    warm_start_path = init_from_ckpt
    if os.path.isdir(warm_start_path):
        warm_start_path = os.path.join(warm_start_path, 'ckpt.pt')
    checkpoint = torch.load(warm_start_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get('config', {})
    source_dataset = checkpoint_config.get('dataset')
    if is_synthetic_offline_dataset(dataset):
        if not is_synthetic_offline_dataset(source_dataset or ''):
            raise ValueError(
                f"Warm-start checkpoint dataset={source_dataset!r} is not a synthetic offline dataset"
            )
        if normalize_offline_dataset_name(source_dataset) != normalize_offline_dataset_name(dataset):
            raise ValueError(
                f"Warm-start dataset mismatch: checkpoint has {source_dataset!r}, "
                f"current config requests {dataset!r}"
            )
        for key in ['offline_single_epoch', 'offline_train_shuffle', 'offline_target_type']:
            if key in checkpoint_config and checkpoint_config[key] != globals()[key]:
                raise ValueError(
                    f"Warm-start mismatch for {key}: checkpoint has {checkpoint_config[key]!r}, "
                    f"current config requests {globals()[key]!r}"
                )
        for key in ['modadd_p', 'modadd_m']:
            if key in checkpoint_config and int(checkpoint_config[key]) != int(globals()[key]):
                raise ValueError(
                    f"Warm-start mismatch for {key}: checkpoint has {checkpoint_config[key]!r}, "
                    f"current config requests {globals()[key]!r}"
                )
        if continue_from_subset_size > 0:
            source_state = checkpoint.get('offline_train_state')
            if source_state is None:
                raise ValueError(
                    "Warm-start continuation requires offline_train_state in the source checkpoint"
                )
            source_n = int(source_state['n'])
            source_pos = int(source_state['pos'])
            if source_n != continue_from_subset_size:
                raise ValueError(
                    f"Warm-start source checkpoint covered n={source_n}, expected "
                    f"continue_from_subset_size={continue_from_subset_size}"
                )
            if source_pos != source_n:
                raise ValueError(
                    f"Warm-start source checkpoint stopped at pos={source_pos}, expected a "
                    f"completed prefix of length {source_n}"
                )
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(
        f"unsupported init_from={init_from!r}; expected 'scratch', 'resume', or 'warm_start'"
    )
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    # so that the checkpoint will have the right value
    model_args['block_size'] = block_size
model.to(device)
if offline_target_type == 'teacher_probs' and offline_dataset_meta is not None:
    expected_vocab_size = offline_dataset_meta.get("vocab_size")
    if expected_vocab_size is not None and int(expected_vocab_size) != int(model.config.vocab_size):
        raise ValueError(
            f"Dataset vocab_size={expected_vocab_size} does not match model "
            f"vocab_size={model.config.vocab_size}"
        )

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type)
resume_offline_train_state = None
warm_start_offline_train_state = None
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    resume_offline_train_state = checkpoint.get('offline_train_state')
elif init_from == 'warm_start':
    optimizer.load_state_dict(checkpoint['optimizer'])
    warm_start_offline_train_state = checkpoint.get('offline_train_state')
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    if is_synthetic_offline_dataset(dataset):
        # LogLossBC evaluates against the fixed clean validation split from
        # the prompt bank, not against newly sampled noisy trajectories.
        for split in ['train', 'val']:
            split_target_type = offline_target_type if split == 'train' else 'tokens'
            if offline_eval_full:
                losses = []
                for batch in iter_eval_batches(
                    split,
                    batch_size=batch_size,
                    device=device,
                    data_dir=data_dir,
                    subset_size=offline_train_subset_size,
                    target_type=split_target_type,
                ):
                    X, Y, teacher_probs = unpack_batch(batch)
                    with ctx:
                        _, loss, _ = compute_batch_loss(
                            model,
                            X,
                            Y,
                            teacher_probs=teacher_probs,
                            split=split,
                        )
                    losses.append(loss.item())
                out[split] = sum(losses) / len(losses)
            else:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y, teacher_probs = unpack_batch(
                        get_batch(split, target_type=split_target_type)
                    )
                    with ctx:
                        _, loss, _ = compute_batch_loss(
                            model,
                            X,
                            Y,
                            teacher_probs=teacher_probs,
                            split=split,
                        )
                    losses[k] = loss.item()
                out[split] = losses.mean()

        if synthetic_eval_metrics_enabled(synthetic_task):
            eval_model = raw_model if 'raw_model' in globals() else model
            metrics = evaluate_saved_clean_modadd_metrics(
                eval_model,
                device=device,
                data_dir=data_dir,
                n_eval=eval_n,
                batch_size=eval_batch_size,
            )
            if synthetic_report_cot_exact(synthetic_task):
                out["val_cot_exact"] = metrics["cot_exact"]
            out["val_clean_full_exact"] = metrics["clean_full_exact"]
            out["val_clean_final_exact"] = metrics["clean_final_exact"]
        if synthetic_clean_train_loss_enabled(synthetic_task):
            eval_model = raw_model if 'raw_model' in globals() else model
            out["train_clean_oracle"] = modadd_estimate_saved_clean_train_loss(
                eval_model,
                device=device,
                data_dir=data_dir,
                eval_iters=eval_iters,
                batch_size=batch_size,
                subset_size=offline_train_subset_size,
            )

    else:
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        if synthetic_eval_metrics_enabled(synthetic_task):
            eval_model = raw_model if 'raw_model' in globals() else model
            metrics = evaluate_clean_modadd_metrics(
                eval_model,
                device=device,
                p=resolved_modadd_p,
                m=resolved_modadd_m,
                n_eval=eval_n,
                seed=eval_seed,
                batch_size=eval_batch_size,
                mode=modadd_mode,
            )
            if synthetic_report_cot_exact(synthetic_task):
                out["val_cot_exact"] = metrics["cot_exact"]
            out["val_clean_full_exact"] = metrics["clean_full_exact"]
            out["val_clean_final_exact"] = metrics["clean_final_exact"]

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)


def get_lr(it):
    lr_start = 1e-6

    # 1) linear warmup from 1e-6 to learning_rate
    if it < warmup_iters:
        return lr_start + (learning_rate - lr_start) * (it + 1) / (warmup_iters + 1)

    # Degenerate schedules with no post-warmup decay window should stay flat.
    if lr_decay_iters <= warmup_iters:
        return learning_rate

    # 2) after decay horizon, stay at min_lr
    if it > lr_decay_iters:
        return min_lr

    # 3) cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# logging
wandb = maybe_init_wandb(
    enabled=wandb_log and master_process,
    project=wandb_project,
    run_name=wandb_run_name,
    run_id=wandb_run_id or None,
    out_dir=out_dir,
    init_from=init_from,
    init_timeout=wandb_init_timeout,
    config=config,
)
if wandb is not None:
    wandb.define_metric("iter")
    wandb.define_metric("lr", step_metric="iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")


def save_eval_summary(reason, losses):
    aliases = {}
    if "train" in losses:
        aliases["train/loss_eval"] = float(losses["train"])
    if "val" in losses:
        aliases["val/loss"] = float(losses["val"])
    if "train_clean_oracle" in losses:
        aliases["train/clean_oracle_loss_eval"] = float(losses["train_clean_oracle"])
    if "val_cot_exact" in losses:
        aliases["val/cot_exact"] = float(losses["val_cot_exact"])
    if "val_clean_full_exact" in losses:
        aliases["val/clean_full_exact"] = float(losses["val_clean_full_exact"])
    if "val_clean_final_exact" in losses:
        aliases["val/clean_final_exact"] = float(losses["val_clean_final_exact"])

    summary = {
        "iter": int(iter_num),
        "reason": reason,
        **{
            key: float(value)
            for key, value in losses.items()
        },
        **aliases,
    }
    with open(os.path.join(out_dir, 'last_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'eval_history.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary) + '\n')


def run_eval_and_checkpoint(reason="periodic", force_save=False):
    global best_val_loss

    losses = estimate_loss()
    save_eval_summary(reason, losses)
    print_msg = f"{reason} step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    if "train_clean_oracle" in losses:
        print_msg += f", train clean_oracle_loss {losses['train_clean_oracle']:.4f}"
    if synthetic_eval_metrics_enabled(synthetic_task):
        if "val_cot_exact" in losses:
            print_msg += f", val cot_exact {losses['val_cot_exact']:.4f}"
        if "val_clean_full_exact" in losses:
            print_msg += f", val clean_full_exact {losses['val_clean_full_exact']:.4f}"
        if "val_clean_final_exact" in losses:
            print_msg += f", val clean_final_exact {losses['val_clean_final_exact']:.4f}"
    print(print_msg)

    if wandb is not None:
        eval_log = {
            "iter": iter_num,
            "eval/reason": reason,
            "train/loss_eval": losses["train"],
            "val/loss": losses["val"],
        }
        if "train_clean_oracle" in losses:
            eval_log["train/clean_oracle_loss_eval"] = losses["train_clean_oracle"]
        if synthetic_eval_metrics_enabled(synthetic_task):
            if "val_cot_exact" in losses:
                eval_log["val/cot_exact"] = losses["val_cot_exact"]
            if "val_clean_full_exact" in losses:
                eval_log["val/clean_full_exact"] = losses["val_clean_full_exact"]
            if "val_clean_final_exact" in losses:
                eval_log["val/clean_final_exact"] = losses["val_clean_final_exact"]
        for key, value in losses.items():
            if "/" in key:
                eval_log[key] = float(value)
        wandb.log(eval_log)

    if losses['val'] < best_val_loss or always_save_checkpoint or force_save:
        best_val_loss = min(best_val_loss, losses['val'])
        if iter_num > 0 or force_save:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            if is_synthetic_offline_dataset(dataset) and offline_single_epoch:
                checkpoint['offline_train_state'] = get_train_epoch_state(data_dir)
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            if save_every > 0 and iter_num % save_every == 0:
                numbered_path = os.path.join(
                    out_dir, f'ckpt_{iter_num:07d}.pt')
                print(f"saving checkpoint to {numbered_path}")
                torch.save(checkpoint, numbered_path)


def mark_run_complete():
    completed_path = os.path.join(out_dir, 'completed.txt')
    with open(completed_path, 'w', encoding='utf-8') as f:
        f.write(f"iter_num={iter_num}\n")

# training loop
if is_synthetic_offline_dataset(dataset) and offline_single_epoch:
    assert gradient_accumulation_steps == 1
    if init_from == 'resume' and resume_offline_train_state is not None:
        set_train_epoch_state(data_dir, resume_offline_train_state)
    else:
        start_pos = continue_from_subset_size if init_from == 'warm_start' else 0
        if init_from == 'warm_start' and continue_from_subset_size > 0:
            if warm_start_offline_train_state is None:
                raise ValueError(
                    "Warm-start continuation requires offline_train_state in the source checkpoint"
                )
        reset_train_epoch(
            data_dir,
            shuffle=offline_train_shuffle,
            seed=1337 + seed_offset,
            subset_size=offline_train_subset_size,
            start_pos=start_pos,
        )
else:
    X, Y, teacher_probs = unpack_batch(get_batch('train'))
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        run_eval_and_checkpoint(reason="periodic")

    if iter_num == 0 and eval_only:
        break

    if is_synthetic_offline_dataset(dataset) and offline_single_epoch:
        try:
            X, Y, teacher_probs = unpack_batch(get_train_batch_once(
                batch_size=batch_size,
                device=device,
                data_dir=data_dir,
                subset_size=offline_train_subset_size,
                target_type=offline_target_type,
            ))
        except StopIteration:
            if final_eval_on_exit and master_process:
                run_eval_and_checkpoint(reason="final", force_save=True)
                mark_run_complete()
            break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, _ = compute_batch_loss(
                model,
                X,
                Y,
                teacher_probs=teacher_probs,
                split='train',
            )
            # scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        if is_synthetic_offline_dataset(dataset) and offline_single_epoch:
            pass
        else:
            X, Y, teacher_probs = unpack_batch(get_batch('train'))
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(
                batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        if wandb is not None:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
            })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        if final_eval_on_exit and master_process:
            run_eval_and_checkpoint(reason="final", force_save=True)
            mark_run_complete()
        break

if ddp:
    destroy_process_group()

if wandb is not None:
    wandb.finish()
