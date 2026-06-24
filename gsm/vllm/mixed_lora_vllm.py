"""Mixed forward + reverse KL on-policy distillation (NAIL-Mixed) with LoRA.

Combines the forward-KL surrogate (hard-label CE on the expert's MC token at
each student prefix) and the reverse-KL surrogate (IS-weighted advantage at
each visited prefix) on a SINGLE student rollout. Both arms share the
expert/student forward passes per chunk — only the per-token loss differs.

Loss:
    L(θ) = (w_fwd · L_forward(θ) + w_rev · L_reverse(θ))
where
    L_forward(θ) = -mean_real_token[ log π_θ(ỹ_t | s_t) ]   with ỹ_t ~ π_E,noisy
    L_reverse(θ) = -mean_real_token[ IW_t · (log π_E,noisy(a_t|s_t) - log π_θ(a_t|s_t).detach()) ]

Both means use the SAME global denominator: total_n_real = sum over the whole
opt-step batch of non-pad answer positions. Pad positions (HF-inserted post-EOS
fillers) are excluded from numerator and denominator in both arms — matching
NeMo-RL's `masked_mean(..., global_normalization_factor=global_valid_toks)` and
Tinker's `sum(loss * mask) / sum(mask)`.

CLI variants (controlled by --beta ∈ [0, 1]):
    Pure NAIL-F      (forward-only, greedy rollouts):
        --beta 0 --student_temperature 0
    Pure OPD-F       (forward-only, temp-1 rollouts):
        --beta 0 --student_temperature 1
    Pure NAIL-R      (reverse-only, greedy rollouts + aux):
        --beta 1 --student_temperature 0 --aux_sample
    Pure OPD-R       (reverse-only, temp-1 rollouts):
        --beta 1 --student_temperature 1
    NAIL-Mixed (default): equal mix, both arms active:
        --beta 0.5

Constraints:
    - If reverse_weight > 0, expert_temperature must be > 0 (the reverse arm
      uses log π_E at the chosen action; T=0 would give -inf at non-argmax).
    - student_temperature is independent of expert_temperature and can be 0
      (greedy) or 1 (temp-1 sampling).

Usage:
    python mixed_lora.py \
        --student_model google/gemma-3-270m-it \
        --expert_model google/gemma-3-1b-it \
        --train_data data/gsm8k/train.jsonl \
        --output_dir output/mixed_lora_r128 \
        --name mixed_lora_r128 --lora_rank 128

    # Resume from latest:
    python mixed_lora.py ... --resume_from_checkpoint auto
"""

import argparse
import glob
import json
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

# Colocated vLLM rollout engine. The env var MUST be set before vllm is imported
# (it is imported inside vllm_rollout) so the engine runs IN-PROCESS rather than
# forking a subprocess — forking after CUDA is initialized in the training
# process crashes. In-process also shares the CUDA context for cheap LoRA sync.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm_rollout import VLLMRolloutGenerator

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_utils import (
    evaluate_on_gsm8k,
    compute_eval_loss,
    compute_rollout_diagnostics,
    SYSTEM_PROMPT,
)


DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class PromptDataset(TorchDataset):
    def __init__(self, data_path, prompt_field="question"):
        self.examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line)[prompt_field])
        print(f"Loaded {len(self.examples)} prompts from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"question": self.examples[idx]}


def collate_prompts(batch, tokenizer, system_prompt):
    prompts = []
    for ex in batch:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=4096)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def get_logprobs_for_actions(logits, actions):
    """log π(a|s): log_softmax then gather."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)


def resolve_resume_checkpoint(output_dir, arg_value):
    if not arg_value or arg_value.lower() == "none":
        return None
    if arg_value.lower() == "auto":
        ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not ckpts:
            return None
        def step_of(p):
            m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        return max(ckpts, key=step_of)
    return arg_value


def save_checkpoint(save_path, model, tokenizer, optimizer, scheduler,
                    global_step, epoch, extra_state=None):
    """Save LoRA adapter + optimizer + scheduler + RNG + step so we can resume."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, os.path.join(save_path, "trainer_state.pt"))


def load_training_state(ckpt_path, optimizer, scheduler):
    state_path = os.path.join(ckpt_path, "trainer_state.pt")
    if not os.path.exists(state_path):
        print(f"WARNING: {state_path} not found — will only restore model weights")
        return 0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    print(f"Restored training state: step={state['global_step']}, epoch={state['epoch']}")
    return state["global_step"], state["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Mixed forward + reverse KL on-policy distillation (NAIL-Mixed) with LoRA")

    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--expert_model", type=str, required=True)

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--prompt_field", type=str, default="question")
    parser.add_argument("--eval_source", type=str, default=None)

    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="NAIL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Micro batch size (per forward/backward pass)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=0,
                        help="Batch size for generate(). 0 = batch_size * grad_accum")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Clip global L2 norm of trainable-param gradients at "
                             "this value (0 = disabled). Default 1.0 matches "
                             "small-cot's `grad_clip: 1.0` and NeMo-RL's default.")
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.15,
                        help="Fraction of GPU memory for the colocated vLLM engine "
                             "(weights + KV cache). ~0.15 holds all rollouts at "
                             "worst-case length for gemma-270m; the rest is for training.")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--student_temperature", type=float, default=1.0,
                        help="Student rollout temperature. 0 = greedy")
    parser.add_argument("--expert_temperature", type=float, default=1.0,
                        help="Temperature applied to expert logits. For forward arm "
                             "it controls the MC sample distribution (0 = argmax). "
                             "For reverse arm it scales log π_E (must be > 0).")

    # Mixed weight: beta is the reverse-arm weight; forward gets (1 - beta).
    # This matches the original CLI (`--beta 0` → NAIL-F, `--beta 1` → NAIL-R).
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Mixing weight. loss = (1 - beta) * forward + beta * reverse. "
                             "beta=0 → pure forward (NAIL-F/OPD-F); beta=1 → pure reverse "
                             "(NAIL-R/OPD-R). Default 0.5 (equal mix).")
    parser.add_argument("--aux_sample", action="store_true", default=False,
                        help="Reverse arm: draw a fresh aux student token at each "
                             "prefix instead of reusing the rollout token. "
                             "Recommended when student_temperature=0 (NAIL-R-style).")

    parser.add_argument("--eval_steps", type=int, default=999999)
    parser.add_argument("--max_eval_examples", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--skip_initial_eval", action="store_true", default=False)

    # GSM8K eval loss (runs every save_steps if set)
    parser.add_argument("--gsm8k_eval_loss_data", type=str, default=None,
                        help="Raw GSM8K jsonl (question/answer fields) for periodic eval loss. "
                             "Computed every save_steps.")
    parser.add_argument("--gsm8k_eval_loss_batch_size", type=int, default=4)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="Default 2*rank")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=None,
                        help=f"LoRA target modules (default: {DEFAULT_LORA_TARGETS})")

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' for latest in output_dir, or path to specific checkpoint")

    args = parser.parse_args()

    # === Sanity checks on the mixed-arm configuration ===
    if not (0.0 <= args.beta <= 1.0):
        raise ValueError(f"--beta must be in [0, 1], got {args.beta}")
    forward_weight = 1.0 - args.beta
    reverse_weight = args.beta
    if reverse_weight > 0:
        assert args.expert_temperature > 0, \
            "Reverse arm requires expert_temperature > 0 (log π_E at non-argmax tokens " \
            "would be -inf at T=0). Use beta=0 (pure forward) or set --expert_temperature 1.0."

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    resume_path = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_path:
        print(f"Resuming from: {resume_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # --- Base student + LoRA wrap ---
    print(f"Loading base student: {args.student_model}")
    base_student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    target_modules = args.lora_target_modules or DEFAULT_LORA_TARGETS
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if resume_path and os.path.isdir(resume_path):
        print(f"Loading LoRA adapter from {resume_path}")
        student = PeftModel.from_pretrained(base_student, resume_path, is_trainable=True)
    else:
        print(f"Creating fresh LoRA adapter (rank={args.lora_rank}, "
              f"alpha={lora_alpha}, targets={target_modules})")
        student = get_peft_model(base_student, lora_config)

    if args.gradient_checkpointing:
        student.enable_input_require_grads()
        student.gradient_checkpointing_enable()
    student.train()

    n_total = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student total: {n_total:,} | Trainable (LoRA): {n_trainable:,} "
          f"({100 * n_trainable / n_total:.3f}%)")

    # --- Expert (frozen) ---
    print(f"Loading expert: {args.expert_model}")
    expert = AutoModelForCausalLM.from_pretrained(
        args.expert_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False
    print("Expert loaded and frozen.")

    # --- Data ---
    from functools import partial
    rollout_bsz = args.rollout_batch_size or (args.batch_size * args.gradient_accumulation_steps)
    assert rollout_bsz % args.batch_size == 0
    n_chunks = rollout_bsz // args.batch_size
    print(f"Rollout batch: {rollout_bsz} | Micro batch: {args.batch_size} | Chunks per opt step: {n_chunks}")

    prompt_ds = PromptDataset(args.train_data, prompt_field=args.prompt_field)
    dataloader = DataLoader(
        prompt_ds,
        batch_size=rollout_bsz,
        shuffle=True,
        collate_fn=partial(collate_prompts, tokenizer=tokenizer,
                           system_prompt=args.system_prompt),
        num_workers=0,
        drop_last=True,
    )

    # --- Optimizer (LoRA params only) ---
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    total_steps = args.max_steps if args.max_steps > 0 else \
        len(dataloader) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def get_lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # --- Restore training state if resuming ---
    start_step = 0
    start_epoch = 0
    if resume_path and os.path.isdir(resume_path):
        start_step, start_epoch = load_training_state(resume_path, optimizer, scheduler)
        if start_step >= total_steps:
            print(f"Resumed step {start_step} >= total_steps {total_steps}; "
                  f"nothing to do.")
        else:
            n_per_epoch = len(dataloader)
            batches_left_in_epoch = n_per_epoch - (start_step % n_per_epoch)
            wasted = n_per_epoch - batches_left_in_epoch
            print(f"Resumed at step {start_step}/{total_steps} (epoch {start_epoch}). "
                  f"NOTE: dataloader restarts from epoch start, so ~{wasted} batches "
                  f"will be re-processed before reaching new work. Training will still "
                  f"stop exactly at global_step={total_steps} (step-based hard cap).")

    # --- Generation config ---
    # Stop rollout at any of: primary EOS or chat-template turn-end markers.
    # Chat-tuned models (Gemma-3-it, Llama-3-Instruct, Qwen-it, …) end an
    # assistant turn by emitting a chat-template token like <end_of_turn>
    # (Gemma) or <|eot_id|> (Llama) rather than the primary <eos>. The model's
    # default `generation_config.eos_token_id` is typically already a list of
    # both (e.g. Gemma-3-it: [1, 106]). Use that list so the rollout halts on
    # whichever terminator the model actually emits — matches the NeMo-RL /
    # Tinker convention. Without this, rollouts ramble past <end_of_turn> and
    # post-EOT tokens leak into the loss as if they were a legitimate response.
    default_eos = getattr(base_student.generation_config, "eos_token_id", None)
    stop_token_ids = default_eos if default_eos is not None else tokenizer.eos_token_id
    print(f"Rollout stop tokens: {stop_token_ids}")
    if args.student_temperature == 0:
        gen_config = GenerationConfig(
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        gen_config = GenerationConfig(
            do_sample=True, temperature=args.student_temperature, top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            max_new_tokens=args.max_new_tokens,
        )

    # --- Colocated vLLM rollout engine (in-process; truly-online per-step sync) ---
    # Replaces student.generate() for the rollout ONLY. The expert forward and the
    # student forward/backward that compute the mixed loss stay in HF/PyTorch
    # (vLLM cannot do backward). Init AFTER the training models are on GPU so vLLM
    # sizes its KV cache around the remaining memory.
    vllm_stop_token_ids = stop_token_ids if isinstance(stop_token_ids, list) else [stop_token_ids]
    _dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "_")
    print(f"Initializing colocated vLLM engine (gpu_mem_util={args.vllm_gpu_mem_util})...")
    rollout_engine = VLLMRolloutGenerator(
        args.student_model, max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.vllm_gpu_mem_util,
        max_model_len=4096, enforce_eager=False, seed=args.seed,
        adapter_dir=f"/dev/shm/nail_vllm_adapter_dev{_dev}",
    )

    # --- WandB ---
    wandb.init(project=args.wandb_project, name=args.name, config=config,
               dir=os.path.dirname(os.path.abspath(__file__)),
               resume="allow")

    # --- Initial eval ---
    if args.eval_source and not args.skip_initial_eval and start_step == 0:
        print(f"\n[Step 0] Running GSM8K baseline eval...")
        torch.cuda.empty_cache()
        acc, n_correct, n_total = evaluate_on_gsm8k(
            student, tokenizer, args.eval_source,
            max_examples=args.max_eval_examples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
        )
        print(f"  Baseline GSM8K accuracy: {n_correct}/{n_total} = {acc * 100:.1f}%")
        wandb.log({"eval/gsm8k_accuracy": acc, "train/global_step": 0}, step=0)

    # --- Training loop ---
    s_mode = "greedy" if args.student_temperature == 0 else f"temp={args.student_temperature}"
    print(f"Starting NAIL-Mixed (LoRA, beta={args.beta}, w_fwd={forward_weight}, "
          f"w_rev={reverse_weight}{', aux_sample' if args.aux_sample else ''}) "
          f"for {total_steps} steps (student={s_mode})"
          f"{' — resuming from step ' + str(start_step) if start_step else ''}")
    student.train()
    running_loss = 0.0
    running_fwd = 0.0
    running_rev = 0.0
    running_advantage = 0.0
    # Running rollout-diagnostic accumulators (averaged over logging_steps)
    running_pct_eos = 0.0
    running_pct_boxed = 0.0
    running_n_real = 0.0
    running_alpha = 0.0
    running_mean_seq_len = 0.0
    running_gen_len = 0.0
    # Gradient-clipping diagnostics
    running_pre_clip_grad_norm = 0.0
    running_grad_clipped = 0.0
    global_step = start_step

    def enforce_save_limit():
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                       key=lambda p: int(re.search(r"checkpoint-(\d+)", os.path.basename(p)).group(1)))
        while len(ckpts) > args.save_total_limit:
            import shutil
            shutil.rmtree(ckpts[0])
            ckpts = ckpts[1:]

    for epoch in range(start_epoch, args.num_train_epochs):
        for batch in dataloader:
            # Step-based hard cap — checked at top of iter so we don't even
            # rollout on the (would-be) post-cap step.
            if global_step >= total_steps:
                break

            prompt_ids_full = batch["input_ids"].to(device)
            prompt_mask_full = batch["attention_mask"].to(device)
            P = prompt_ids_full.shape[1]
            total_B = prompt_ids_full.shape[0]

            # === Student rollout via colocated vLLM (truly-online) ===
            # Sync the CURRENT LoRA into the engine EVERY step BEFORE generating,
            # so the rollout policy == the gradient policy (importance weight ≈ 1,
            # exactly as with the in-framework student.generate()). Then generate
            # the greedy/temperature rollout with vLLM and reassemble the same
            # [B, P+gen_len] tensor the loss code below expects.
            rollout_engine.sync(student)
            student_out_full = rollout_engine.generate(
                prompt_ids_full, prompt_mask_full, tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.student_temperature,
                stop_token_ids=vllm_stop_token_ids,
            )
            gen_len = student_out_full.shape[1] - P
            student.train()

            if gen_len == 0:
                continue

            full_mask_full = torch.cat([
                prompt_mask_full,
                torch.ones(total_B, gen_len, device=device, dtype=torch.long),
            ], dim=1)

            # === Global pad mask + denominator for the whole opt step ===
            # The pad mask is determined by the ROLLOUT tokens — any position
            # past a row's EOS is `pad_token_id`. Both arms exclude these.
            rollout_actions_full = student_out_full[:, P:]
            pad_mask_full = (rollout_actions_full != tokenizer.pad_token_id)
            total_n_real = pad_mask_full.sum().clamp(min=1).to(torch.float32)

            # === Rollout diagnostics (logged every logging_steps) ===
            diag = compute_rollout_diagnostics(
                rollout_actions_full, pad_mask_full,
                tokenizer, stop_token_ids,
            )

            # === Chunked expert/student forward + backward ===
            optimizer.zero_grad()
            chunk_loss_sum = 0.0
            chunk_fwd_sum_log = 0.0   # for separate forward-arm wandb logging
            chunk_rev_sum_log = 0.0   # for separate reverse-arm wandb logging
            chunk_adv_sum = 0.0
            for chunk_idx in range(n_chunks):
                start = chunk_idx * args.batch_size
                end = start + args.batch_size
                full_seq = student_out_full[start:end]
                full_mask = full_mask_full[start:end]
                rollout_actions = full_seq[:, P:]
                chunk_pad_mask_b = pad_mask_full[start:end]   # bool, [chunk_B, gen_len]

                # === Expert forward (shared by both arms) ===
                # We get raw expert logits at every answer position; the forward
                # arm uses them to sample an MC token (with optional temperature
                # scaling), and the reverse arm uses them to evaluate log π_E
                # at the chosen action (always with temperature scaling).
                with torch.no_grad():
                    e_logits = expert(input_ids=full_seq, attention_mask=full_mask).logits
                    e_answer_logits = e_logits[:, P - 1 : P + gen_len - 1, :]
                    if args.expert_temperature != 1.0:
                        e_answer_logits_scaled = e_answer_logits / args.expert_temperature
                    else:
                        e_answer_logits_scaled = e_answer_logits

                # === Student forward with grad (shared by both arms) ===
                p_logits = student(input_ids=full_seq, attention_mask=full_mask).logits
                p_answer_logits = p_logits[:, P - 1 : P + gen_len - 1, :]

                mask = chunk_pad_mask_b.to(p_answer_logits.dtype)

                # === Forward-KL arm: NLL on expert MC token ===
                if forward_weight > 0:
                    if args.expert_temperature == 0:
                        # argmax doesn't depend on scaling
                        expert_tokens = e_answer_logits.argmax(dim=-1)
                    else:
                        probs = F.softmax(e_answer_logits_scaled, dim=-1)
                        Bc, Tc, Vc = probs.shape
                        expert_tokens = torch.multinomial(
                            probs.reshape(Bc * Tc, Vc), num_samples=1
                        ).view(Bc, Tc)
                    log_p_at_expert = get_logprobs_for_actions(p_answer_logits, expert_tokens)
                    fwd_chunk_signed_sum = -(log_p_at_expert * mask).sum()   # positive (NLL)
                else:
                    fwd_chunk_signed_sum = torch.zeros(
                        (), device=device, dtype=p_answer_logits.dtype)

                # === Reverse-KL arm: IS-weighted advantage at chosen action ===
                if reverse_weight > 0:
                    if args.aux_sample:
                        # Draw aux student token from p_θ at each prefix.
                        with torch.no_grad():
                            aux_probs = F.softmax(
                                p_answer_logits.detach().float(), dim=-1)
                            Bc, Tc, Vc = aux_probs.shape
                            actions = torch.multinomial(
                                aux_probs.reshape(Bc * Tc, Vc), num_samples=1
                            ).view(Bc, Tc)
                        log_q = get_logprobs_for_actions(p_answer_logits.detach(), actions)
                    else:
                        # Reuse the rollout token. Run a second (no-grad)
                        # student forward to recover log_q at the rollout token.
                        with torch.no_grad():
                            q_logits = student(
                                input_ids=full_seq, attention_mask=full_mask,
                            ).logits
                            q_answer_logits = q_logits[:, P - 1 : P + gen_len - 1, :]
                            log_q = get_logprobs_for_actions(q_answer_logits, rollout_actions)
                        actions = rollout_actions

                    log_expert = get_logprobs_for_actions(e_answer_logits_scaled, actions)
                    advantage = log_expert - log_q

                    log_p_at_action = get_logprobs_for_actions(p_answer_logits, actions)
                    importance_weight = torch.exp(log_p_at_action - log_q.detach())

                    rev_per_token = importance_weight * advantage.detach() * mask
                    # rev loss = -mean (IW * A); we accumulate the signed sum.
                    rev_chunk_signed_sum = -rev_per_token.sum()
                    chunk_adv_sum += (advantage * mask).sum().item()
                else:
                    rev_chunk_signed_sum = torch.zeros(
                        (), device=device, dtype=p_answer_logits.dtype)

                # === Combine and backprop (global denominator) ===
                chunk_loss_signed_sum = (
                    forward_weight * fwd_chunk_signed_sum
                    + reverse_weight * rev_chunk_signed_sum
                )
                loss = chunk_loss_signed_sum / total_n_real
                loss.backward()

                # Logging: surface each arm's contribution per real token.
                chunk_loss_sum += chunk_loss_signed_sum.item()
                chunk_fwd_sum_log += fwd_chunk_signed_sum.item()
                chunk_rev_sum_log += rev_chunk_signed_sum.item()

            # Normalize logging quantities by global denominator so they're
            # comparable across opt steps regardless of pad fraction.
            total_n_real_f = total_n_real.item()
            running_loss += chunk_loss_sum / total_n_real_f
            running_fwd  += chunk_fwd_sum_log / total_n_real_f
            running_rev  += chunk_rev_sum_log / total_n_real_f
            running_advantage += chunk_adv_sum / total_n_real_f

            # Accumulate rollout diagnostics
            running_pct_eos      += diag["pct_eos"]
            running_pct_boxed    += diag["pct_boxed"]
            running_n_real       += diag["n_real"]
            running_alpha        += diag["alpha"]
            running_mean_seq_len += diag["mean_seq_len"]
            running_gen_len      += diag["gen_len"]

            # Gradient clipping (matches small-cot's grad_clip / NeMo-RL's max_grad_norm).
            # Always compute the pre-clip norm — even when max_grad_norm=0 (clip
            # disabled) we want the diagnostic so we can compare clipped vs
            # unclipped runs. Only the LoRA params have requires_grad=True, so
            # we clip just those.
            trainable_params = [p for p in student.parameters() if p.requires_grad]
            clip_value = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=clip_value,
            )
            running_pre_clip_grad_norm += float(pre_clip_grad_norm)
            running_grad_clipped += float(
                args.max_grad_norm > 0 and pre_clip_grad_norm > args.max_grad_norm
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / args.logging_steps
                avg_fwd  = running_fwd  / args.logging_steps
                avg_rev  = running_rev  / args.logging_steps
                avg_adv  = running_advantage / args.logging_steps
                avg_pct_eos      = running_pct_eos      / args.logging_steps
                avg_pct_boxed    = running_pct_boxed    / args.logging_steps
                avg_n_real       = running_n_real       / args.logging_steps
                avg_alpha        = running_alpha        / args.logging_steps
                avg_mean_seq_len = running_mean_seq_len / args.logging_steps
                avg_gen_len      = running_gen_len      / args.logging_steps
                avg_pre_clip_grad_norm = running_pre_clip_grad_norm / args.logging_steps
                avg_grad_clipped       = running_grad_clipped       / args.logging_steps
                lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss":          avg_loss,
                    "train/loss_forward":  avg_fwd,
                    "train/loss_reverse":  avg_rev,
                    "train/advantage":     avg_adv,
                    "train/learning_rate": lr,
                    "train/global_step":   global_step,
                    "train/pre_clip_grad_norm": avg_pre_clip_grad_norm,
                    "train/grad_clipped":      avg_grad_clipped,
                    "rollout/pct_eos":     avg_pct_eos,
                    "rollout/pct_boxed":   avg_pct_boxed,
                    "rollout/n_real":      avg_n_real,
                    "rollout/alpha":       avg_alpha,
                    "rollout/mean_seq_len":avg_mean_seq_len,
                    "rollout/gen_len":     avg_gen_len,
                }, step=global_step)
                print(f"Step {global_step}/{total_steps} | "
                      f"loss: {avg_loss:.4f} (fwd: {avg_fwd:.4f}, rev: {avg_rev:.4f}) | "
                      f"adv: {avg_adv:.4f} | EOS%: {100*avg_pct_eos:.1f} | "
                      f"boxed%: {100*avg_pct_boxed:.1f} | α: {avg_alpha:.3f} | "
                      f"n_real: {avg_n_real:.0f} | "
                      f"len(mean/gen): {avg_mean_seq_len:.0f}/{avg_gen_len:.0f} | "
                      f"|g|: {avg_pre_clip_grad_norm:.3f} (clip%: {100*avg_grad_clipped:.0f}) | "
                      f"lr: {lr:.2e}")
                running_loss = 0.0
                running_fwd = 0.0
                running_rev = 0.0
                running_advantage = 0.0
                running_pct_eos = 0.0
                running_pct_boxed = 0.0
                running_n_real = 0.0
                running_alpha = 0.0
                running_mean_seq_len = 0.0
                running_gen_len = 0.0
                running_pre_clip_grad_norm = 0.0
                running_grad_clipped = 0.0

            if args.eval_source and global_step % args.eval_steps == 0:
                print(f"\n[Step {global_step}] Running GSM8K eval...")
                torch.cuda.empty_cache()
                acc, n_correct, n_total = evaluate_on_gsm8k(
                    student, tokenizer, args.eval_source,
                    max_examples=args.max_eval_examples,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size,
                )
                print(f"  GSM8K accuracy: {n_correct}/{n_total} = {acc * 100:.1f}%")
                wandb.log({"eval/gsm8k_accuracy": acc, "train/global_step": global_step},
                          step=global_step)

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_checkpoint(save_path, student, tokenizer, optimizer, scheduler,
                                global_step, epoch)
                print(f"  Saved to {save_path}")
                enforce_save_limit()

                # GSM8K eval loss every save_steps
                if args.gsm8k_eval_loss_data and os.path.exists(args.gsm8k_eval_loss_data):
                    torch.cuda.empty_cache()
                    eval_loss, n_eval = compute_eval_loss(
                        student, tokenizer, args.gsm8k_eval_loss_data,
                        max_length=2048,
                        batch_size=args.gsm8k_eval_loss_batch_size,
                        prompt_field="question", completion_field="answer",
                        system_prompt=args.system_prompt,
                    )
                    print(f"  GSM8K eval_loss: {eval_loss:.4f} (n={n_eval})")
                    wandb.log({"eval/loss": eval_loss,
                               "train/global_step": global_step}, step=global_step)

            # Step-based hard cap. Once we reach `total_steps`, training stops
            # regardless of which epoch we're in. This makes the script robust
            # to preemption + resume — the dataloader restarts from epoch 0
            # on resume, but we still terminate exactly at `total_steps`.
            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    # --- Final save ---
    final_path = os.path.join(args.output_dir, "final")
    save_checkpoint(final_path, student, tokenizer, optimizer, scheduler,
                    global_step, args.num_train_epochs)
    print(f"Final adapter saved to {final_path}")

    if args.eval_source:
        print(f"\nFinal eval on {args.eval_source}...")
        torch.cuda.empty_cache()
        acc, n_correct, n_total = evaluate_on_gsm8k(
            student, tokenizer, args.eval_source,
            max_examples=None,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
        )
        print(f"Final GSM8K accuracy: {n_correct}/{n_total} = {acc * 100:.1f}%")
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump({"accuracy": round(acc, 4), "n_correct": n_correct, "n_total": n_total}, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
