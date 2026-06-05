"""Mixed-KL on-policy distillation with LoRA.

Blend of the forward-KL (MC, hard-label CE on the expert's sampled token) and
reverse-KL (importance-weighted policy gradient) objectives on the **same**
greedy student prefix:

    L(theta) = (1 - beta) * L_forward + beta * L_reverse

For NAIL (greedy rollouts) the reverse arm should draw a fresh auxiliary
student token at each prefix (`--aux_sample`) to keep the estimator unbiased;
the rollout-token shortcut is biased when student_temperature == 0. The
forward arm is the same MC forward-KL surrogate as `forward_lora.py`: sample
y_t ~ pi_E(.|s_t) and take student NLL on y_t.

Usage:
    python mixed_lora.py \
        --student_model google/gemma-3-270m-it \
        --expert_model google/gemma-3-1b-it \
        --train_data data/tinygsm/tinygsm_400k.jsonl \
        --output_dir output/mixed_lora_r128_beta0p5 \
        --name mixed_lora_r128_beta0p5 \
        --beta 0.5 --aux_sample --student_temperature 0.0

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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_utils import (
    evaluate_on_gsm8k,
    compute_eval_loss,
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
    """log pi(a|s): log_softmax then gather."""
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
    parser = argparse.ArgumentParser(description="Mixed-KL (forward + reverse) with LoRA")

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
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--student_temperature", type=float, default=0.0,
                        help="Student rollout temperature. 0 = greedy (NAIL default)")
    parser.add_argument("--expert_temperature", type=float, default=1.0,
                        help="Forward arm: temperature for sampling the expert's token; "
                             "0 = argmax. Reverse arm: scaling applied to expert logits "
                             "when computing log pi_E(a|s) (default 1.0 keeps the expert "
                             "unchanged; >1 is the noisy expert used by NAIL).")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Mixing weight. loss = (1-beta)*forward + beta*reverse. "
                             "beta=0 is pure forward (NAIL-F), beta=1 is pure reverse "
                             "(NAIL-R).")
    parser.add_argument("--aux_sample", action="store_true", default=False,
                        help="Reverse arm only: draw a fresh auxiliary student token at "
                             "each prefix as the reverse-KL MC sample. Recommended for "
                             "NAIL (greedy rollouts) — the rollout-token shortcut is "
                             "biased when student_temperature == 0.")

    parser.add_argument("--eval_steps", type=int, default=999999)
    parser.add_argument("--max_eval_examples", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--skip_initial_eval", action="store_true", default=False)

    # GSM8K eval loss (runs every save_steps if set)
    parser.add_argument("--gsm8k_eval_loss_data", type=str, default=None,
                        help="Raw GSM8K jsonl (question/answer fields) for periodic eval loss. "
                             "Computed every save_steps.")
    parser.add_argument("--gsm8k_eval_loss_batch_size", type=int, default=4,
                        help="Per-device batch size for GSM8K eval loss. Keep low (4) — "
                             "gemma-3 262k vocab fp32 logits are memory-heavy.")

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

    if not (0.0 <= args.beta <= 1.0):
        raise ValueError(f"--beta must be in [0, 1], got {args.beta}")
    forward_weight = 1.0 - args.beta
    reverse_weight = args.beta
    run_forward_arm = forward_weight > 0.0
    run_reverse_arm = reverse_weight > 0.0

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

    # --- Generation config ---
    if args.student_temperature == 0:
        gen_config = GenerationConfig(
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        gen_config = GenerationConfig(
            do_sample=True, temperature=args.student_temperature, top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
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
    print(f"Starting Mixed-KL (LoRA) for {total_steps} steps "
          f"(student={s_mode}, beta={args.beta}, aux_sample={args.aux_sample})"
          f"{' — resuming from step ' + str(start_step) if start_step else ''}")
    student.train()
    running_loss = 0.0
    running_forward = 0.0
    running_reverse = 0.0
    running_advantage = 0.0
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
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            prompt_ids_full = batch["input_ids"].to(device)
            prompt_mask_full = batch["attention_mask"].to(device)
            P = prompt_ids_full.shape[1]
            total_B = prompt_ids_full.shape[0]

            # === Student rollout (greedy by default for NAIL) ===
            student.eval()
            if args.gradient_checkpointing:
                student.gradient_checkpointing_disable()
                student.config.use_cache = True
            with torch.no_grad():
                student_out_full = student.generate(
                    input_ids=prompt_ids_full,
                    attention_mask=prompt_mask_full,
                    generation_config=gen_config,
                )
                gen_len = student_out_full.shape[1] - P
            if args.gradient_checkpointing:
                student.config.use_cache = False
                student.gradient_checkpointing_enable()
            student.train()

            if gen_len == 0:
                continue

            full_mask_full = torch.cat([
                prompt_mask_full,
                torch.ones(total_B, gen_len, device=device, dtype=torch.long),
            ], dim=1)

            # === Chunked expert/student forward + backward ===
            optimizer.zero_grad()
            chunk_loss_sum = 0.0
            chunk_forward_sum = 0.0
            chunk_reverse_sum = 0.0
            chunk_adv_sum = 0.0
            for chunk_idx in range(n_chunks):
                start = chunk_idx * args.batch_size
                end = start + args.batch_size
                full_seq = student_out_full[start:end]
                full_mask = full_mask_full[start:end]
                rollout_actions = full_seq[:, P:]

                # Expert forward — shared by both arms (raw logits + temp-scaled copy).
                with torch.no_grad():
                    e_logits = expert(input_ids=full_seq, attention_mask=full_mask).logits
                    e_answer_logits = e_logits[:, P - 1 : P + gen_len - 1, :]
                    B, T, V = e_answer_logits.shape

                # Student loss-time forward (with grad). Shared by both arms.
                p_logits = student(input_ids=full_seq, attention_mask=full_mask).logits
                p_answer_logits = p_logits[:, P - 1 : P + gen_len - 1, :]

                # Detached zero scaffolding so the autograd graph stays intact
                # at beta=0 or beta=1 when one arm contributes no gradient.
                zero_loss = p_answer_logits.sum() * 0.0
                forward_loss = zero_loss
                reverse_loss = zero_loss
                advantage_mean_val = 0.0

                # === Forward arm: MC forward KL (NAIL-F) ===
                # Sample expert token at expert_temperature (0 = argmax) and take
                # student CE on it. Equivalent to forward_lora.py.
                if run_forward_arm:
                    with torch.no_grad():
                        if args.expert_temperature == 0:
                            expert_tokens = e_answer_logits.argmax(dim=-1)
                        else:
                            scaled = e_answer_logits / args.expert_temperature
                            probs = F.softmax(scaled.float(), dim=-1)
                            expert_tokens = torch.multinomial(
                                probs.view(B * T, V), num_samples=1,
                            ).view(B, T)
                    log_p_expert = get_logprobs_for_actions(p_answer_logits, expert_tokens)
                    # No pad mask: NAIL rollouts are greedy and don't contain
                    # pad tokens; OPD-style padded rollouts would mask here.
                    forward_loss = -log_p_expert.mean()

                # === Reverse arm: importance-weighted TM (NAIL-R / OPD-R) ===
                if run_reverse_arm:
                    if args.expert_temperature != 1.0:
                        e_answer_logits_rev = e_answer_logits / args.expert_temperature
                    else:
                        e_answer_logits_rev = e_answer_logits

                    if args.aux_sample:
                        with torch.no_grad():
                            aux_probs = F.softmax(p_answer_logits.detach().float(), dim=-1)
                            actions = torch.multinomial(
                                aux_probs.reshape(B * T, V), num_samples=1,
                            ).view(B, T)
                        log_q = get_logprobs_for_actions(p_answer_logits.detach(), actions)
                        answer_mask = torch.ones_like(actions, dtype=p_answer_logits.dtype)
                    else:
                        with torch.no_grad():
                            q_logits = student(
                                input_ids=full_seq, attention_mask=full_mask,
                            ).logits
                            q_answer_logits = q_logits[:, P - 1 : P + gen_len - 1, :]
                            log_q = get_logprobs_for_actions(q_answer_logits, rollout_actions)
                        actions = rollout_actions
                        answer_mask = (actions != tokenizer.pad_token_id).float()

                    log_expert = get_logprobs_for_actions(e_answer_logits_rev, actions)
                    advantage = log_expert - log_q
                    log_p = get_logprobs_for_actions(p_answer_logits, actions)
                    importance_weight = torch.exp(log_p - log_q.detach())
                    num_tokens = answer_mask.sum().clamp(min=1)
                    reverse_loss = -(importance_weight * advantage.detach() * answer_mask).sum() / num_tokens
                    advantage_mean_val = (advantage * answer_mask).sum().item() / num_tokens.item()

                # === Mix and backprop ===
                loss = forward_weight * forward_loss + reverse_weight * reverse_loss
                loss = loss / n_chunks
                loss.backward()

                chunk_loss_sum += loss.item() * n_chunks
                chunk_forward_sum += forward_loss.item()
                chunk_reverse_sum += reverse_loss.item()
                chunk_adv_sum += advantage_mean_val

            running_loss += chunk_loss_sum / n_chunks
            running_forward += chunk_forward_sum / n_chunks
            running_reverse += chunk_reverse_sum / n_chunks
            running_advantage += chunk_adv_sum / n_chunks

            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / args.logging_steps
                avg_fwd = running_forward / args.logging_steps
                avg_rev = running_reverse / args.logging_steps
                avg_adv = running_advantage / args.logging_steps
                lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": avg_loss,
                    "train/forward_loss": avg_fwd,
                    "train/reverse_loss": avg_rev,
                    "train/advantage": avg_adv,
                    "train/beta": args.beta,
                    "train/learning_rate": lr,
                    "train/global_step": global_step,
                }, step=global_step)
                print(f"Step {global_step}/{total_steps} | loss: {avg_loss:.4f} | "
                      f"fwd: {avg_fwd:.4f} | rev: {avg_rev:.4f} | "
                      f"adv: {avg_adv:.4f} | lr: {lr:.2e}")
                running_loss = 0.0
                running_forward = 0.0
                running_reverse = 0.0
                running_advantage = 0.0

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

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
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
