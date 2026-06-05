"""LogLossBC with LoRA: SFT a small student on pre-generated (noisy) teacher
rollouts using low-rank adapters, with full training-state checkpointing
for resume.

  - Wraps the student with PEFT LoRA (rank / alpha / dropout configurable).
  - Default LoRA targets: all standard attention + MLP projection modules.
  - Uses HF Trainer which automatically saves optimizer/scheduler/RNG/step
    state inside each checkpoint directory, enabling `--resume_from_checkpoint`.

Usage:
    python offline_bc_lora.py \
        --student_model google/gemma-3-270m-it \
        --train_data data/teacher_rollouts/train.jsonl \
        --output_dir output/obc_lora_r128 \
        --name obc_lora_r128 \
        --lora_rank 128

    # Resume from latest checkpoint:
    python offline_bc_lora.py ... --resume_from_checkpoint auto

    # Resume from specific checkpoint:
    python offline_bc_lora.py ... --resume_from_checkpoint output/obc_lora_r128/checkpoint-200
"""

import argparse
import glob
import json
import os
import re

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

# Shared GSM utilities (dataset, collator, callbacks, answer extraction)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_utils import (
    RolloutSFTDataset, SFTDataCollator, GSM8KAccuracyCallback,
    evaluate_on_gsm8k, SYSTEM_PROMPT,
)


# Default LoRA target modules: covers Gemma / LLaMA / Qwen architectures.
# (attention projections + MLP projections)
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def resolve_resume_checkpoint(output_dir, arg_value):
    """Resolve --resume_from_checkpoint:
      - None/empty/"none" → None
      - "auto" → latest checkpoint-* in output_dir, or None if none exists
      - other string → used as-is (path)
    """
    if not arg_value or arg_value.lower() == "none":
        return None
    if arg_value.lower() == "auto":
        ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not ckpts:
            return None
        # Pick by step number (highest)
        def step_of(p):
            m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        latest = max(ckpts, key=step_of)
        return latest
    return arg_value


def main():
    parser = argparse.ArgumentParser(description="Offline BC with LoRA for real math")

    parser.add_argument("--student_model", type=str, required=True,
                        help="HuggingFace model ID or local path for the base model")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, nargs="+", default=None,
                        help="One or more eval JSONL files for eval loss (same fields as train)")
    parser.add_argument("--gsm8k_eval_loss_data", type=str, default=None,
                        help="Raw GSM8K jsonl (with 'question' and 'answer' fields) to compute "
                             "eval loss on the GSM8K test split every `save_steps`.")
    parser.add_argument("--gsm8k_eval_loss_batch_size", type=int, default=4,
                        help="Per-device eval batch size for GSM8K loss eval. Default 4 (conservative "
                             "since fp32 logits on gemma-3's 262k vocab are memory-heavy).")
    parser.add_argument("--eval_source", type=str, default=None,
                        help="Original GSM8K-format eval data for generation accuracy")
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--completion_field", type=str, default="completion")
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--max_length", type=int, default=2048)

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="NAIL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # Eval
    parser.add_argument("--max_eval_examples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--skip_initial_eval", action="store_true", default=False)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (defaults to 2*rank if unset)")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=None,
                        help=f"LoRA target modules (default: {DEFAULT_LORA_TARGETS})")

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to resume from latest checkpoint in output_dir, "
                             "or a path to a specific checkpoint dir. "
                             "HF Trainer restores model/optimizer/scheduler/RNG/step.")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    resume_path = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_path:
        print(f"Resuming from: {resume_path}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- Base model ---
    print(f"Loading base model: {args.student_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # --- LoRA wrap ---
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
        # Load adapter weights from the checkpoint.
        # HF Trainer will separately restore optimizer/scheduler/RNG/step from the same dir.
        print(f"Loading LoRA adapter from {resume_path}")
        model = PeftModel.from_pretrained(base_model, resume_path, is_trainable=True)
    else:
        print(f"Creating fresh LoRA adapter (rank={args.lora_rank}, "
              f"alpha={lora_alpha}, targets={target_modules})")
        model = get_peft_model(base_model, lora_config)

    # Report trainable params
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_total:,} | Trainable (LoRA): {n_trainable:,} "
          f"({100 * n_trainable / n_total:.3f}%)")
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Data ---
    print(f"Loading training data from {args.train_data}")
    train_dataset = RolloutSFTDataset(
        args.train_data, tokenizer, max_length=args.max_length,
        prompt_field=args.prompt_field, completion_field=args.completion_field,
        system_prompt=args.system_prompt,
    )

    eval_dataset = None
    eval_datasets = {}
    if args.eval_data:
        for path in args.eval_data:
            if not os.path.exists(path):
                print(f"WARNING: eval data not found, skipping: {path}")
                continue
            name = os.path.splitext(os.path.basename(path))[0]
            if name in eval_datasets:
                name = os.path.basename(os.path.dirname(path)) + "_" + name
            print(f"Loading eval data '{name}' from {path}")
            eval_datasets[name] = RolloutSFTDataset(
                path, tokenizer, max_length=args.max_length,
                prompt_field=args.prompt_field, completion_field=args.completion_field,
                system_prompt=args.system_prompt,
            )

    # GSM8K raw test-set eval loss (computed on the ground-truth `answer` completion).
    # Uses `question`/`answer` field names since that's the raw GSM8K format.
    if args.gsm8k_eval_loss_data and os.path.exists(args.gsm8k_eval_loss_data):
        print(f"Loading GSM8K eval-loss data from {args.gsm8k_eval_loss_data}")
        eval_datasets["gsm8k_test"] = RolloutSFTDataset(
            args.gsm8k_eval_loss_data, tokenizer, max_length=args.max_length,
            prompt_field="question", completion_field="answer",
            system_prompt=args.system_prompt,
        )

    if len(eval_datasets) == 1:
        eval_dataset = next(iter(eval_datasets.values()))
    elif len(eval_datasets) > 1:
        eval_dataset = eval_datasets

    # --- Training args ---
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_DIR", os.path.dirname(os.path.abspath(__file__)))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.01},
        weight_decay=args.weight_decay,
        max_grad_norm=1e9,  # effectively disable grad clipping (HF default is 1.0)
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        # If GSM8K loss eval is active, tie cadence to save_steps so every saved
        # ckpt has an accompanying eval_loss datapoint.
        eval_steps=(args.save_steps if args.gsm8k_eval_loss_data
                    else (args.eval_steps if eval_dataset else None)),
        bf16=args.bf16,
        report_to="wandb",
        run_name=args.name,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        per_device_eval_batch_size=(args.gsm8k_eval_loss_batch_size
                                    if args.gsm8k_eval_loss_data
                                    else args.eval_batch_size),
        save_strategy="steps",
        # HF Trainer saves optimizer.pt, scheduler.pt, rng_state.pth, trainer_state.json
        # inside each checkpoint-XXXX dir automatically — needed for resume.
    )

    # --- Callbacks ---
    callbacks = []
    if args.eval_source and os.path.exists(args.eval_source):
        gsm8k_callback = GSM8KAccuracyCallback(
            tokenizer=tokenizer,
            eval_data_path=args.eval_source,
            eval_steps=args.eval_steps,
            max_eval_examples=args.max_eval_examples,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            skip_initial_eval=args.skip_initial_eval,
        )
        callbacks.append(gsm8k_callback)
        print(f"GSM8K accuracy eval enabled every {args.eval_steps} steps")

    # --- Trainer ---
    collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)

    import transformers
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if int(transformers.__version__.split(".")[0]) >= 5:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    # --- Train ---
    print("Starting training...")
    # HF Trainer saves LoRA adapter + optimizer.pt + scheduler.pt + rng_state.pth +
    # trainer_state.json inside each checkpoint-XXXX dir. Passing resume_from_checkpoint
    # restores ALL of these (step, optimizer momentums, LR schedule position, RNG).
    trainer.train(resume_from_checkpoint=resume_path)

    # --- Save final ---
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Adapter saved to {final_path}")

    # --- Final GSM8K eval ---
    if args.eval_source and os.path.exists(args.eval_source):
        print(f"\nRunning final eval on {args.eval_source}...")
        accuracy, n_correct, n_total = evaluate_on_gsm8k(
            model, tokenizer, args.eval_source, max_examples=None,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
        )
        print(f"Final GSM8K accuracy: {n_correct}/{n_total} = {accuracy * 100:.1f}%")
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump({
                "eval_data": args.eval_source,
                "accuracy": round(accuracy, 4),
                "n_correct": n_correct,
                "n_total": n_total,
            }, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
