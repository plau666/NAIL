"""Shared vLLM rollout backend for the on-policy NAIL trainers.

The forward, reverse, and mixed trainers differ in their losses, but all consume
the same rollout representation: a `[B, P + gen_len]` token tensor with
left-padded prompts followed by generated tokens right-padded with `pad_id`.
The trainers then re-run an HF forward pass with gradients to compute their
losses. This wrapper isolates vLLM generation from loss computation, expert
scoring, masking, clipping, and evaluation.

This module wraps:
  * a colocated vLLM engine holding the base student + a hot-swappable LoRA, and
  * a `sync()` method that pushes the current training LoRA into the engine each
    step by writing the adapter to tmpfs and re-registering it under a fresh
    `lora_int_id`. vLLM caches adapters by id, so a new id forces a reload of
    the just-written weights.

`generate()` returns a tensor in the exact `student.generate()` layout so the
existing trainer code can use it as a drop-in replacement for the
`student_out_full = student.generate(...)` call.
"""

import os
import time

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class VLLMRolloutGenerator:
    def __init__(self, base_model, max_lora_rank=128, gpu_memory_utilization=0.35,
                 max_model_len=4096, adapter_dir="/dev/shm/nail_vllm_adapter",
                 dtype="bfloat16", seed=0, enforce_eager=False):
        """Stand up a colocated vLLM engine.

        gpu_memory_utilization is intentionally LOW: the training process (base
        student + LoRA + optimizer + frozen 1B expert + activations) already
        holds GPU memory, and vLLM grabs `gpu_memory_utilization` of *total* GPU
        memory for weights + KV cache on top of that. Tune so the two fit.
        """
        self.adapter_dir = adapter_dir
        os.makedirs(self.adapter_dir, exist_ok=True)
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=enforce_eager,   # False => CUDA graphs (faster decode)
        )
        self._lora_id = 0
        self._cur_lora = None
        # cumulative timing diagnostics
        self.sync_time = 0.0
        self.gen_time = 0.0
        self.n_sync = 0
        self.n_gen = 0

    @torch.no_grad()
    def sync(self, peft_model):
        """Push the current LoRA weights into the engine.

        Call once per step before `generate()` so rollouts use the same policy
        whose gradients are being computed.

        Writes the adapter to tmpfs and bumps the lora id to force a reload.
        Returns the wall-clock seconds spent (the per-step sync tax).
        """
        t0 = time.time()
        # PeftModel.save_pretrained writes adapter_config.json + safetensors only
        # (the frozen base is not written). To tmpfs this is a ~100MB memcpy.
        peft_model.save_pretrained(self.adapter_dir, safe_serialization=True)
        self._lora_id += 1
        self._cur_lora = LoRARequest(
            lora_name=f"step{self._lora_id}",
            lora_int_id=self._lora_id,
            lora_path=self.adapter_dir,
        )
        # Force the (lazy) load now so the cost is attributed to sync, not gen:
        # a 1-token dummy decode pins the new adapter into the engine.
        self.llm.generate(
            [{"prompt_token_ids": [2]}],
            SamplingParams(max_tokens=1, temperature=0.0),
            lora_request=self._cur_lora,
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        dt = time.time() - t0
        self.sync_time += dt
        self.n_sync += 1
        return dt

    @torch.no_grad()
    def generate(self, prompt_ids_full, prompt_mask_full, pad_id,
                 max_new_tokens, temperature=0.0, stop_token_ids=None,
                 seed=None):
        """Drop-in for `student.generate(...)`.

        Returns `student_out_full` shaped `[B, P + gen_len]`: the left-padded
        prompt columns (verbatim from `prompt_ids_full`) followed by the
        generated tokens right-padded with `pad_id` to the batch-max gen length.
        """
        assert self._cur_lora is not None, "call sync() before generate()"
        device = prompt_ids_full.device
        B, P = prompt_ids_full.shape

        # Strip left-padding -> real prompt token-id list per row.
        prompts = []
        for ids, mask in zip(prompt_ids_full, prompt_mask_full):
            real_ids = ids[mask.bool()].tolist()
            prompts.append({"prompt_token_ids": real_ids})

        # NOTE: per-request seeding was tried for reproducibility but does NOT
        # achieve it on this hardware: temp-1 vLLM has a ~16% irreducible
        # token-flip rate from GPU FP nondeterminism even with identical
        # seeds/inputs back-to-back. Rollouts are stochastic training data, so
        # engine-seed-only (set at LLM init) is the standard. `seed` is left in
        # the signature but applies one value to the whole batch (None = unseeded).
        sp = SamplingParams(
            n=1, temperature=temperature, top_p=1.0,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            seed=seed,
        )
        t0 = time.time()
        outs = self.llm.generate(prompts, sp, lora_request=self._cur_lora,
                                 use_tqdm=False)
        torch.cuda.synchronize()
        self.gen_time += time.time() - t0
        self.n_gen += 1

        gens = [list(o.outputs[0].token_ids) for o in outs]
        max_gen = max((len(g) for g in gens), default=0)
        if max_gen == 0:
            # nothing generated; signal empty like gen_len==0
            return prompt_ids_full

        gen_tensor = torch.full((B, max_gen), pad_id, dtype=prompt_ids_full.dtype)
        for i, g in enumerate(gens):
            if g:
                gen_tensor[i, : len(g)] = torch.tensor(g, dtype=prompt_ids_full.dtype)
        gen_tensor = gen_tensor.to(device)
        return torch.cat([prompt_ids_full, gen_tensor], dim=1)

    def timing_summary(self):
        s = self.sync_time / max(self.n_sync, 1)
        g = self.gen_time / max(self.n_gen, 1)
        return {"mean_sync_s": s, "mean_gen_s": g,
                "n_sync": self.n_sync, "n_gen": self.n_gen}
