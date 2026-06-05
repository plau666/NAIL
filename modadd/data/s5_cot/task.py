"""S5 composition task used by the synthetic paper-method experiments.

The paper's S5 examples are symbolic permutation-composition chains. This file
defines the token vocabulary, clean CoT generation, corruptible value tokens,
and clean autoregressive evaluation metrics.
"""

import itertools
import random
import torch

from data.s5_cot.prompt_bank import build_xy_from_prompt_and_target
from data.synthetic.corruption import (
    corrupt_ids as generic_corrupt_ids,
    corrupt_token as generic_corrupt_token,
)
from data.synthetic.eval import (
    estimate_saved_clean_train_loss as shared_estimate_saved_clean_train_loss,
    evaluate_saved_clean_metrics,
    greedy_generate_target_ids_batched as shared_greedy_generate_target_ids_batched,
    teacher_forced_exact_batch as shared_teacher_forced_exact_batch,
)

TOKENS = ['(', ')', '='] + [str(i) for i in range(1, 6)]
stoi = {t: i for i, t in enumerate(TOKENS)}
itos = {i: t for t, i in stoi.items()}
VOCAB_SIZE = len(TOKENS)  # 8
LPAREN_ID = stoi['(']
RPAREN_ID = stoi[')']
EQUALS_ID = stoi['=']
DIGIT_START_ID = stoi['1']
DIGIT_END_ID = stoi['5']
ALL_PERMS = tuple(itertools.permutations(range(1, 6)))
ENCODED_ALL_PERMS = tuple(
    (LPAREN_ID, *(digit + 2 for digit in perm), RPAREN_ID)
    for perm in ALL_PERMS
)

def encode(tokens):
    return [stoi[t] for t in tokens]

def decode(ids):
    return [itos[int(i)] for i in ids]

def compose(sigma, pi):
    # paper: sigma o pi = (sigma_{pi_1}, ..., sigma_{pi_5})
    return [sigma[j - 1] for j in pi]

def perm_tokens(perm):
    return ['('] + [str(x) for x in perm] + [')']

def sample_perm():
    p = [1, 2, 3, 4, 5]
    random.shuffle(p)
    return p

def sample_cot_example(m=21):
    prompt = []
    cot = []
    running = None

    for _ in range(m):
        sigma = sample_perm()
        prompt.extend(perm_tokens(sigma))
        running = sigma if running is None else compose(running, sigma)
        cot.extend(perm_tokens(running))

    prompt.append('=')
    return prompt, cot

def sample_base_example(m=21):
    prompt = []
    running = None

    for _ in range(m):
        sigma = sample_perm()
        prompt.extend(perm_tokens(sigma))
        running = sigma if running is None else compose(running, sigma)

    prompt.append('=')
    target = perm_tokens(running)
    return prompt, target

def sample_xy(mode='cot', m=21):
    if mode == 'cot':
        prompt, target = sample_cot_example(m=m)
    elif mode == 'base':
        prompt, target = sample_base_example(m=m)
    else:
        raise ValueError(mode)

    seq = prompt + target
    ids = encode(seq)

    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)

    prompt_len = len(prompt)
    y[:prompt_len - 1] = -1  # only compute loss on continuation
    return x, y

def get_batch(batch_size, device, mode='cot', m=21):
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = sample_xy(mode=mode, m=m)
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    return X, Y

def continuation_exact_from_logits(logits, y):
    pred = logits.argmax(dim=-1)
    mask = (y != -1)
    ok = torch.logical_or(pred.eq(y), ~mask)
    return ok.all(dim=1).float().mean().item()

# for training the noisy expert

def sample_prompt_only(m=21):
    prompt = []
    for _ in range(m):
        sigma = sample_perm()
        prompt.extend(perm_tokens(sigma))
    prompt.append('=')
    return prompt

# Only permutation value tokens are semantically corruptible. Parentheses and
# "=" are scaffold tokens, so noisy laws leave them clean or force them valid.
CORRUPTIBLE_IDS = [stoi[str(i)] for i in range(1, 6)]  # for S5

def corrupt_token(tok_id, eta):
    return generic_corrupt_token(
        tok_id,
        eta,
        corruptible_token_ids=CORRUPTIBLE_IDS,
    )

def corrupt_ids(ids, eta):
    return generic_corrupt_ids(
        ids,
        eta,
        corruptible_token_ids=CORRUPTIBLE_IDS,
    )

def sample_perm_from_rng(rng):
    p = [1, 2, 3, 4, 5]
    rng.shuffle(p)
    return p

def compose_perm(sigma, pi):
    return tuple(sigma[j - 1] for j in pi)

def sample_cot_example_ids_from_rng(rng, m=21):
    prompt_ids = []
    cot_ids = []
    running = None
    for _ in range(m):
        perm_idx = rng.randrange(len(ALL_PERMS))
        sigma = ALL_PERMS[perm_idx]
        prompt_ids.extend(ENCODED_ALL_PERMS[perm_idx])
        running = sigma if running is None else compose_perm(running, sigma)
        cot_ids.extend((LPAREN_ID, *(digit + 2 for digit in running), RPAREN_ID))
    prompt_ids.append(EQUALS_ID)
    return prompt_ids, cot_ids

def sample_cot_example_from_rng(rng, m=21):
    prompt = []
    cot = []
    running = None
    for _ in range(m):
        sigma = sample_perm_from_rng(rng)
        prompt.extend(perm_tokens(sigma))
        running = sigma if running is None else compose(running, sigma)
        cot.extend(perm_tokens(running))
    prompt.append('=')
    return prompt, cot

@torch.no_grad()
def greedy_generate_ids(model, prompt_ids, max_new_tokens, device):
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    full_seq = torch.empty((1, prompt.size(1) + max_new_tokens), dtype=torch.long, device=device)
    full_seq[:, :prompt.size(1)] = prompt
    input_ids = prompt
    past_key_values = None

    for step in range(max_new_tokens):
        logits, _, past_key_values = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        full_seq[:, prompt.size(1) + step] = next_id.squeeze(1)
        input_ids = next_id

    return full_seq[0].tolist()


@torch.no_grad()
def greedy_generate_target_ids_batched(model, prompt_ids_batch, max_new_tokens, device):
    return shared_greedy_generate_target_ids_batched(
        model,
        prompt_ids_batch,
        max_new_tokens,
        device,
    )


@torch.no_grad()
def teacher_forced_exact_batch(model, prompt_ids_batch, cot_ids_batch, device):
    return shared_teacher_forced_exact_batch(
        model,
        prompt_ids_batch,
        cot_ids_batch,
        device,
    )


def _sample_eval_batch_from_rng(rng, batch_n, m):
    prompt_batches = []
    cot_batches = []
    for _ in range(batch_n):
        prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, m=m)
        prompt_batches.append(prompt_ids)
        cot_batches.append(cot_ids)
    prompt_ids_batch = torch.tensor(prompt_batches, dtype=torch.long)
    cot_ids_batch = torch.tensor(cot_batches, dtype=torch.long)
    return prompt_ids_batch, cot_ids_batch

@torch.no_grad()
def evaluate_clean_s5_metrics(
    model,
    device,
    n_eval=256,
    m=21,
    seed=123,
    batch_size=256,
):
    """
    Returns three clean-task metrics:
      - cot_exact: teacher-forced full-CoT exact
      - clean_full_exact: autoregressive full-CoT exact
      - clean_final_exact: autoregressive final-answer exact
    """
    rng = random.Random(seed)

    tf_full_ok = 0
    ar_full_ok = 0
    ar_final_ok = 0

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        prompt_ids_batch, cot_ids_batch = _sample_eval_batch_from_rng(rng, end - start, m)

        tf_ok = teacher_forced_exact_batch(model, prompt_ids_batch, cot_ids_batch, device)
        tf_full_ok += int(tf_ok.sum().item())

        pred_ids_batch = greedy_generate_target_ids_batched(
            model,
            prompt_ids_batch,
            cot_ids_batch.size(1),
            device,
        )
        ar_full_ok += int(pred_ids_batch.eq(cot_ids_batch).all(dim=1).sum().item())
        ar_final_ok += int(pred_ids_batch[:, -7:].eq(cot_ids_batch[:, -7:]).all(dim=1).sum().item())

    return {
        "cot_exact": tf_full_ok / n_eval,
        "clean_full_exact": ar_full_ok / n_eval,
        "clean_final_exact": ar_final_ok / n_eval,
    }

@torch.no_grad()
def evaluate_saved_clean_s5_metrics(
    model,
    device,
    data_dir,
    n_eval=None,
    batch_size=256,
):
    return evaluate_saved_clean_metrics(
        model,
        device=device,
        data_dir=data_dir,
        final_answer_len=7,
        n_eval=n_eval,
        batch_size=batch_size,
    )


@torch.no_grad()
def estimate_saved_clean_train_loss(
    model,
    device,
    data_dir,
    eval_iters,
    batch_size,
    subset_size=None,
):
    return shared_estimate_saved_clean_train_loss(
        model,
        device=device,
        data_dir=data_dir,
        eval_iters=eval_iters,
        batch_size=batch_size,
        subset_size=subset_size,
    )
