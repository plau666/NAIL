import argparse
import json
import multiprocessing as mp
import signal
from tqdm import tqdm


def _timeout_handler(signum, frame):
    raise TimeoutError()


def run_one(entry):
    code = entry["code"]
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(2)
    try:
        ns = {}
        exec(code, ns)
        fn = None
        for name, val in ns.items():
            if callable(val) and name not in ("__builtins__",):
                fn = val
                break
        if fn is None:
            return {**entry, "answer": None, "error": "no_function"}
        result = fn()
        signal.alarm(0)
        json.dumps(result)  # ensure serializable
        return {**entry, "answer": result}
    except TimeoutError:
        return {**entry, "answer": None, "error": "timeout"}
    except Exception as e:
        return {**entry, "answer": None, "error": f"{type(e).__name__}: {e}"[:200]}
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train_short.jsonl")
    ap.add_argument("--output", default="train_short_with_answer.jsonl")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    def gen():
        with open(args.input) as f:
            for i, line in enumerate(f):
                if args.limit is not None and i >= args.limit:
                    break
                yield json.loads(line)

    total = args.limit
    if total is None:
        with open(args.input) as f:
            total = sum(1 for _ in f)

    print(f"Executing {total} snippets with {args.workers} workers...")

    ok, err = 0, 0
    with mp.Pool(args.workers) as pool, open(args.output, "w") as fout:
        for res in tqdm(pool.imap_unordered(run_one, gen(), chunksize=256), total=total):
            fout.write(json.dumps(res) + "\n")
            if res.get("error"):
                err += 1
            else:
                ok += 1

    print(f"Done. ok={ok} err={err} -> {args.output}")


if __name__ == "__main__":
    main()
