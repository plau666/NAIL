import json
import os
from datasets import load_dataset


dataset = load_dataset("TinyGSM/TinyGSM", split="train")

out_path = "data/tinygsm/train_short.jsonl"
os.makedirs("data/tinygsm", exist_ok=True)
kept = 0
total = 0
with  open(out_path, "w") as fout:
    for entry in dataset:
        total += 1
        if len(entry["code"]) <= 1024:
            fout.write(json.dumps({"question": entry["question"], "code": entry["code"]}) + "\n")
            kept += 1

print(f"Kept {kept}/{total} entries (code length <= 1024) -> {out_path}")
