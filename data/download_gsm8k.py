"""Download GSM8K train and test splits.

GSM8K (Grade School Math 8K) is a dataset of 8.5K grade school math word problems.
Each example has a question and a step-by-step solution ending with #### <answer>.

Source: https://huggingface.co/datasets/openai/gsm8k

Usage:
    python download_gsm8k.py [--output_dir data/gsm8k]
"""

import argparse
import json
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/gsm8k")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Downloading GSM8K from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")

    for split in ["train", "test"]:
        ds = dataset[split]
        output_path = os.path.join(args.output_dir, f"{split}.jsonl")

        with open(output_path, "w") as f:
            for example in ds:
                # Each example has 'question' and 'answer' fields
                # The answer field contains step-by-step solution ending with #### <final_answer>
                f.write(json.dumps(example) + "\n")

        print(f"  {split}: {len(ds)} examples -> {output_path}")

    # Print a few examples
    print("\n--- Example ---")
    ex = dataset["train"][0]
    print(f"Question: {ex['question'][:200]}...")
    print(f"Answer: {ex['answer'][:200]}...")

    # Extract just the final numeric answer
    final_answer = ex["answer"].split("####")[-1].strip()
    print(f"Final answer: {final_answer}")

    print(f"\nDone! Data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
