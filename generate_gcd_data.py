"""
Generate fixed GCD train/eval datasets for grokking study.

Mirrors the exact encoding used by train.py with:
  --operation gcd --maxint 113 --minint 1 --base 1000

Supports two encodings (--encoding flag):
  positional (default): PositionalInts(base=1000)
    Input: V2 + {a} + {b}    Output: + {gcd(a,b)}
  symbolic: SymbolicInts(min=1, max=113)
    Input: V2 {a} {b}         Output: {gcd(a,b)}
"""

import argparse
import math
import os
import random

MAXINT = 113
MININT = 1
TRAIN_FRAC = 0.3
EVAL_FRAC = 0.3
SEED = 22  # same seed as interpret_grok.ipynb

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", type=str, default="positional",
                    choices=["positional", "symbolic"],
                    help="Encoding scheme: positional (default) or symbolic")
args = parser.parse_args()

# Generate all pairs and encode
lines = []
for a in range(MININT, MAXINT + 1):
    for b in range(MININT, MAXINT + 1):
        g = math.gcd(a, b)
        if args.encoding == "symbolic":
            line = f"V2 {a} {b}\t{g}"
        else:
            line = f"V2 + {a} + {b}\t+ {g}"
        lines.append(line)

print(f"Total pairs: {len(lines)}")

# Shuffle and split
random.seed(SEED)
random.shuffle(lines)

n = len(lines)
train_size = int(TRAIN_FRAC * n)
eval_size = int(EVAL_FRAC * n)

train_lines = lines[:train_size]
eval_lines = lines[train_size:train_size + eval_size]

print(f"Train: {len(train_lines)}, Eval: {len(eval_lines)}, Holdout: {n - train_size - eval_size}")

# Save
os.makedirs("data", exist_ok=True)

if args.encoding == "symbolic":
    train_path = "data/gcd_sym_train.txt"
    eval_path = "data/gcd_sym_eval.txt"
else:
    train_path = "data/gcd_train.txt"
    eval_path = "data/gcd_eval.txt"

with open(train_path, "w") as f:
    for line in train_lines:
        f.write(line + "\n")

with open(eval_path, "w") as f:
    for line in eval_lines:
        f.write(line + "\n")

print(f"Saved {train_path} and {eval_path}")
