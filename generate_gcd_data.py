"""
Generate fixed GCD train/eval datasets for grokking study.

Mirrors the exact encoding used by train.py with:
  --operation gcd --maxint 113 --minint 1 --base 1000

Input encoder:  NumberArray(max_dim=2, dim_prefix='V', tensor_dim=1)
                with subencoder PositionalInts(base=1000)
Output encoder: PositionalInts(base=1000)

File format (tab-separated): V2 + {a} + {b}\t+ {gcd(a,b)}
"""

import math
import os
import random

MAXINT = 113
MININT = 1
TRAIN_FRAC = 0.3
EVAL_FRAC = 0.3
SEED = 22  # same seed as interpret_grok.ipynb

# Generate all pairs and encode
lines = []
for a in range(MININT, MAXINT + 1):
    for b in range(MININT, MAXINT + 1):
        g = math.gcd(a, b)
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

with open("data/gcd_train.txt", "w") as f:
    for line in train_lines:
        f.write(line + "\n")

with open("data/gcd_eval.txt", "w") as f:
    for line in eval_lines:
        f.write(line + "\n")

print("Saved data/gcd_train.txt and data/gcd_eval.txt")
