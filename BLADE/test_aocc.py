"""Test script to diagnose AOCC calculation issue."""
import os
import ioh
import numpy as np
import pandas as pd
from ioh import logger as ioh_logger
from iohblade.utils import aoc_logger, correct_aoc

print("=" * 60)
print("Testing ManyAffine and AOCC calculation")
print("=" * 60)

# Load data files
base_path = os.path.join(os.path.dirname(__file__), "iohblade", "problems", "mabbob")
weights = pd.read_csv(os.path.join(base_path, "weights.csv"), index_col=0)
iids = pd.read_csv(os.path.join(base_path, "iids.csv"), index_col=0)
opt_locs = pd.read_csv(os.path.join(base_path, "opt_locs.csv"), index_col=0)

print(f"Loaded data files from: {base_path}")
print(f"  weights shape: {weights.shape}")
print(f"  iids shape: {iids.shape}")
print(f"  opt_locs shape: {opt_locs.shape}")

# Test with instance 0, dim 5
dim = 5
idx = 0
budget = 100

print(f"\nCreating ManyAffine instance {idx}, dim {dim}:")
print(f"  xopt[:dim]: {np.array(opt_locs.iloc[idx])[:dim]}")
print(f"  weights: {np.array(weights.iloc[idx])}")
print(f"  instances: {np.array(iids.iloc[idx], dtype=int)}")

f_new = ioh.problem.ManyAffine(
    xopt=np.array(opt_locs.iloc[idx])[:dim],
    weights=np.array(weights.iloc[idx]),
    instances=np.array(iids.iloc[idx], dtype=int),
    n_variables=dim,
)
f_new.set_id(100)
f_new.set_instance(idx)

print(f"ManyAffine created successfully")
print(f"Bounds: {f_new.bounds}")

# Create logger
l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
f_new.attach_logger(l2)

print(f"\nLogger initialized with budget={budget}, upper=1e2")
print(f"Initial logger.aoc: {l2.aoc}")

# Run some evaluations
print("\nRunning 20 evaluations:")
for i in range(20):
    x = np.random.uniform(-5, 5, dim)
    y = f_new(x)
    print(f"  Eval {i+1}: y={y:.4f}, logger.aoc so far={l2.aoc:.6f}")

print(f"\nAfter 20 evals:")
print(f"  logger.aoc: {l2.aoc}")
print(f"  evaluations: {f_new.state.evaluations}")
print(f"  current_best: {f_new.state.current_best_internal.y if f_new.state.current_best_internal else 'N/A'}")

# Calculate corrected AOCC
final_aoc = correct_aoc(f_new, l2, budget)
print(f"\nCorrected AOCC: {final_aoc}")

# Check if the trigger is working
print("\n" + "=" * 60)
print("Checking if logger trigger is working:")
print(f"Logger upper: {l2.upper}")
print(f"Logger lower: {l2.lower}")
print(f"Budget: {l2.budget}")
