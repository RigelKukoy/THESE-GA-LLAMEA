"""Example: Integrating GA-LLAMEA with BLADE.

This example demonstrates how to use GA-LLAMEA from the standalone ga_llamea
package with BLADE's optimization framework.

NO CHANGES TO BLADE CODE ARE REQUIRED!

Usage:
    cd BLADE-Original
    uv run python ga_llamea/examples/integrate_with_blade.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import ga_llamea
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from BLADE
from iohblade.llm import LLM
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment

# Import from GA-LLAMEA (standalone package)
from ga_llamea import GA_LLaMEA

print("=" * 70)
print("GA-LLAMEA Integration Example")
print("=" * 70)
print("\nThis example shows how to use the standalone ga_llamea package")
print("with BLADE - no changes to BLADE code required!")
print("=" * 70)

# Initialize LLM
print("\n🔧 Initializing LLM...")
llm = LLM(
    model="gemini-2.0-flash",  # or "gpt-4", "claude-3-5-sonnet-20241022", etc.
)
print("   ✓ LLM initialized")

# Create GA-LLAMEA method from standalone package
print("\n🔧 Creating GA-LLAMEA method from ga_llamea package...")
method = GA_LLaMEA(
    llm=llm,
    budget=50,  # Total LLM queries (reduced for quick demo)
    solution_class=Solution,  # Pass BLADE's Solution class!
    name="GA-LLAMEA-Integration-Demo",
    n_parents=4,
    n_offspring=16,
    elitism=True,
    discount=0.9,
    tau_max=1.0,
)
print("   ✓ Method created")

# Create problem
print("\n🔧 Creating BBOB problem...")
problem = MA_BBOB(
    function_id=1,  # Sphere function
    dimension=5,
    instance=1,
)
print("   ✓ Problem created: f1 (Sphere), 5D")

# Create and run experiment
print("\n" + "=" * 70)
print("Running Experiment...")
print("=" * 70)

experiment = Experiment(
    problem=problem,
    method=method,
    experiment_name="GA-LLAMEA-Integration-Demo",
    num_workers=1,
)

# Run experiment
results = experiment.run()

# Print Results
print("\n" + "=" * 70)
print("✅ Results")
print("=" * 70)
print(f"Best fitness: {results['best_fitness']:.6f}")
print(f"Best solution: {results['best_solution'].name}")

# Print bandit statistics
print("\n" + "=" * 70)
print("📊 Bandit Statistics (Operator Selection)")
print("=" * 70)

bandit_state = method.bandit.get_state_snapshot()
print(f"\n{'Operator':<15} {'Pulls':>6} {'Mean Reward':>12} {'Std Dev':>10}")
print("-" * 45)
for operator, stats in bandit_state.items():
    print(
        f"{operator:<15} {stats['pulls']:>6d} {stats['mean']:>12.4f} "
        f"{stats['std']:>10.4f}"
    )

# Calculate selection percentages
total_pulls = sum(s["pulls"] for s in bandit_state.values())
print("\n" + "=" * 70)
print("📈 Operator Selection Percentages")
print("=" * 70)
for operator, stats in bandit_state.items():
    pct = (stats["pulls"] / total_pulls * 100) if total_pulls > 0 else 0
    print(f"{operator:<15}: {pct:5.1f}% ({stats['pulls']} / {total_pulls})")

print("\n" + "=" * 70)
print("✅ Integration successful! GA-LLAMEA works with BLADE without any changes.")
print("=" * 70)
