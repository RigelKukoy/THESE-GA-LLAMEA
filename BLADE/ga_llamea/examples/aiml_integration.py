"""Example: Integrating GA-LLAMEA with AIML API.

This example demonstrates how to use GA-LLAMEA with a custom OpenAI-compatible 
endpoint (AIML API) using BLADE's LLM class.

Usage:
    cd BLADE
    uv run python ga_llamea/examples/aiml_integration.py
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

# Import from GA-LLAMEA
from ga_llamea import GA_LLaMEA

print("=" * 70)
print("GA-LLAMEA AIML API Integration Example")
print("=" * 70)

# Initialize LLM with AIML endpoint
# Using the provided API key: 7baca5864cfb4bc4a6553e68e69b8f6a
print("\n🔧 Initializing LLM with AIML endpoint...")
llm = LLM(
    api_key="7baca5864cfb4bc4a6553e68e69b8f6a",
    model="gpt-4o-mini",
    base_url="https://api.aimlapi.com/v1"
)
print("   ✓ LLM initialized with AIML API")

# Create GA-LLAMEA method
print("\n🔧 Creating GA-LLAMEA method...")
method = GA_LLaMEA(
    llm=llm,
    budget=10,  # Small budget for demo
    solution_class=Solution,
    name="GA-LLAMEA-AIML-Demo",
    n_parents=2,
    n_offspring=4,
)
print("   ✓ Method created")

# Create problem
print("\n🔧 Creating BBOB problem...")
problem = MA_BBOB(
    function_id=1,  # Sphere function
    dimension=2,
    instance=1,
)
print("   ✓ Problem created: f1 (Sphere), 2D")

# Create and run experiment
print("\n" + "=" * 70)
print("Running Experiment (Simplified)...")
print("=" * 70)

# We can call the method directly on the problem for a quick test
try:
    best_sol = method(problem)
    print(f"\n✅ Experiment successful!")
    print(f"Best fitness: {best_sol.fitness:.6f}")
    print(f"Algorithm Name: {best_sol.name}")
except Exception as e:
    print(f"\n❌ Experiment failed: {e}")
    print("\nNote: Ensure your AIML API key has sufficient quota and the endpoint is reachable.")

print("\n" + "=" * 70)
print("GA-LLAMEA works seamlessly with custom endpoints via base_url!")
print("=" * 70)
