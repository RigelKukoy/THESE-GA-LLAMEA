"""Test script to evaluate a generated algorithm exactly like BLADE does."""
import os
import ioh
import numpy as np
import pandas as pd
from ioh import logger as ioh_logger
from iohblade.utils import aoc_logger, correct_aoc, OverBudgetException

print("=" * 60)
print("Testing generated algorithm evaluation")
print("=" * 60)

# The generated DifferentialEvolution code from the log
code = '''
import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, mutation_factor=0.8, crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.bounds = np.array([-5.0, 5.0])  # Lower and upper bounds

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def mutate(self, population, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return np.clip(mutant, self.bounds[0], self.bounds[1])  # Ensure within bounds

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]
'''

# Execute the code
safe_globals = {"np": np}
local_env = {}
exec(code, safe_globals, local_env)
algorithm_name = "DifferentialEvolution"

# Load data files
base_path = os.path.join(os.path.dirname(__file__), "iohblade", "problems", "mabbob")
weights = pd.read_csv(os.path.join(base_path, "weights.csv"), index_col=0)
iids = pd.read_csv(os.path.join(base_path, "iids.csv"), index_col=0)
opt_locs = pd.read_csv(os.path.join(base_path, "opt_locs.csv"), index_col=0)

# Test parameters (matching experiment config)
dim = 5
budget_factor = 1000  # From experiment
budget = budget_factor * dim  # = 5000

print(f"Budget: {budget} (budget_factor={budget_factor} * dim={dim})")
print(f"Algorithm population_size: 20")
print(f"Algorithm budget loop iterations: {budget} (outer) * 20 (inner) = {budget * 20} func calls!")
print()
print("!!! PROBLEM IDENTIFIED !!!")
print("The algorithm has nested loops: budget iterations * population_size evaluations")
print(f"Expected: {budget} function evaluations")
print(f"Actual: {budget * 20 + 20} function evaluations (20 initial + 20*5000 in loop)")
print()

# Now let's verify by running it with just a small budget
test_budget = 100
print(f"Running test with budget={test_budget}:")
print(f"  Expected evaluations: {test_budget * 20 + 20} (outer loop * population)")

# Create problem
idx = 0
f_new = ioh.problem.ManyAffine(
    xopt=np.array(opt_locs.iloc[idx])[:dim],
    weights=np.array(weights.iloc[idx]),
    instances=np.array(iids.iloc[idx], dtype=int),
    n_variables=dim,
)
f_new.set_id(100)
f_new.set_instance(idx)

# Create logger that will track and limit budget
l2 = aoc_logger(test_budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
f_new.attach_logger(l2)

print(f"  Attaching aoc_logger with budget={test_budget}")

try:
    algorithm = local_env[algorithm_name](budget=test_budget, dim=dim)
    print(f"  Algorithm created with budget={test_budget}, dim={dim}")
    print("  Running algorithm...")
    algorithm(f_new)
    print(f"  Algorithm completed normally")
except OverBudgetException:
    print(f"  OverBudgetException raised after {f_new.state.evaluations} evaluations")

print(f"\nFinal evaluations: {f_new.state.evaluations}")
print(f"Logger.aoc: {l2.aoc}")

if f_new.state.evaluations > 0:
    corrected_aoc = correct_aoc(f_new, l2, test_budget)
    print(f"Corrected AOCC: {corrected_aoc}")
