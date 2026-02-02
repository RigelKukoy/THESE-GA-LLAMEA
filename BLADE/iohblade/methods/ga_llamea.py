"""GA-LLAMEA: LLaMEA with Discounted Thompson Sampling for adaptive operator selection.

This implementation is integrated into the iohblade.methods package.
"""

import random
import re
from typing import List, Optional, Tuple, Any, Type

from ..utils import convert_to_serializable
from .ds_ts import DiscountedThompsonSampler

# Define protocols locally to avoid circular imports if needed, 
# or import them if they are available. For now we use Any to be flexible.
# from .interfaces import LLMProtocol, SolutionProtocol, ProblemProtocol

def calculate_reward(parent_score: float, child_score: float, is_valid: bool) -> float:
    """Calculate reward for bandit update.

    Args:
        parent_score: Parent fitness
        child_score: Child fitness
        is_valid: Whether the child is valid (no errors)

    Returns:
        Reward value (0.0 for invalid, max(0, delta_fitness) otherwise)
    """
    if not is_valid:
        return 0.0  # Explicitly punish invalid code
    return max(0.0, child_score - parent_score)


class GA_LLaMEA:
    """GA-LLAMEA: LLaMEA with Discounted Thompson Sampling for operator selection.

    This method uses a multi-armed bandit (D-TS) to adaptively select between three
    genetic operators:
    - **Mutation**: Improve a single parent
    - **Crossover**: Combine two parents
    - **Random New**: Generate a completely new algorithm

    The bandit learns which operator works best over time, adapting to non-stationary
    rewards through exponential discounting.
    """

    def __init__(
        self,
        llm,
        budget: int,
        solution_class: Type[Any] = None,
        name: str = "GA-LLAMEA",
        n_parents: int = 4,
        n_offspring: int = 16,
        elitism: bool = True,
        discount: float = 0.9,
        tau_max: float = 1.0,
        reward_variance: float = 1.0,
        **kwargs,
    ):
        """Initialize GA-LLAMEA.

        Args:
            llm: LLM instance for code generation
            budget: Total number of LLM queries
            solution_class: Solution class to use
            name: Method name
            n_parents: Population size (μ)
            n_offspring: Offspring per generation (λ)
            elitism: Use (μ+λ) selection if True, (μ,λ) otherwise
            discount: D-TS discount factor γ ∈ (0, 1]
            tau_max: Maximum posterior uncertainty
            reward_variance: Expected reward variance
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.budget = budget
        self.name = name
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.elitism = elitism
        self.kwargs = kwargs
        
        # Solution factory
        if solution_class is not None:
            self._solution_class = solution_class
        else:
            # Import strictly here to avoid circular dependency issues at module level
            from ..solution import Solution
            self._solution_class = Solution

        # Initialize D-TS bandit with three operators
        self.bandit = DiscountedThompsonSampler(
            arm_names=["mutation", "crossover", "random_new"],
            discount=discount,
            tau_max=tau_max,
            reward_variance=reward_variance,
        )

        self.population: List[Any] = []
        self.best_solution: Optional[Any] = None
        self.generation = 0

    def _create_solution(self, code: str = "", fitness: float = 0.0) -> Any:
        """Factory method to create a Solution instance."""
        # Check if _solution_class is the iohblade Solution class which takes specific args
        try:
            sol = self._solution_class(code=code)
        except TypeError:
            # Fallback for simpler solution classes
            sol = self._solution_class()
            sol.code = code
            
        sol.fitness = fitness
        if not hasattr(sol, 'metadata') or sol.metadata is None:
            sol.metadata = {}
        if not hasattr(sol, 'error'):
            sol.error = ""
        # Extract class name from code - BLADE's mabbob.py requires this
        if not hasattr(sol, 'name') or not sol.name:
            sol.name = self._extract_classname(code)
        # Ensure ID exists
        if not hasattr(sol, 'id'):
            import uuid
            sol.id = str(uuid.uuid4())
            
        return sol
    
    def _extract_classname(self, code: str) -> str:
        """Extract the Python class name from generated code."""
        try:
            match = re.search(r"class\s*(\w*)(?:\(\w*\))?:", code, re.IGNORECASE)
            if match:
                return match.group(1)
        except Exception:
            pass
        return "UnknownAlgorithm"

    def __call__(self, problem) -> Any:
        """Execute GA-LLAMEA evolution.

        Args:
            problem: Problem instance to optimize

        Returns:
            Best solution found
        """
        # Initialize population
        self._initialize_population(problem)
        self.llm_calls = self.n_parents  # Track LLM calls internally
        
        # Handle empty population edge case
        if not self.population:
            print("   [WARNING] Population initialization failed. Trying simple random search as fallback.")
            # Could implement fallback here, but for now just return None or raise
            if self.best_solution:
                return self.best_solution
            # Try to return at least something if possible, or raise error
            raise RuntimeError("Population initialization failed - no valid solutions were generated.")

        # Calculate number of generations
        remaining_budget = self.budget - self.n_parents
        n_generations = max(1, remaining_budget // self.n_offspring)

        # Evolutionary loop
        for gen in range(n_generations):
            self.generation = gen + 1
            
            current_best_fit = self.best_solution.fitness if self.best_solution else -float('inf')
            print(f"Generation {self.generation}: Best Fitness = {current_best_fit:.4f}")

            offspring = []

            # Generate offspring
            for _ in range(self.n_offspring):
                if self.llm_calls >= self.budget:
                    break

                # Select operator using D-TS
                operator, theta = self.bandit.select_arm()

                # Generate offspring based on selected operator
                try:
                    child = None
                    if operator == "mutation":
                        parent = self._select_parent()
                        child = self._mutation(parent, problem)
                    elif operator == "crossover":
                        parent1, parent2 = self._select_two_parents()
                        child = self._crossover(parent1, parent2, problem)
                    else:  # random_new
                        child = self._random_new(problem)

                    # Evaluate offspring
                    # CRITICAL FIX: Use problem(child) to ensure Sandbox evaluation and consistent logging
                    if child and child.code:
                        # Log operator before evaluation just in case, but usually metadata is preserved
                        child.metadata["operator"] = operator
                        child.metadata["theta_sampled"] = theta
                        child.metadata["generation"] = self.generation

                        # Evaluate using the safe wrapper
                        # This handles timeouts, errors, and logging to file if configured
                        try:
                            child = problem(child) 
                        except Exception as e:
                            # Although problem() catches most, strictly ensure we don't crash loop
                            child.fitness = -float('inf')
                            child.error = str(e)
                        
                        offspring.append(child)
                        self.llm_calls += 1  # Track LLM call

                        # Update bandit with reward
                        # Use best solution's fitness as baseline for ALL operators
                        baseline_fitness = self.best_solution.fitness if self.best_solution else 0.0
                        
                        # Handle -inf fitness (crashes)
                        child_fitness = child.fitness
                        if child_fitness == -float('inf') or child.error:
                            child_fitness = 0.0 # Treat crash as 0.0 for reward calculation
                            is_valid = False
                        else:
                            is_valid = True
                            
                        reward = calculate_reward(
                            baseline_fitness, child_fitness, is_valid
                        )
                        self.bandit.update(operator, reward)
                        
                        # Store reward in metadata
                        child.metadata["reward"] = reward

                        # Note: We rely on problem(child) to log to the experiment logger
                        # The experiment logger is attached to the problem instance

                except Exception as e:
                    print(f"   [ERROR] Offspring generation failed: {e}")
                    # Failed offspring - punish operator
                    self.bandit.update(operator, 0.0)
                    continue

            # Selection
            if self.elitism:
                # (μ+λ) selection: combine parents and offspring
                combined = self.population + offspring
                self.population = self._select(combined, self.n_parents)
            else:
                # (μ,λ) selection: select only from offspring
                self.population = self._select(offspring, self.n_parents)

            # Update best solution
            if self.population:
                # Filter out -inf solutions for best selection if possible
                valid_pop = [s for s in self.population if s.fitness != -float('inf')]
                if valid_pop:
                    current_best = max(valid_pop, key=lambda s: s.fitness)
                else:
                    current_best = self.population[0] # All crashed
                    
                if (
                    self.best_solution is None
                    or current_best.fitness > self.best_solution.fitness
                ):
                    self.best_solution = current_best

        return self.best_solution if self.best_solution else self.population[0]

    def _initialize_population(self, problem) -> None:
        """Initialize population with diverse algorithms."""
        print(f"   [INFO] Initializing population with {self.n_parents} parents...")
        for i in range(self.n_parents):
            if i >= self.budget:
                break
            
            # Generate initial algorithm
            try:
                # Use standard task prompt for initialization
                prompt = self._get_task_prompt(problem)
                child = self._generate_solution(prompt, problem)
                if child and child.code:
                    child.metadata["generation"] = 0
                    child.metadata["operator"] = "init"
                    
                    # CRITICAL FIX: Use problem(child) for safe evaluation
                    try:
                        child = problem(child)
                    except Exception as e:
                        child.fitness = -float('inf')
                        child.error = str(e)
                    
                    self.population.append(child)
            except Exception as e:
                import traceback
                traceback.print_exc()

        if self.population:
            # Find best non-crashed solution
            valid_pop = [s for s in self.population if s.fitness != -float('inf')]
            if valid_pop:
                self.best_solution = max(valid_pop, key=lambda s: s.fitness)
            else:
                self.best_solution = self.population[0]
                
            print(f"   [INFO] Initialization complete. Best Fitness: {self.best_solution.fitness:.4f}")

    def _select_parent(self) -> Any:
        """Select a random parent from population."""
        # Prefer valid parents
        valid_pop = [s for s in self.population if s.fitness != -float('inf')]
        if valid_pop:
            return random.choice(valid_pop)
        return random.choice(self.population)

    def _select_two_parents(self) -> Tuple[Any, Any]:
        """Select two distinct parents for crossover."""
        # Work with valid population if possible (or full if all crashed)
        pool = [s for s in self.population if s.fitness != -float('inf')]
        if not pool:
            pool = self.population
            
        sorted_pop = sorted(pool, key=lambda s: s.fitness, reverse=True)

        # Best parent
        parent1 = sorted_pop[0]

        # Second parent: diverse from first
        if len(sorted_pop) > 1:
            # Prefer different code
            for p in sorted_pop[1:]:
                if p.code != parent1.code:
                    parent2 = p
                    break
            else:
                parent2 = sorted_pop[1]
        else:
            parent2 = parent1

        return parent1, parent2

    def _select(self, population: List[Any], n: int) -> List[Any]:
        """Select best n individuals with diversity preservation."""
        if not population:
            return []

        # Sort by fitness (descending), putting -inf last
        sorted_pop = sorted(population, key=lambda s: s.fitness if s.fitness != -float('inf') else -1e9, reverse=True)

        # Always include best
        selected = [sorted_pop[0]]
        if n == 1:
            return selected

        # Add diverse individuals (different code)
        seen_codes = {sorted_pop[0].code}
        for sol in sorted_pop[1:]:
            if len(selected) >= n:
                break
            if sol.code not in seen_codes:
                selected.append(sol)
                seen_codes.add(sol.code)

        # Fill remaining slots with next-best fitness
        for sol in sorted_pop:
            if len(selected) >= n:
                break
            if sol not in selected:
                selected.append(sol)

        return selected[:n]

    def _mutation(self, parent: Any, problem) -> Optional[Any]:
        """Mutation operator."""
        prompt = self._build_mutation_prompt(parent, problem)
        return self._generate_solution(prompt, problem)

    def _crossover(self, parent1: Any, parent2: Any, problem) -> Optional[Any]:
        """Crossover operator."""
        prompt = self._build_crossover_prompt(parent1, parent2, problem)
        return self._generate_solution(prompt, problem)

    def _random_new(self, problem) -> Optional[Any]:
        """Random new operator."""
        prompt = self._build_random_new_prompt(problem)
        return self._generate_solution(prompt, problem)

    def _get_task_prompt(self, problem) -> str:
        """Get the standardized Task Prompt (S) with Guardrails."""
        example_code = getattr(problem, 'example_prompt', '')
        prompt = f"""You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems.
You are a Python developer working on a new optimization algorithm.
Your task is to develop a novel heuristic optimization algorithm for continuous optimization problems.
The optimization algorithm should handle a wide range of tasks, which is evaluated on the Many Affine BBOB test suite of noiseless functions. Your task is to write the optimization algorithm in Python code. 
Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. 
An example of such code is as follows:

{example_code}

Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```"""
        return prompt

    def _get_population_history(self) -> str:
        """Get the list of previously generated algorithm names with mean AOCC score."""
        if not self.population:
            return ""
        
        # Sort by fitness (descending)
        sorted_pop = sorted(self.population, key=lambda s: s.fitness if s.fitness != -float('inf') else -1e9, reverse=True)
        
        history = "The current population of algorithms already evaluated (name, description, score) is:\n"
        for sol in sorted_pop:
            name = sol.name if sol.name else "Unknown"
            desc = sol.description if hasattr(sol, 'description') and sol.description else "No description"
            fitness_str = f"{sol.fitness}" if sol.fitness != -float('inf') else "-inf"
            history += f"{name}: {desc} (Score: {fitness_str})\n"
        return history

    def _build_mutation_prompt(self, parent: Any, problem) -> str:
        """Build prompt for mutation operator."""
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history()
        
        error_block = ""
        if hasattr(parent, 'error') and parent.error:
            error_block = f"\n### Error Encountered\n{parent.error}\n"

        algo_details = f"""
The selected solution to update is:
{parent.description if hasattr(parent, 'description') and parent.description else parent.name}

With code:
```python
{parent.code}
```

The algorithm {parent.name} scored {parent.fitness} on AOCC (higher is better, 1.0 is the best).
{error_block}
"""
        instruction = "Refine the strategy of the selected algorithm to improve it."

        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"

    def _build_crossover_prompt(self, parent1: Any, parent2: Any, problem) -> str:
        """Build prompt for crossover operator."""
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history()
        
        algo_details = f"""
Selected algorithms to combine:

Algorithm 1 (fitness: {parent1.fitness:.4f}):
```python
{parent1.code}
```

Algorithm 2 (fitness: {parent2.fitness:.4f}):
```python
{parent2.code}
```
"""
        instruction = "Combine the best strategies from both algorithms into a single improved algorithm. Take inspiration from both approaches and create a hybrid that leverages their strengths."

        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"

    def _build_random_new_prompt(self, problem) -> str:
        """Build prompt for random new operator."""
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history()
        instruction = "Generate a completely new and different algorithm for this optimization problem. Create a novel approach that explores a different strategy than existing solutions."

        return f"{task_prompt}\n\n{history}\n\n{instruction}\n\n{problem.format_prompt}"

    def _generate_solution(self, prompt: str, problem) -> Optional[Any]:
        """Generate a solution from LLM response."""
        try:
            session_messages = [{"role": "user", "content": prompt}]
            response = self.llm.query(session_messages)

            code = self._extract_code(response)
            if not code:
                return None

            solution = self._create_solution(code=code, fitness=0.0)
            # Add metadata if solution supports it
            if hasattr(solution, 'metadata'):
                solution.metadata["llm_response"] = response
            
            description = self._extract_description(response)
            if description and hasattr(solution, 'description'):
                solution.description = description

            return solution

        except Exception as e:
            print(f"   [ERROR] _generate_solution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        if "def " in response or "class " in response:
            return response.strip()

        return None

    def _extract_description(self, response: str) -> str:
        """Extract description or name from LLM response."""
        match = re.search(r"#\s*(?:Description|Name):\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def to_dict(self):
        """Returns a dictionary representation of the method."""
        return {
            "method_name": self.name,
            "budget": self.budget,
            "n_parents": self.n_parents,
            "n_offspring": self.n_offspring,
            "elitism": self.elitism,
            "bandit_state": self.bandit.get_state_snapshot() if hasattr(self, "bandit") else {},
            **self.kwargs,
        }
