"""GA-LLAMEA: LLaMEA with Discounted Thompson Sampling for adaptive operator selection.

This is a standalone implementation that works with BLADE via Protocol-based interfaces.
No changes to BLADE code are required - pass BLADE's classes directly to GA_LLaMEA.
"""

import random
import re
from typing import List, Optional, Tuple, Any, Type, Callable

from .ds_ts import DiscountedThompsonSampler
from .interfaces import LLMProtocol, SolutionProtocol, ProblemProtocol


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
    
    Example Usage with BLADE:
        from iohblade.llm import LLM
        from iohblade.solution import Solution
        from iohblade.problems import MA_BBOB
        from ga_llamea import GA_LLaMEA
        
        llm = LLM(model="gemini-2.0-flash")
        method = GA_LLaMEA(
            llm=llm,
            budget=100,
            solution_class=Solution,  # Pass BLADE's Solution class
        )
        problem = MA_BBOB(function_id=1, dimension=5, instance=1)
        best = method(problem)
    """

    def __init__(
        self,
        llm: LLMProtocol,
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
            llm: LLM instance for code generation (BLADE's LLM class works directly)
            budget: Total number of LLM queries
            solution_class: Solution class to use (pass BLADE's Solution class)
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
        
        # Solution factory - allows using BLADE's Solution class or custom implementation
        if solution_class is not None:
            self._solution_class = solution_class
        else:
            # Default minimal Solution implementation
            self._solution_class = _DefaultSolution

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
        sol = self._solution_class(code=code)
        sol.fitness = fitness
        if not hasattr(sol, 'metadata'):
            sol.metadata = {}
        if not hasattr(sol, 'error'):
            sol.error = ""
        # Extract class name from code - BLADE's mabbob.py requires this
        if not hasattr(sol, 'name') or not sol.name:
            sol.name = self._extract_classname(code)
        return sol
    
    def _extract_classname(self, code: str) -> str:
        """Extract the Python class name from generated code.
        
        Args:
            code: Python code string
            
        Returns:
            Class name or empty string if not found
        """
        try:
            match = re.search(r"class\s*(\w*)(?:\(\w*\))?:", code, re.IGNORECASE)
            if match:
                return match.group(1)
        except Exception:
            pass
        return ""

    def __call__(self, problem: ProblemProtocol) -> Any:
        """Execute GA-LLAMEA evolution.

        Args:
            problem: Problem instance to optimize (BLADE's Problem works directly)

        Returns:
            Best solution found
        """
        # Initialize population
        self._initialize_population(problem)
        self.llm_calls = self.n_parents  # Track LLM calls internally
        
        # Handle empty population edge case
        if not self.population:
            raise RuntimeError(
                "Population initialization failed - no valid solutions were generated. "
                "Check that the LLM is responding correctly and generating valid code."
            )

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
                    if operator == "mutation":
                        parent = self._select_parent()
                        child = self._mutation(parent, problem)
                    elif operator == "crossover":
                        parent1, parent2 = self._select_two_parents()
                        child = self._crossover(parent1, parent2, problem)
                    else:  # random_new
                        child = self._random_new(problem)

                    # Evaluate offspring
                    if child and child.code:
                        problem.evaluate(child)
                        offspring.append(child)
                        self.llm_calls += 1  # Track LLM call

                        # Update bandit with reward
                        # Use best solution's fitness as baseline for ALL operators
                        # This ensures fair comparison across mutation, crossover, and random_new
                        baseline_fitness = self.best_solution.fitness if self.best_solution else 0.0
                        is_valid = not bool(child.error)  # Valid if no error
                        reward = calculate_reward(
                            baseline_fitness, child.fitness, is_valid
                        )
                        self.bandit.update(operator, reward)

                        # Log operator selection
                        child.metadata["operator"] = operator
                        child.metadata["theta_sampled"] = theta
                        child.metadata["reward"] = reward
                        child.metadata["generation"] = self.generation

                        # Log to experiment logger if available
                        if hasattr(self.llm, "logger") and hasattr(self.llm.logger, "log_individual"):
                            self.llm.logger.log_individual(child)

                except Exception as e:
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
                current_best = max(self.population, key=lambda s: s.fitness)
                if (
                    self.best_solution is None
                    or current_best.fitness > self.best_solution.fitness
                ):
                    self.best_solution = current_best

        return self.best_solution if self.best_solution else self.population[0]

    def _initialize_population(self, problem: ProblemProtocol) -> None:
        """Initialize population with diverse algorithms."""
        # Initialize population with diverse algorithms
        # print(f"   [DEBUG] Initializing population with {self.n_parents} parents...")
        for i in range(self.n_parents):
            if i >= self.budget:
                break
            
            # Generate initial algorithm
            try:
                # Use standard task prompt for initialization
                prompt = self._get_task_prompt(problem)
                child = self._generate_solution(prompt, problem)
                if child and child.code:
                    problem.evaluate(child)
                    child.metadata["generation"] = 0
                    child.metadata["operator"] = "init"
                    
                    # Log to experiment logger if available
                    if hasattr(self.llm, "logger") and hasattr(self.llm.logger, "log_individual"):
                        self.llm.logger.log_individual(child)

                    self.population.append(child)
            except Exception as e:
                import traceback
                traceback.print_exc()

        if self.population:
            self.best_solution = max(self.population, key=lambda s: s.fitness)
            print(f"   [INFO] Initialization complete. Best Fitness: {self.best_solution.fitness:.4f}")

    def _select_parent(self) -> Any:
        """Select a random parent from population."""
        return random.choice(self.population)

    def _select_two_parents(self) -> Tuple[Any, Any]:
        """Select two distinct parents for crossover.

        Prefers high-fitness parents but allows diversity.
        """
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)

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
        """Select best n individuals with diversity preservation.

        Args:
            population: Pool of solutions
            n: Number to select

        Returns:
            Selected solutions
        """
        if not population:
            return []

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)

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

    def _mutation(self, parent: Any, problem: ProblemProtocol) -> Optional[Any]:
        """Mutation operator: Improve a single parent.

        Args:
            parent: Parent solution to improve
            problem: Problem instance

        Returns:
            Child solution or None if generation failed
        """
        prompt = self._build_mutation_prompt(parent, problem)
        return self._generate_solution(prompt, problem)

    def _crossover(
        self, parent1: Any, parent2: Any, problem: ProblemProtocol
    ) -> Optional[Any]:
        """Crossover operator: Combine two parents.

        Args:
            parent1: First parent (usually best)
            parent2: Second parent (diverse)
            problem: Problem instance

        Returns:
            Child solution or None if generation failed
        """
        prompt = self._build_crossover_prompt(parent1, parent2, problem)
        return self._generate_solution(prompt, problem)

    def _random_new(self, problem: ProblemProtocol) -> Optional[Any]:
        """Random new operator: Generate a completely new algorithm.

        Args:
            problem: Problem instance

        Returns:
            Child solution or None if generation failed
        """
        prompt = self._build_random_new_prompt(problem)
        return self._generate_solution(prompt, problem)

    def _get_task_prompt(self, problem: ProblemProtocol) -> str:
        """Get the standardized Task Prompt (S)."""
        example_code = getattr(problem, 'example_prompt', '')
        prompt = f"""You are an excellent Python programmer.
You are a Python developer working on a new optimization algorithm.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems. The optimization algorithm should handle a wide range of tasks, which is evaluated on a large test suite of noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain one function def __call__(self, f), which should optimize the black box function f using budget function evaluations. The f() can only be called as many times as the budget allows. An example of such code is as follows:

{example_code}

Give a novel heuristic algorithm to solve this task. Give the response in the format:
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
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        
        history = "List of previously generated algorithm names with mean AOCC score:\n"
        for sol in sorted_pop:
            name = sol.name if sol.name else "Unknown"
            fitness = sol.fitness if sol.fitness is not None else 0.0
            history += f"- {name}: {fitness:.4f}\n"
        return history

    def _build_mutation_prompt(self, parent: Any, problem: ProblemProtocol) -> str:
        """Build prompt for mutation operator."""
        # 1. Task Prompt (S)
        task_prompt = self._get_task_prompt(problem)
        
        # 2. History
        history = self._get_population_history()
        
        # 3. Selected algorithm to refine
        algo_details = f"""
Selected algorithm to refine:
Name: {parent.name}
Fitness: {parent.fitness:.4f}
Code:
```python
{parent.code}
```
"""
        # 4. Feedback/Instruction
        instruction = "Either refine or redesign to improve the algorithm."

        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"

    def _build_crossover_prompt(
        self, parent1: Any, parent2: Any, problem: ProblemProtocol
    ) -> str:
        """Build prompt for crossover operator."""
        # 1. Task Prompt (S)
        task_prompt = self._get_task_prompt(problem)
        
        # 2. History
        history = self._get_population_history()
        
        # 3. Selected algorithms to combine
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
        # 4. Feedback/Instruction
        instruction = "Combine the best strategies from both algorithms into a single improved algorithm.\nTake inspiration from both approaches and create a hybrid that leverages their strengths."

        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"

    def _build_random_new_prompt(self, problem: ProblemProtocol) -> str:
        """Build prompt for random new operator."""
        # 1. Task Prompt (S)
        task_prompt = self._get_task_prompt(problem)
        
        # 2. History
        history = self._get_population_history()
        
        # 4. Feedback/Instruction
        instruction = "Generate a completely new and different algorithm for this optimization problem.\nCreate a novel approach that explores a different strategy than existing solutions."

        return f"{task_prompt}\n\n{history}\n\n{instruction}\n\n{problem.format_prompt}"

    def _generate_solution(self, prompt: str, problem: ProblemProtocol) -> Optional[Any]:
        """Generate a solution from LLM response.

        Args:
            prompt: Prompt to send to LLM
            problem: Problem instance

        Returns:
            Solution object or None if parsing failed
        """
        try:
            # Query LLM - format prompt as session messages (list of dicts)
            # BLADE's LLM expects: [{"role": "user", "content": "..."}]
            session_messages = [{"role": "user", "content": prompt}]
            response = self.llm.query(session_messages)

            # Extract code
            code = self._extract_code(response)
            if not code:
                print(f"   [DEBUG] Failed to extract code from response")
                if response:
                    print(f"   [DEBUG] Response preview: {response[:200]}...")
                return None

            # Create solution using the configured solution class
            solution = self._create_solution(code=code, fitness=0.0)
            solution.metadata["llm_response"] = response
            
            # Extract description
            description = self._extract_description(response)
            if description:
                solution.description = description

            return solution

        except Exception as e:
            print(f"   [DEBUG] _generate_solution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted code or None if not found
        """
        # Try to find code in markdown code blocks
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        # Fallback: return entire response if it looks like code
        if "def " in response or "class " in response:
            return response.strip()

        return None

    def _extract_description(self, response: str) -> str:
        """Extract description or name from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted description or empty string
        """
        # Look for # Description: or # Name:
        match = re.search(r"#\s*(?:Description|Name):\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def to_dict(self):
        """Returns a dictionary representation of the method.

        Returns:
            dict: Dictionary representation
        """
        return {
            "method_name": self.name,
            "budget": self.budget,
            "n_parents": self.n_parents,
            "n_offspring": self.n_offspring,
            "elitism": self.elitism,
            "bandit_state": self.bandit.get_state_snapshot() if hasattr(self, "bandit") else {},
            **self.kwargs,
        }


class _DefaultSolution:
    """Default minimal Solution implementation for standalone use."""
    
    def __init__(self, code: str = "", **kwargs):
        self.code = code
        self.fitness = 0.0
        self.error = ""
        self.name = ""
        self.metadata = {}
        for k, v in kwargs.items():
            setattr(self, k, v)
