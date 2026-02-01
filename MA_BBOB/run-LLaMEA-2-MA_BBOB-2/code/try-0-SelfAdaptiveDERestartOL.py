import numpy as np

class SelfAdaptiveDERestartOL:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F=0.5, CR=0.7, stagnation_threshold=100, restart_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F
        self.CR = CR
        self.stagnation_threshold = stagnation_threshold
        self.restart_probability = restart_probability
        self.stagnation_counter = 0
        self.f_opt_history = []

    def orthogonal_learning(self, population, fitness, lb, ub, num_samples=5):
        """Applies orthogonal learning to generate diverse solutions."""
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        new_solutions = []
        for _ in range(num_samples):
            basis_vector = np.random.uniform(-1, 1, size=self.dim)
            basis_vector /= np.linalg.norm(basis_vector)  # Normalize

            # Generate a new solution along the orthogonal direction
            step_size = np.random.uniform(0.1 * (ub - lb), 0.5 * (ub - lb)) #random step size
            new_solution = best_solution + step_size * basis_vector
            new_solution = np.clip(new_solution, lb, ub)
            new_solutions.append(new_solution)
        
        return np.array(new_solutions)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.f_opt_history.append(self.f_opt)
        
        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                mutant = self.population[idxs[0]] + self.F * (self.population[idxs[1]] - self.population[idxs[2]])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
            self.f_opt_history.append(self.f_opt)
            
            # Stagnation detection
            if len(self.f_opt_history) > self.stagnation_threshold:
                if np.abs(self.f_opt_history[-1] - self.f_opt_history[-self.stagnation_threshold]) < 1e-6:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

            # Restart mechanism
            if self.stagnation_counter > self.stagnation_threshold:
                if np.random.rand() < self.restart_probability:
                    # Apply orthogonal learning
                    new_solutions = self.orthogonal_learning(self.population, self.fitness, lb, ub)

                    # Replace worst solutions with orthogonal learning solutions
                    worst_indices = np.argsort(self.fitness)[-len(new_solutions):]
                    for j, idx in enumerate(worst_indices):
                        self.population[idx] = new_solutions[j]
                        self.fitness[idx] = func(new_solutions[j])
                        self.eval_count += 1
                        if self.fitness[idx] < self.f_opt:
                            self.f_opt = self.fitness[idx]
                            self.x_opt = self.population[idx]

                    # Reset stagnation counter
                    self.stagnation_counter = 0
                    self.f_opt_history = [self.f_opt] 
                    
            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt