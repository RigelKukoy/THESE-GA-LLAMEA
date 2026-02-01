import numpy as np

class AdaptiveDEOrthogonalRestart:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, orthogonal_trials=5, restart_trigger=50, restart_factor=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.orthogonal_trials = orthogonal_trials
        self.restart_trigger = restart_trigger
        self.restart_factor = restart_factor
        self.best_fitness_history = []

    def _mutation(self, population, best_idx, i):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        return population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])

    def _crossover(self, mutant, individual):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, mutant, individual)
        return trial_vector

    def _orthogonal_learning(self, population, func, bounds):
        # Orthogonal experimental design for creating trial vectors
        levels = 3  # Number of levels for each dimension
        L = np.zeros((levels, self.dim))

        # Sample points using Latin hypercube sampling for diversification
        for j in range(self.dim):
            L[:, j] = np.linspace(bounds.lb[j], bounds.ub[j], levels)

        trial_vectors = []
        fitness_values = []

        for _ in range(self.orthogonal_trials):
            # Create a random orthogonal array
            oa = np.random.randint(0, levels, size=self.dim)
            trial_vector = np.array([L[oa[i], i] for i in range(self.dim)])
            trial_vectors.append(trial_vector)
            fitness_values.append(func(trial_vector))
        
        best_idx = np.argmin(fitness_values)
        return trial_vectors[best_idx], fitness_values[best_idx]

    def _restart(self, population, best_x, func):
        # Restart the population around the current best solution with reduced bounds.
        new_population = np.zeros_like(population)
        for i in range(self.pop_size):
            new_x = best_x + np.random.uniform(-self.restart_factor, self.restart_factor, size=self.dim) * (func.bounds.ub - func.bounds.lb)
            new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
            new_population[i] = new_x
        return new_population

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        used_budget = self.pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        self.best_fitness_history.append(self.f_opt)
        
        iteration = 0

        while used_budget < self.budget:
            iteration += 1
            for i in range(self.pop_size):
                # Mutation
                mutant = self._mutation(population, best_idx, i)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial_vector = self._crossover(mutant, population[i])

                # Orthogonal learning to improve the trial vector
                orthogonal_trial, orthogonal_fitness = self._orthogonal_learning(population, func, func.bounds)
                used_budget += self.orthogonal_trials
                if orthogonal_fitness < func(trial_vector): # Only check, func is already called in orth_learning
                   trial_vector = orthogonal_trial
                   f = orthogonal_fitness
                else: 
                  f = func(trial_vector)
                  used_budget += 1

                # Selection
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial_vector

                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial_vector
            
            best_idx = np.argmin(fitness)
            
            self.best_fitness_history.append(self.f_opt)

            # Restart mechanism
            if iteration > self.restart_trigger and np.std(self.best_fitness_history[-self.restart_trigger:]) < 1e-6:
                population = self._restart(population, self.x_opt, func)
                fitness = np.array([func(x) for x in population])
                used_budget += self.pop_size
                best_idx = np.argmin(fitness)
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]
                self.best_fitness_history.append(self.f_opt)
                iteration = 0  # Reset iteration counter

        return self.f_opt, self.x_opt