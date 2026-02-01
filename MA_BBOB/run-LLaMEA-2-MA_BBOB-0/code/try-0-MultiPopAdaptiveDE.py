import numpy as np

class MultiPopAdaptiveDE:
    def __init__(self, budget=10000, dim=10, num_pops=3, pop_size=30, migration_interval=50, migration_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_pops = num_pops
        self.pop_size = pop_size
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.populations = [np.random.uniform(-5, 5, size=(pop_size, dim)) for _ in range(num_pops)]
        self.fitness = [np.zeros(pop_size) for _ in range(num_pops)]
        self.F = [0.5] * num_pops
        self.CR = [0.9] * num_pops
        self.best_fitness = [np.inf] * num_pops
        self.best_solutions = [None] * num_pops
        self.overall_best_f = np.inf
        self.overall_best_x = None
        self.eval_counts = [0] * num_pops

    def evaluate_population(self, func, pop_idx):
        for i in range(self.pop_size):
            if self.eval_counts[pop_idx] < self.budget:
                self.fitness[pop_idx][i] = func(self.populations[pop_idx][i])
                self.eval_counts[pop_idx] += 1
                if self.fitness[pop_idx][i] < self.best_fitness[pop_idx]:
                    self.best_fitness[pop_idx] = self.fitness[pop_idx][i]
                    self.best_solutions[pop_idx] = self.populations[pop_idx][i].copy()
                    if self.fitness[pop_idx][i] < self.overall_best_f:
                        self.overall_best_f = self.fitness[pop_idx][i]
                        self.overall_best_x = self.populations[pop_idx][i].copy()
            else:
                return False
        return True

    def evolve_population(self, func, pop_idx):
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.populations[pop_idx][idxs]
            mutant = x_r1 + self.F[pop_idx] * (x_r2 - x_r3)
            mutant = np.clip(mutant, -5, 5)
            
            crossover = np.random.uniform(size=self.dim) < self.CR[pop_idx]
            trial = np.where(crossover, mutant, self.populations[pop_idx][i])

            if self.eval_counts[pop_idx] < self.budget:
                f_trial = func(trial)
                self.eval_counts[pop_idx] += 1

                if f_trial < self.overall_best_f:
                    self.overall_best_f = f_trial
                    self.overall_best_x = trial.copy()
                    
                if f_trial < self.fitness[pop_idx][i]:
                    self.fitness[pop_idx][i] = f_trial
                    self.populations[pop_idx][i] = trial.copy()
                    if f_trial < self.best_fitness[pop_idx]:
                        self.best_fitness[pop_idx] = f_trial
                        self.best_solutions[pop_idx] = trial.copy()
            else:
                return False
        return True

    def migrate(self):
        # Select two random populations
        pop_indices = np.random.choice(self.num_pops, 2, replace=False)
        pop_idx1, pop_idx2 = pop_indices[0], pop_indices[1]

        # Select individuals to migrate
        num_migrants = int(self.migration_rate * self.pop_size)
        
        # Replace worst individuals in pop_idx2 with best from pop_idx1
        indices_to_replace = np.argsort(self.fitness[pop_idx2])[-num_migrants:]
        sorted_indices_pop1 = np.argsort(self.fitness[pop_idx1])[:num_migrants]

        for i in range(num_migrants):
            self.populations[pop_idx2][indices_to_replace[i]] = self.populations[pop_idx1][sorted_indices_pop1[i]].copy()
            self.fitness[pop_idx2][indices_to_replace[i]] = self.fitness[pop_idx1][sorted_indices_pop1[i]]
        
        # Update best fitness and solutions after migration
        if np.min(self.fitness[pop_idx2]) < self.best_fitness[pop_idx2]:
            best_idx = np.argmin(self.fitness[pop_idx2])
            self.best_fitness[pop_idx2] = self.fitness[pop_idx2][best_idx]
            self.best_solutions[pop_idx2] = self.populations[pop_idx2][best_idx].copy()


    def adapt_parameters(self):
        for i in range(self.num_pops):
            # Simplified adaptation: adjust F and CR based on fitness improvement
            improvement = self.best_fitness[i] - np.min(self.fitness[i])
            if improvement < 0:
                self.F[i] *= 0.9
                self.CR[i] *= 1.1
            else:
                self.F[i] *= 1.1
                self.CR[i] *= 0.9
            self.F[i] = np.clip(self.F[i], 0.1, 1.0)
            self.CR[i] = np.clip(self.CR[i], 0.1, 1.0)

    def __call__(self, func):
        # Initialize and evaluate populations
        for i in range(self.num_pops):
            if not self.evaluate_population(func, i):
                return self.overall_best_f, self.overall_best_x

        generation = 0
        while min(self.eval_counts) < self.budget:
            for i in range(self.num_pops):
                if not self.evolve_population(func, i):
                    return self.overall_best_f, self.overall_best_x

            if generation % self.migration_interval == 0:
                self.migrate()

            self.adapt_parameters()
            generation += 1

        return self.overall_best_f, self.overall_best_x