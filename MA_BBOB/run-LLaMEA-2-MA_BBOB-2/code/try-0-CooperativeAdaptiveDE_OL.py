import numpy as np

class CooperativeAdaptiveDE_OL(object):
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.archive = []
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        used_budget = self.pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Main loop
        while used_budget < self.budget:
            # Cooperative DE: Each individual interacts with the best and a random individual
            mutants = []
            for i in range(self.pop_size):
                best_individual = population[np.argmin(fitness)]
                random_individual = population[np.random.randint(self.pop_size)]
                idxs = np.random.choice(self.pop_size, 2, replace=False)

                # DE/current-to-best/1
                mutant = population[i] + self.F * (best_individual - population[i]) + self.F * (population[idxs[0]] - population[idxs[1]])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                mutants.append(mutant)
            mutants = np.array(mutants)

            # Crossover
            crossover_mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial_vectors = np.where(crossover_mask, mutants, population)

            # Orthogonal Learning: Select two dimensions and optimize
            for i in range(self.pop_size):
                d1, d2 = np.random.choice(self.dim, 2, replace=False)
                
                # Simple line search along the selected dimensions
                alpha = np.linspace(-1, 1, 5)  # Evaluate 5 points
                fitness_values = []
                for a in alpha:
                    x_ol = trial_vectors[i].copy()
                    x_ol[d1] = trial_vectors[i][d1] + a * (func.bounds.ub[d1] - func.bounds.lb[d1]) * 0.01 # scale small amount
                    x_ol[d2] = trial_vectors[i][d2] + a * (func.bounds.ub[d2] - func.bounds.lb[d2]) * 0.01 # scale small amount
                    x_ol = np.clip(x_ol, func.bounds.lb, func.bounds.ub)
                    fitness_values.append(func(x_ol))
                
                best_alpha_idx = np.argmin(fitness_values)
                x_ol = trial_vectors[i].copy()
                x_ol[d1] = trial_vectors[i][d1] + alpha[best_alpha_idx] * (func.bounds.ub[d1] - func.bounds.lb[d1])* 0.01
                x_ol[d2] = trial_vectors[i][d2] + alpha[best_alpha_idx] * (func.bounds.ub[d2] - func.bounds.lb[d2])* 0.01
                x_ol = np.clip(x_ol, func.bounds.lb, func.bounds.ub)
                
                fitness_ol = func(x_ol)
                used_budget += 5  # Each individual uses 5 evaluations on orthogonal learning
                if fitness_ol < fitness[i]:
                   trial_vectors[i] = x_ol
                   fitness[i] = fitness_ol


            # Evaluate trial vectors
            trial_fitness = np.array([func(x) for x in trial_vectors])
            used_budget += self.pop_size

            # Selection
            improved = trial_fitness < fitness
            population[improved] = trial_vectors[improved]
            fitness[improved] = trial_fitness[improved]

            # Update archive
            for x in population[improved]:
                if len(self.archive) < self.archive_size:
                    self.archive.append(x)
                else:
                    # Replace the worst individual in the archive
                    archive_fitness = [func(a) for a in self.archive]
                    worst_idx = np.argmax(archive_fitness)
                    if func(x) < archive_fitness[worst_idx]:
                        self.archive[worst_idx] = x

            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt