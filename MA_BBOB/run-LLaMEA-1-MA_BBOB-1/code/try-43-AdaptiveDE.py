import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=50, local_search_init_radius=0.1, local_search_decay=0.95, rejuvenation_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.archive = []
        self.local_search_init_radius = local_search_init_radius
        self.local_search_decay = local_search_decay
        self.rejuvenation_rate = rejuvenation_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update optimal solution
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        generation = 0
        stagnation_counter = 0
        local_search_radius = self.local_search_init_radius

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Simplified Adaptive F and CR
                F = np.random.uniform(0.3, 0.9)
                CR = np.random.uniform(0.1, 0.9)

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                # Perturb archive usage
                if len(self.archive) > 0 and np.random.rand() < 0.2:
                    archive_idx = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_idx] + 0.01 * np.random.randn(self.dim))  # Add perturbation
                else:
                    mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < CR
                trial[crossover_points] = mutant[crossover_points]
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Update archive (replace the worst)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        worst_idx = np.argmax([func(x) for x in self.archive])  #Find worst using func evals, not stored fitness
                        self.archive[worst_idx] = population[i]

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    #Archive update with trial if it's better than worst in archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                    elif f_trial < np.max([func(x) for x in self.archive]):
                        worst_idx = np.argmax([func(x) for x in self.archive])
                        self.archive[worst_idx] = trial
                        

            population = new_population
            fitness = new_fitness

            # Stagnation check and local search with dynamic radius
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:
                    stagnation_counter += 1
                    if stagnation_counter >= 2:
                        # Local search
                        x_local = np.copy(self.x_opt)
                        for _ in range(min(self.budget // 10, 50)):
                            x_new = x_local + np.random.uniform(-local_search_radius, local_search_radius, size=self.dim)
                            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                            f_new = func(x_new)
                            self.budget -= 1
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new
                                x_local = np.copy(x_new)  # Move center
                        local_search_radius *= self.local_search_decay #Decay search radius

                        stagnation_counter = 0  # Reset stagnation
                else:
                    stagnation_counter = 0
                    local_search_radius = self.local_search_init_radius #Reset radius if not stagnating

                # Population Rejuvenation (More aggressive)
                num_rejuvenated = int(self.rejuvenation_rate * self.pop_size)
                if num_rejuvenated > 0: #Check if any to rejuvenate
                    new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_rejuvenated, self.dim))
                    new_fitnesses = np.array([func(x) for x in new_individuals])
                    self.budget -= num_rejuvenated

                    worst_indices = np.argsort(fitness)[-num_rejuvenated:]
                    population[worst_indices] = new_individuals
                    fitness[worst_indices] = new_fitnesses

                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]


            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt