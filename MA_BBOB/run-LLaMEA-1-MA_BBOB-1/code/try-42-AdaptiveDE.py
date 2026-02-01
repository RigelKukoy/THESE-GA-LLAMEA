import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, local_search_radius=0.1, momentum=0.1, age_limit=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.local_search_radius = local_search_radius
        self.momentum = momentum
        self.velocity = np.zeros((pop_size, dim))  # Initialize velocity for momentum
        self.age_limit = age_limit
        self.ages = np.zeros(pop_size, dtype=int)  # Initialize ages for each individual

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update optimal solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]

        generation = 0
        stagnation_counter = 0
        CR_history = np.full(self.pop_size, self.CR_base)  # CR history for self-adaptation

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            self.ages += 1  # Increase age for all individuals

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                F = np.clip(F, 0.1, 1.0)

                # Self-adaptive CR
                CR = CR_history[i] + np.random.normal(0, self.CR_range)
                CR = np.clip(CR, 0.1, 0.9)

                # Mutation with momentum
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                # Momentum update
                self.velocity[i] = self.momentum * self.velocity[i] + F * (x2 - x3)
                mutant = x1 + self.velocity[i]

                # Use archive
                if np.random.rand() < 0.1 and len(self.archive) > 0:
                    archive_index = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_index])

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    CR_history[i] = CR  # store successful CR
                    self.ages[i] = 0 #reset age

                    # Add replaced vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        self.archive[worst_archive_index] = population[i]
                        self.archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    # Add trial vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        self.archive[worst_archive_index] = trial
                        self.archive_fitness[worst_archive_index] = f_trial

            population = new_population
            fitness = new_fitness

            # Dynamic population size adjustment
            fitness_std = np.std(fitness)
            if fitness_std < 1e-5 and self.pop_size > 10:
                self.pop_size = max(10, int(self.pop_size * 0.9))  # Reduce population size
                population = population[:self.pop_size]
                fitness = fitness[:self.pop_size]
                self.velocity = self.velocity[:self.pop_size]
                self.ages = self.ages[:self.pop_size]
                CR_history = CR_history[:self.pop_size]

            elif fitness_std > 1e-2 and self.pop_size < 100:
                self.pop_size = min(100, int(self.pop_size * 1.1))  # Increase population size
                # Add new random individuals
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - len(population), self.dim))
                new_fitnesses = np.array([func(x) for x in new_individuals])
                self.budget -= len(new_individuals)

                population = np.concatenate([population, new_individuals])
                fitness = np.concatenate([fitness, new_fitnesses])
                self.velocity = np.concatenate([self.velocity, np.zeros((len(new_individuals), self.dim))])
                self.ages = np.concatenate([self.ages, np.zeros(len(new_individuals), dtype=int)])
                CR_history = np.concatenate([CR_history, np.full(len(new_individuals), self.CR_base)])

            # Aging mechanism
            for i in range(self.pop_size):
                if self.ages[i] > self.age_limit:
                    # Replace old individual with a new random one
                    population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    fitness[i] = func(population[i])
                    self.budget -= 1
                    self.velocity[i] = np.zeros(self.dim)
                    self.ages[i] = 0

                    # Update optimal solution
                    if fitness[i] < self.f_opt:
                        self.f_opt = fitness[i]
                        self.x_opt = population[i]

            # Stagnation check and periodic population rejuvenation
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:  # Stagnation criterion
                    stagnation_counter += 1
                    if stagnation_counter >= 2:
                        # Perform local search around the best solution
                        x_local = np.copy(self.x_opt)
                        for _ in range(min(self.budget // 10, 100)):  # Limited local search budget
                            x_new = x_local + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                            f_new = func(x_new)
                            self.budget -= 1
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new
                                x_local = np.copy(x_new)  # Move center of local search
                        stagnation_counter = 0  # Reset stagnation

                else:
                    stagnation_counter = 0  # Reset stagnation

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt