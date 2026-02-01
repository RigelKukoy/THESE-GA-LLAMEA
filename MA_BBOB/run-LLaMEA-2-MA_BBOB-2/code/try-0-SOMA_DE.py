import numpy as np

class SOMA_DE:
    def __init__(self, budget=10000, dim=10, pop_size=40, path_length=0.1, step_size=0.1, perturbation_chance=0.1, migration_interval=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.path_length = path_length
        self.step_size = step_size
        self.perturbation_chance = perturbation_chance
        self.migration_interval = migration_interval
        self.population = None
        self.fitness = None
        self.leader_index = None
        self.generation = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.leader_index = np.argmin(self.fitness)

    def migrate(self, func):
        for i in range(self.pop_size):
            if i == self.leader_index:
                continue

            for step in np.arange(self.step_size, self.path_length + self.step_size, self.step_size):
                new_position = self.population[i] + step * (self.population[self.leader_index] - self.population[i])

                # Perturbation
                for d in range(self.dim):
                    if np.random.rand() < self.perturbation_chance:
                        new_position[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)

                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                new_fitness = func(new_position)
                self.budget -= 1
                if self.budget <= 0:
                    return

                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    if new_fitness < self.fitness[self.leader_index]:
                        self.leader_index = i

    def differential_evolution_mutation(self, func):
        # Apply DE mutation to each individual (except the leader)
        for i in range(self.pop_size):
            if i == self.leader_index:
                continue

            # Select three random individuals (a, b, c)
            indices = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = self.population[indices]

            # Mutation: v = a + F * (b - c)
            mutation_factor = 0.8  # Fixed mutation factor
            mutated_vector = a + mutation_factor * (b - c)
            mutated_vector = np.clip(mutated_vector, func.bounds.lb, func.bounds.ub)

            # Crossover (Binomial/Uniform)
            crossover_rate = 0.7
            trial_vector = np.copy(self.population[i])  # Start with the current individual
            for d in range(self.dim):
                if np.random.rand() < crossover_rate:
                    trial_vector[d] = mutated_vector[d]

            # Evaluate the trial vector
            trial_fitness = func(trial_vector)
            self.budget -= 1
            if self.budget <= 0:
                return
            

            # Selection: Replace if the trial vector is better
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial_vector
                self.fitness[i] = trial_fitness

                if trial_fitness < self.fitness[self.leader_index]:
                    self.leader_index = i
    
    def adapt_population(self, func):
        # Check for stagnation (e.g., little improvement in leader's fitness)
        stagnation_threshold = 1e-6
        if self.generation > 50 and np.abs(self.fitness[self.leader_index] - self.previous_leader_fitness) < stagnation_threshold:
            # Introduce new random individuals to increase diversity
            num_new_individuals = int(self.pop_size * 0.2)  # Replace 20% of population
            new_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_new_individuals, self.dim))
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= num_new_individuals
            if self.budget <= 0:
                return
            

            # Replace the worst individuals with the new ones
            worst_indices = np.argsort(self.fitness)[-num_new_individuals:]
            self.population[worst_indices] = new_population
            self.fitness[worst_indices] = new_fitness

            # Update the leader
            self.leader_index = np.argmin(self.fitness)

        self.previous_leader_fitness = self.fitness[self.leader_index]

    def __call__(self, func):
        self.initialize_population(func)
        self.previous_leader_fitness = self.fitness[self.leader_index]

        while self.budget > 0:
            self.migrate(func)
            if self.budget <= 0:
                break
            
            self.differential_evolution_mutation(func)
            if self.budget <= 0:
                break
            
            self.adapt_population(func)
            if self.budget <= 0:
                break

            self.generation += 1

        self.f_opt = self.fitness[self.leader_index]
        self.x_opt = self.population[self.leader_index]
        return self.f_opt, self.x_opt