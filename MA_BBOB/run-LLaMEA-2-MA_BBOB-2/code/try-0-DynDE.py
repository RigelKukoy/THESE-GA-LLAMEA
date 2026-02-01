import numpy as np

class DynDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F_initial=0.5, CR=0.7, restart_trigger=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F_initial
        self.CR = CR
        self.restart_trigger = restart_trigger
        self.success_rate = 0.5 # Initial success rate

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.success_history = []

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation:
                indices = np.random.choice(self.popsize, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]

                mutant = self.population[i] + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.success_history.append(1)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    self.success_history.append(0)
                
                # Dynamic F adaptation based on success rate
                if len(self.success_history) > 50:
                    recent_success_rate = np.mean(self.success_history[-50:])
                    if recent_success_rate > 0.6:
                        self.F = min(self.F * 1.1, 1.0)  # Increase F if doing well
                    elif recent_success_rate < 0.2:
                        self.F = max(self.F * 0.9, 0.1)  # Decrease F if not doing well

            # Restart mechanism: If no improvement for a while, restart a portion of the population
            if len(self.success_history) > 100 and np.mean(self.success_history[-100:]) < self.restart_trigger:
                 # Sort the population according to the fitness
                sorted_indices = np.argsort(self.fitness)
                
                # Keep the best individuals
                num_elites = int(self.popsize * 0.2)
                elites_indices = sorted_indices[:num_elites]
                elites = self.population[elites_indices]
                elites_fitness = self.fitness[elites_indices]
                
                # Generate new random individuals for the rest of the population
                remaining_popsize = self.popsize - num_elites
                new_population = np.random.uniform(lb, ub, size=(remaining_popsize, self.dim))
                new_fitness = np.array([func(x) for x in new_population])
                self.eval_count += remaining_popsize

                # Combine the elites with the new population
                self.population = np.concatenate((elites, new_population), axis=0)
                self.fitness = np.concatenate((elites_fitness, new_fitness), axis=0)
                
                # Update the best solution if needed
                min_fitness_index = np.argmin(self.fitness)
                self.f_opt = self.fitness[min_fitness_index]
                self.x_opt = self.population[min_fitness_index]

                self.success_history = []

            if self.eval_count > self.budget:
                break
        return self.f_opt, self.x_opt