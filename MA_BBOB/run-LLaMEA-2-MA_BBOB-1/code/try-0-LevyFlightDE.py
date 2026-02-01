import numpy as np

class LevyFlightDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, levy_exponent=1.5, stagnation_threshold=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate
        self.levy_exponent = levy_exponent  # Exponent for Lévy flight
        self.stagnation_threshold = stagnation_threshold # Number of iterations without improvement to trigger restart.
        self.best_fitness_history = []

    def levy_flight(self, size):
        # Generate Lévy flight steps
        num = np.random.randn(size) * np.sqrt(self.sigma(self.levy_exponent))
        den = np.power(np.abs(np.random.randn(size)), (1/self.levy_exponent))
        steps = num / den
        return steps

    def sigma(self, beta):
        num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = np.math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)
        sigma = np.power((num / den), (1 / beta))
        return sigma

    def __call__(self, func):
        # Initialization
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        self.best_fitness_history.append(self.f_opt)

        # Evolution loop
        iteration = 0
        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation using Lévy flight
                levy_steps = self.levy_flight(self.dim)
                mutated_vector = population[i] + self.F * levy_steps * (self.x_opt - population[i]) # biased towards best solution, combined with Levy flight

                mutated_vector = np.clip(mutated_vector, lb, ub) # keep within bounds

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial_vector[j] = mutated_vector[j]

                # Evaluation
                f_trial = func(trial_vector)
                self.budget -= 1

                if f_trial < fitness[i]:
                    # Replacement
                    fitness[i] = f_trial
                    population[i] = trial_vector

                    # Update best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial_vector.copy()
                        self.best_fitness_history.append(self.f_opt)
                else:
                    self.best_fitness_history.append(self.f_opt)

            # Stagnation Check and Restart Mechanism
            if iteration > self.stagnation_threshold and np.std(self.best_fitness_history[-self.stagnation_threshold:]) < 1e-6:
                # Restart: Re-initialize a portion of the population
                num_to_restart = int(0.2 * self.pop_size) # Restart 20% of population
                indices_to_restart = np.random.choice(self.pop_size, num_to_restart, replace=False)
                population[indices_to_restart] = np.random.uniform(lb, ub, size=(num_to_restart, self.dim))
                fitness[indices_to_restart] = np.array([func(x) for x in population[indices_to_restart]])
                self.budget -= num_to_restart # Update budget
                
                best_index = np.argmin(fitness) #Recalculate best solution
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
                self.best_fitness_history.append(self.f_opt)
                
                self.F = min(self.F * 1.2, 1.0) # slightly increase F to promote diversity after restart.


            iteration += 1

        return self.f_opt, self.x_opt