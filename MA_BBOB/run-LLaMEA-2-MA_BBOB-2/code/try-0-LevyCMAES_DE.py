import numpy as np

class LevyCMAES_DE:
    def __init__(self, budget=10000, dim=10, popsize=None, F_initial=0.5, CR_initial=0.7, restart_threshold=100, CMA_learning_rate=0.1, rejuvenation_frequency=50):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F_initial
        self.CR = CR_initial
        self.restart_threshold = restart_threshold
        self.CMA_learning_rate = CMA_learning_rate
        self.rejuvenation_frequency = rejuvenation_frequency
        self.best_fitness_history = []
        self.mean = None
        self.C = None # Covariance matrix
        self.ps = None # Evolution path for sigma
        self.pc = None # Evolution path for covariance
        self.sigma = 0.1 # Overall standard deviation
        self.eval_count = 0

    def levy_flight(self, beta=1.5):
        """
        Generates a Lévy flight step.
        """
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v)**(1/beta)
        return step

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize population and CMA-ES parameters
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        self.mean = self.x_opt.copy()  # Initialize mean with the best solution
        self.C = np.eye(self.dim)  # Initialize covariance matrix
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        stagnation_counter = 0

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Self-adaptive parameters
                self.F = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0)
                self.CR = np.clip(np.random.normal(self.CR, 0.1), 0.1, 1.0)

                # Mutation using Levy Flight
                levy_step = self.levy_flight()
                mutant = self.population[i] + self.F * levy_step * self.sigma
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])
                trial = np.clip(trial, lb, ub)


                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            # CMA-ES update
            weights = np.clip(0.5 + np.array([func(x) for x in self.population]) / self.f_opt, 0, 1) # weighting the fitness function
            weights /= np.sum(weights)
            old_mean = self.mean.copy()
            self.mean = np.sum(self.population * weights[:, np.newaxis], axis=0)

            self.ps = (1 - self.CMA_learning_rate) * self.ps + np.sqrt(self.CMA_learning_rate * (2 - self.CMA_learning_rate)) * (self.mean - old_mean) / self.sigma
            self.pc = (1 - self.CMA_learning_rate) * self.pc + np.sqrt(self.CMA_learning_rate * (2 - self.CMA_learning_rate)) * (self.mean - old_mean) / self.sigma

            self.C = (1 - self.CMA_learning_rate) * self.C + self.CMA_learning_rate * (np.outer(self.pc, self.pc) + 0.001 * np.eye(self.dim)) # Adding the identity matrix to avoid singularity

            # Update sigma (simplified version)
            self.sigma *= np.exp(0.5 * (np.linalg.norm(self.ps)**2 - self.dim) / (self.dim + 5))

            # Stagnation Check and Restart
            self.best_fitness_history.append(self.f_opt)
            if len(self.best_fitness_history) > self.restart_threshold:
                if abs(self.best_fitness_history[-1] - self.best_fitness_history[-self.restart_threshold]) < 1e-6:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                if stagnation_counter > self.restart_threshold // 2:
                    # Restart Population
                    self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                    self.fitness = np.array([func(x) for x in self.population])
                    self.eval_count += self.popsize

                    self.f_opt = np.min(self.fitness)
                    self.x_opt = self.population[np.argmin(self.fitness)]
                    self.best_fitness_history = [self.f_opt]  # Reset fitness history

                    # Re-initialize CMA-ES parameters
                    self.mean = self.x_opt.copy()
                    self.C = np.eye(self.dim)
                    self.ps = np.zeros(self.dim)
                    self.pc = np.zeros(self.dim)
                    self.sigma = 0.1

                    stagnation_counter = 0

            # Population Rejuvenation
            if (self.eval_count // self.popsize) % self.rejuvenation_frequency == 0:
                indices_to_rejuvenate = np.random.choice(self.popsize, self.popsize // 2, replace=False)
                self.population[indices_to_rejuvenate] = np.random.uniform(lb, ub, size=(len(indices_to_rejuvenate), self.dim))
                self.fitness[indices_to_rejuvenate] = np.array([func(x) for x in self.population[indices_to_rejuvenate]])
                self.eval_count += len(indices_to_rejuvenate)
                best_index = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

        return self.f_opt, self.x_opt