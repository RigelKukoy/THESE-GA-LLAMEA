import numpy as np

class LearningCauchyDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.learning_rate = learning_rate
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Learning-based mutation: Adapt F based on success
                if np.random.rand() < 0.5:  # Apply with a probability
                    success = False
                    
                    # Mutation
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[indices]
                    v = self.population[i] + self.F * (x_r1 - x_r2)
                    v = np.clip(v, func.bounds.lb, func.bounds.ub)

                    # Crossover
                    j_rand = np.random.randint(self.dim)
                    u = self.population[i].copy()
                    for j in range(self.dim):
                        if np.random.rand() < self.CR or j == j_rand:
                            u[j] = v[j]
                    
                    # Cauchy perturbation
                    cauchy_noise = np.random.standard_cauchy(size=self.dim) * 0.01  # Adjust scale
                    u = np.clip(u + cauchy_noise, func.bounds.lb, func.bounds.ub)

                    # Evaluation
                    f_u = func(u)
                    self.budget -= 1

                    # Selection
                    if f_u < self.fitness[i]:
                        success = True
                        self.fitness[i] = f_u
                        self.population[i] = u
                        if f_u < self.f_opt:
                            self.f_opt = f_u
                            self.x_opt = u.copy()

                    # Update F based on success
                    if success:
                        self.F += self.learning_rate * (1 - self.F)  # Increase F if successful
                    else:
                        self.F -= self.learning_rate * self.F  # Decrease F if unsuccessful
                    self.F = np.clip(self.F, 0.1, 1.0)  # Keep F within bounds

                else:
                    # Standard DE mutation if learning not applied

                    # Mutation
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[indices]
                    v = self.population[i] + self.F * (x_r1 - x_r2)
                    v = np.clip(v, func.bounds.lb, func.bounds.ub)

                    # Crossover
                    j_rand = np.random.randint(self.dim)
                    u = self.population[i].copy()
                    for j in range(self.dim):
                        if np.random.rand() < self.CR or j == j_rand:
                            u[j] = v[j]

                    # Cauchy perturbation
                    cauchy_noise = np.random.standard_cauchy(size=self.dim) * 0.01  # Adjust scale
                    u = np.clip(u + cauchy_noise, func.bounds.lb, func.bounds.ub)
                    
                    # Evaluation
                    f_u = func(u)
                    self.budget -= 1

                    # Selection
                    if f_u < self.fitness[i]:
                        self.fitness[i] = f_u
                        self.population[i] = u
                        if f_u < self.f_opt:
                            self.f_opt = f_u
                            self.x_opt = u.copy()
                            
        return self.f_opt, self.x_opt