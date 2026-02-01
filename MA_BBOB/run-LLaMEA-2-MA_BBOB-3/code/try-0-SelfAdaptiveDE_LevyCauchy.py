import numpy as np

class SelfAdaptiveDE_LevyCauchy:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_CR=0.5, initial_F=0.7, levy_exponent=1.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = initial_CR
        self.F = initial_F
        self.levy_exponent = levy_exponent
        self.population = None
        self.fitness = None
        self.archive = []  # Archive for storing discarded solutions

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

    def levy_flight(self, shape, exponent=1.5):
        """Generate Levy distributed steps."""
        numerator = np.gamma(1 + exponent) * np.sin(np.pi * exponent / 2)
        denominator = np.gamma((1 + exponent) / 2) * exponent * (2 ** ((exponent - 1) / 2))
        sigma = (numerator / denominator) ** (1 / exponent)
        u = np.random.normal(0, sigma, shape)
        v = np.random.normal(0, 1, shape)
        return u / (np.abs(v) ** (1 / exponent))

    def mutate(self, x_i, func):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        
        levy_steps = self.levy_flight(self.dim, self.levy_exponent)
        v_i = x_r1 + self.F * (x_r2 - x_r3) + levy_steps * (func.bounds.ub - func.bounds.lb) 
        
        return v_i

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        
        # Cauchy distributed crossover probability
        crossover_mask = np.random.rand(self.dim) < np.abs(np.random.standard_cauchy(size=self.dim) * self.CR)
        crossover_mask[j_rand] = True  # Ensure at least one gene is exchanged
        u_i[crossover_mask] = v_i[crossover_mask]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i], func)

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    # Archive the replaced solution
                    self.archive.append(self.population[i].copy())  
                    
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            # Self-adaptation of CR and F (simple adaptation)
            self.CR = np.clip(np.random.normal(self.CR, 0.1), 0.1, 0.9)
            self.F = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0)
            

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt