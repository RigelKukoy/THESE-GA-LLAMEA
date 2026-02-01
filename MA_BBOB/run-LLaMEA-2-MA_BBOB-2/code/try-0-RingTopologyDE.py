import numpy as np

class RingTopologyDE:
    def __init__(self, budget=10000, dim=10, popsize=40, F=0.5, CR=0.7, topology_size=5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize
        self.F = F
        self.CR = CR
        self.topology_size = topology_size
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.f_opt = np.inf
        self.x_opt = None

    def initialize(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

    def evolve(self, func):
        for i in range(self.popsize):
            # Ring topology selection
            neighbors_idx = [(i - j) % self.popsize for j in range(self.topology_size // 2, 0, -1)] + \
                            [i] + \
                            [(i + j) % self.popsize for j in range(1, self.topology_size // 2 + 1)]

            # Adaptive DE mutation strategy
            if np.random.rand() < 0.5: # Use best neighbor
                best_neighbor_idx = neighbors_idx[np.argmin(self.fitness[neighbors_idx])]
                
                candidates = np.random.choice(neighbors_idx, size=2, replace=False)
                
                x_r1 = self.population[candidates[0]]
                x_r2 = self.population[candidates[1]]
                
                v = self.population[best_neighbor_idx] + self.F * (x_r1 - x_r2)
            else:  # Use random neighbor
                candidates = np.random.choice(neighbors_idx, size=3, replace=False)
                x_r1 = self.population[candidates[0]]
                x_r2 = self.population[candidates[1]]
                x_r3 = self.population[candidates[2]]
                v = x_r1 + self.F * (x_r2 - x_r3)

            # Crossover
            u = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == np.random.randint(self.dim):
                    u[j] = v[j]
                else:
                    u[j] = self.population[i][j]
                    
            # Clip individuals to respect boundaries
            lb = func.bounds.lb
            ub = func.bounds.ub
            u = np.clip(u, lb, ub)

            # Evaluation
            f = func(u)
            self.eval_count += 1

            # Selection
            if f < self.fitness[i]:
                self.population[i] = u
                self.fitness[i] = f

                # Update optimal solution
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = u

    def __call__(self, func):
        self.initialize(func)

        while self.eval_count < self.budget:
            self.evolve(func)
            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt