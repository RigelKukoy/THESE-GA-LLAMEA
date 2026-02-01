import numpy as np

class OrthogonalDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.9, F=0.5, orthogonal_samples=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.F = F
        self.orthogonal_samples = orthogonal_samples
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.budget -= self.pop_size

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        return x_r1 + self.F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def orthogonal_design(self, x_i, func):
        levels = 3  # Using 3 levels for each dimension
        design = self.generate_orthogonal_array(self.dim, levels)
        
        candidates = []
        fitnesses = []

        for i in range(design.shape[0]):  # Iterate through orthogonal samples
            x_new = np.copy(x_i)
            for j in range(self.dim):
                level = design[i, j]
                range_dim = func.bounds.ub[j] - func.bounds.lb[j]
                x_new[j] = func.bounds.lb[j] + (level / (levels - 1)) * range_dim
            
            x_new = self.repair(x_new, func)
            f_new = func(x_new)
            self.budget -= 1
            
            candidates.append(x_new)
            fitnesses.append(f_new)
            
        best_index = np.argmin(fitnesses)
        return candidates[best_index], fitnesses[best_index]

    def generate_orthogonal_array(self, factors, levels):
        # This is a simplified example and might not cover all cases.
        # For a more robust implementation, consider using libraries like pyDOE.
        if factors <= levels:
            design = np.zeros((levels, factors), dtype=int)
            for i in range(levels):
                for j in range(factors):
                    design[i, j] = i
            return design
        else:
            # Simple alternative for factors > levels
            design = np.random.randint(0, levels, size=(levels, factors))
            return design
            

    def self_adaptive_parameters(self, success):
        if success:
            self.CR = min(1.0, self.CR + 0.1)
            self.F = min(1.0, self.F + 0.1)
        else:
            self.CR = max(0.1, self.CR - 0.1)
            self.F = max(0.1, self.F - 0.1)

    def diversity_preservation(self):
        # Replace the worst individual with a random one if population is too similar
        std = np.std(self.population)
        if std < 1e-6:  # Threshold for similarity
            worst_index = np.argmax(self.fitness)
            self.population[worst_index] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
            self.fitness[worst_index] = func(self.population[worst_index])
            self.budget -= 1

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i])

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Orthogonal learning
                u_i, f_u_i = self.orthogonal_design(u_i, func)
                
                success = False
                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    success = True

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i
                
                self.self_adaptive_parameters(success)
                
            self.diversity_preservation()

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt