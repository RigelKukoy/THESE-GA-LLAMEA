import numpy as np

class SHADE_LocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, memory_size=10, p_local_search=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_F = np.full(self.memory_size, 0.5)
        self.archive = []
        self.archive_size = pop_size
        self.p_local_search = p_local_search

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for i in range(self.pop_size):
                # Selection of CR and F
                rand_index = np.random.randint(self.memory_size)
                CR = self.memory_CR[rand_index]
                F = self.memory_F[rand_index]

                # Mutation
                p_best_index = np.argmin(fitness)
                x_pbest = population[p_best_index]
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x_r1, x_r2 = population[indices]

                v = population[i] + F * (x_pbest - population[i]) + F * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1
                if f_u < fitness[i]:
                    new_fitness[i] = f_u
                    new_population[i] = u
                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = population[i]
                
                #Probabilistic Local Search
                if np.random.rand() < self.p_local_search:
                    x_local = np.clip(population[i] + np.random.normal(0, 0.05, self.dim), func.bounds.lb, func.bounds.ub)
                    f_local = func(x_local)
                    self.budget -= 1
                    if f_local < new_fitness[i]:
                        new_fitness[i] = f_local
                        new_population[i] = x_local
                
                # Update best solution
                if new_fitness[i] < self.f_opt:
                    self.f_opt = new_fitness[i]
                    self.x_opt = new_population[i].copy() # Important to make a copy!
            
            # Update population and fitness
            population = new_population
            fitness = new_fitness

            # SHADE Memory Update (simplified)
            successful_CRs = []
            successful_Fs = []
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    successful_CRs.append(self.memory_CR[np.random.randint(self.memory_size)])
                    successful_Fs.append(self.memory_F[np.random.randint(self.memory_size)])
            
            if successful_CRs: #Check if there are successful crossovers before calculating mean
                self.memory_CR[np.random.randint(self.memory_size)] = np.mean(successful_CRs)
            if successful_Fs: #Check if there are successful Fs before calculating mean
                self.memory_F[np.random.randint(self.memory_size)] = np.mean(successful_Fs)
                self.memory_F[np.random.randint(self.memory_size)] = np.clip(self.memory_F[np.random.randint(self.memory_size)], 0.1, 1.0)

        return self.f_opt, self.x_opt