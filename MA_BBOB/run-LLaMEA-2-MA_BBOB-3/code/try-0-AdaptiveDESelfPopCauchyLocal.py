import numpy as np

class AdaptiveDESelfPopCauchyLocal:
    def __init__(self, budget=10000, dim=10, pop_size_init=50, CR=0.9, F=0.5, local_search_prob=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size_init
        self.CR = CR
        self.F = F
        self.local_search_prob = local_search_prob
        self.local_search_radius = local_search_radius
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size_history = [pop_size_init]


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
        # Cauchy mutation
        return x_r1 + self.F * (x_r2 - x_r3) + np.random.standard_cauchy(size=self.dim) * 0.01


    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def local_search(self, x, func):
        x_new = x + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
        x_new = self.repair(x_new, func)
        f_new = func(x_new)
        self.budget -= 1
        return x_new, f_new

    def adjust_population_size(self):
        improvement_threshold = 1e-5
        recent_history = self.pop_size_history[-min(5, len(self.pop_size_history)):]
        if len(recent_history) < 5:
            return # not enough data

        if self.f_opt - np.min(self.fitness) > improvement_threshold:
            self.pop_size = min(self.pop_size + 5, 200)  # Increase if improving
        else:
            self.pop_size = max(self.pop_size - 5, 10)  # Decrease if stagnating
        
        self.pop_size_history.append(self.pop_size)



    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_population_size()
            
            # Resize population if pop_size changed
            if self.pop_size != len(self.population):
                old_pop_size = len(self.population)
                
                if self.pop_size > old_pop_size:
                    # Add new random individuals
                    new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - old_pop_size, self.dim))
                    new_fitness = np.array([func(x) for x in new_individuals])
                    self.budget -= (self.pop_size - old_pop_size)
                    self.population = np.vstack((self.population, new_individuals))
                    self.fitness = np.concatenate((self.fitness, new_fitness))
                else:
                    # Reduce population size by removing the worst individuals
                    indices_to_remove = np.argsort(self.fitness)[self.pop_size:]
                    self.population = np.delete(self.population, indices_to_remove, axis=0)
                    self.fitness = np.delete(self.fitness, indices_to_remove)
            
            
            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i])

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            # Local search around the best solution
            if np.random.rand() < self.local_search_prob:
                x_new, f_new = self.local_search(self.x_opt, func)
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_new

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt