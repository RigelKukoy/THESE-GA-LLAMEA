import numpy as np

class OrthogonalLearningDE:
    def __init__(self, budget=10000, dim=10, pop_size=30, F=0.5, Cr=0.9, lr_init=0.1, lr_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.lr_init = lr_init
        self.lr_decay = lr_decay

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        best_solution = population[best_index].copy()

        learning_rate = self.lr_init

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial_vector = np.where(crossover_mask, mutant, population[i])

                # Evaluate
                trial_fitness = func(trial_vector)
                self.budget -= 1
                if self.budget <= 0:
                    break

                # Selection
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial_vector

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector.copy()
                        learning_rate = self.lr_init  # Reset learning rate upon improvement
                
                # Orthogonal Learning-based Population Update
                else:
                    # Create orthogonal array (simplified - 2-level factorial design)
                    oa_matrix = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) # 2 factors, 2 levels
                    
                    # Select two random dimensions
                    dims_to_explore = np.random.choice(self.dim, 2, replace=False)
                    
                    # Iterate through orthogonal array points
                    for oa_point in oa_matrix:
                        new_individual = population[i].copy()

                        # Map OA levels to function bounds for selected dimensions
                        for j, dim_idx in enumerate(dims_to_explore):
                            if oa_point[j] == 1:
                                new_individual[dim_idx] = np.random.uniform(population[i][dim_idx], func.bounds.ub)
                            else:
                                new_individual[dim_idx] = np.random.uniform(func.bounds.lb, population[i][dim_idx])
                        
                            new_individual[dim_idx] = np.clip(new_individual[dim_idx], func.bounds.lb, func.bounds.ub)
                        
                        new_fitness = func(new_individual)
                        self.budget -= 1
                        if self.budget <= 0:
                           break
                        
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness
                            population[i] = new_individual.copy()
                            if new_fitness < best_fitness:
                                best_fitness = new_fitness
                                best_solution = new_individual.copy()
                        
                    if self.budget <= 0:
                        break


            learning_rate *= self.lr_decay #Decay learning rate each generation

        self.f_opt = best_fitness
        self.x_opt = best_solution
        return self.f_opt, self.x_opt