import numpy as np

class AdaptiveDERestart:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, restart_prob=0.05):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.restart_prob = restart_prob
        self.learning_rate = 1.0  # Initial learning rate for parameter adaptation
        self.learning_rate_decay = 0.995  # Decay factor for learning rate

    def _mutation_rand1(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        return population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])

    def _crossover(self, mutant, target):
         return np.where(np.random.rand(self.dim) < self.CR, mutant, target)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        used_budget = self.pop_size
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        generation = 0
        while used_budget < self.budget:
            generation += 1
            
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            for i in range(self.pop_size):
                # Mutation
                mutant = self._mutation_rand1(population, best_idx)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                trial_vector = self._crossover(mutant, population[i])
                
                # Evaluation
                f = func(trial_vector)
                used_budget += 1
                
                # Selection
                if f < fitness[i]:
                    new_fitness[i] = f
                    new_population[i] = trial_vector
                    
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial_vector

            population = new_population
            fitness = new_fitness
            best_idx = np.argmin(fitness)
            
            # Parameter Adaptation with learning rate annealing
            self.F = np.clip(self.F + self.learning_rate * np.random.normal(0, 0.01), 0.1, 0.9)
            self.CR = np.clip(self.CR + self.learning_rate * np.random.normal(0, 0.01), 0.1, 0.9)
            
            # Restart mechanism
            if np.random.rand() < self.restart_prob:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                used_budget += self.pop_size
                
                best_idx = np.argmin(fitness)
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

            # Anneal the learning rate
            self.learning_rate *= self.learning_rate_decay


                
        return self.f_opt, self.x_opt