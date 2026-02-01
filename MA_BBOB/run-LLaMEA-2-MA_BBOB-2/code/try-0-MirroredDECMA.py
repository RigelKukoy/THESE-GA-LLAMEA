import numpy as np

class MirroredDECMA:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.learning_rate = learning_rate
        self.C = np.eye(dim)  # Covariance matrix
        self.mean = None

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        if self.mean is None:
            self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        
        population = np.random.multivariate_normal(self.mean, self.C, size=self.pop_size)
        population = np.clip(population, func.bounds.lb, func.bounds.ub)
        
        fitness = np.array([func(x) for x in population])
        used_budget = self.pop_size
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        while used_budget < self.budget:
            # Generate mutants using DE/rand/1 and mirrored sampling
            mutants = []
            mirrored_mutants = []
            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                mutant = population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                mutants.append(mutant)

                # Mirrored sampling: Reflect the mutant around the mean
                mirrored_mutant = 2 * self.mean - mutant
                mirrored_mutant = np.clip(mirrored_mutant, func.bounds.lb, func.bounds.ub)
                mirrored_mutants.append(mirrored_mutant)

            mutants = np.array(mutants)
            mirrored_mutants = np.array(mirrored_mutants)
            
            # Crossover
            crossover_mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial_vectors = np.where(crossover_mask, mutants, population)
            mirrored_trial_vectors = np.where(crossover_mask, mirrored_mutants, population) # Use mirrored mutants also
            
            # Evaluate trial vectors
            trial_fitness = np.array([func(x) for x in trial_vectors])
            mirrored_trial_fitness = np.array([func(x) for x in mirrored_trial_vectors])
            used_budget += 2*self.pop_size
            
            # Selection
            improved = trial_fitness < fitness
            mirrored_improved = mirrored_trial_fitness < fitness
            
            # Update population and fitness
            fitness[improved] = trial_fitness[improved]
            population[improved] = trial_vectors[improved]

            fitness[mirrored_improved] = mirrored_trial_fitness[mirrored_improved]
            population[mirrored_improved] = mirrored_trial_vectors[mirrored_improved]
            
            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]
            
            # Update mean and covariance matrix adaptively
            delta = population - self.mean
            weighted_delta = np.mean(delta[improved], axis=0) if np.any(improved) else np.zeros(self.dim)
            
            self.mean = (1 - self.learning_rate) * self.mean + self.learning_rate * np.mean(population, axis=0)
            
            # Update covariance matrix using rank-one update (simplified)
            if np.any(improved) or np.any(mirrored_improved):
                improved_indices = np.concatenate([np.where(improved)[0], np.where(mirrored_improved)[0]])
                d = population[improved_indices] - self.mean
                if d.size > 0:
                    self.C = (1 - self.learning_rate) * self.C + self.learning_rate * np.cov(d.T)
                else:
                    self.C = (1 - self.learning_rate) * self.C + self.learning_rate * np.eye(self.dim)
            else:
                 self.C = (1- self.learning_rate) * self.C + self.learning_rate * np.eye(self.dim)
            

            # Ensure covariance matrix is positive semi-definite
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)  # Reset if not positive semi-definite

            used_budget = min(used_budget, self.budget)  # Ensure not exceeding budget
            if used_budget >= self.budget:
                break
                
        return self.f_opt, self.x_opt