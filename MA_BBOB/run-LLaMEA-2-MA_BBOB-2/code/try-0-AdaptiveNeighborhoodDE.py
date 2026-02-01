import numpy as np

class AdaptiveNeighborhoodDE:
    def __init__(self, budget=10000, dim=10, popsize=None, step_size_init=0.1, step_size_min=0.001, de_mutation_factor=0.5, de_crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 5 * self.dim
        self.step_size = step_size_init
        self.step_size_min = step_size_min
        self.de_mutation_factor = de_mutation_factor
        self.de_crossover_rate = de_crossover_rate

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize a single solution (no population)
        self.x_opt = np.random.uniform(lb, ub, size=self.dim)
        self.f_opt = func(self.x_opt)
        self.eval_count = 1

        while self.eval_count < self.budget:
            # Generate neighbors using adaptive step size
            neighbors = []
            for _ in range(self.popsize):
                neighbor = self.x_opt + np.random.normal(0, self.step_size, size=self.dim)
                neighbor = np.clip(neighbor, lb, ub)
                neighbors.append(neighbor)
            
            neighbors = np.array(neighbors)

            # Differential Evolution inspired exploration among neighbors
            if self.popsize >= 3:
                idxs = np.random.choice(self.popsize, size=(self.popsize, 3), replace=False)
                v = neighbors[idxs[:, 0]] + self.de_mutation_factor * (neighbors[idxs[:, 1]] - neighbors[idxs[:, 2]])
                v = np.clip(v, lb, ub)
                
                # Crossover
                cross_mask = np.random.rand(self.popsize, self.dim) < self.de_crossover_rate
                trial_vectors = np.where(cross_mask, v, neighbors)
            else:
                trial_vectors = neighbors #DE cannot be applied

            # Evaluate neighbors and trial vectors
            fitness_neighbors = np.array([func(x) for x in neighbors])
            fitness_trials = np.array([func(x) for x in trial_vectors])
            self.eval_count += 2 * self.popsize
            
            # Selection: keep the best among current solution, neighbors, and trial vectors
            all_candidates = np.concatenate(([self.x_opt], neighbors, trial_vectors))
            all_fitnesses = np.concatenate(([self.f_opt], fitness_neighbors, fitness_trials))
            
            best_index = np.argmin(all_fitnesses)
            
            if best_index == 0:
                pass  # Current solution is still the best
            elif best_index <= self.popsize:
                self.x_opt = neighbors[best_index-1]
                self.f_opt = fitness_neighbors[best_index-1]
            else:
                self.x_opt = trial_vectors[best_index-1-self.popsize]
                self.f_opt = fitness_trials[best_index-1-self.popsize]


            # Adaptive step size adjustment
            if self.f_opt < min(fitness_neighbors.min(), fitness_trials.min()):
                 self.step_size *= 1.05  # Increase step size if improvement
            else:
                self.step_size *= 0.95  # Decrease step size if no improvement
            self.step_size = max(self.step_size, self.step_size_min)


        return self.f_opt, self.x_opt