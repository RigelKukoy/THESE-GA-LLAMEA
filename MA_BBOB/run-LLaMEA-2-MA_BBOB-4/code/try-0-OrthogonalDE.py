import numpy as np

class OrthogonalDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.7, F=0.5, orthogonal_samples=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.F = F
        self.orthogonal_samples = orthogonal_samples # Number of orthogonal samples
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()
            
    def generate_orthogonal_array(self, n, k, l):
        """Generates an orthogonal array using the method described in Taguchi's Orthogonal Arrays.
        n: Number of runs (samples)
        k: Number of factors (variables)
        l: Number of levels (values per variable)
        Note that this simplified version requires n = l**m, where m is an integer.
        """
        if n != l**int(np.log(n)/np.log(l)):
            raise ValueError("n must be a power of l")

        array = np.zeros((n, k), dtype=int)

        # First column
        for i in range(n):
            array[i, 0] = i % l

        # Subsequent columns
        for j in range(1, k):
            for i in range(n):
                array[i, j] = (array[i, 0] + i // (l**(int(np.log(n)/np.log(l)) - int(np.log(j+1)/np.log(l))))) % l
        return array

    def evolve(self, func):
        for i in range(self.pop_size):
            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            x_mutated = x_r1 + self.F * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Orthogonal Learning
            levels = self.orthogonal_samples
            if levels <= 1:
                f_trial = func(x_trial)
                self.eval_count +=1
                if f_trial < self.fitness[i]:
                    self.population[i] = x_trial
                    self.fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = x_trial.copy()
            else:
                n_samples = levels**2 if levels <=5 else levels
                if n_samples > self.budget-self.eval_count:
                    n_samples = max(1, self.budget-self.eval_count)
                
                if n_samples > 1:
                  
                    # Only generate orthogonal array if dimension is suitable for the array size
                    if n_samples >= levels and self.dim <= levels: 
                        try:
                            orthogonal_array = self.generate_orthogonal_array(n_samples, self.dim, levels)
                        except ValueError:
                            orthogonal_array = np.random.randint(0, levels, size=(n_samples, self.dim))

                        candidates = np.zeros((n_samples, self.dim))
                        for k in range(n_samples):
                            for d in range(self.dim):
                                candidates[k, d] = x_trial[d] + (func.bounds.ub[d] - func.bounds.lb[d]) * (orthogonal_array[k, d] / (levels-1) - 0.5)  # Map orthogonal points to search space
                                candidates[k, d] = np.clip(candidates[k, d], func.bounds.lb[d], func.bounds.ub[d])
                                
                        fitness_candidates = np.array([func(x) for x in candidates])
                        self.eval_count += n_samples

                        best_candidate_idx = np.argmin(fitness_candidates)
                        if fitness_candidates[best_candidate_idx] < self.fitness[i]:
                            self.population[i] = candidates[best_candidate_idx]
                            self.fitness[i] = fitness_candidates[best_candidate_idx]

                            if fitness_candidates[best_candidate_idx] < self.f_opt:
                                self.f_opt = fitness_candidates[best_candidate_idx]
                                self.x_opt = candidates[best_candidate_idx].copy()
                    else:
                        f_trial = func(x_trial)
                        self.eval_count +=1
                        if f_trial < self.fitness[i]:
                            self.population[i] = x_trial
                            self.fitness[i] = f_trial
                            if f_trial < self.f_opt:
                                self.f_opt = f_trial
                                self.x_opt = x_trial.copy()

            if self.eval_count >= self.budget:
                break

    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
        return self.f_opt, self.x_opt