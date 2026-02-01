import numpy as np

class OrthogonalAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, orthogonal_sample_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Initial mutation factor
        self.CR = CR  # Initial crossover rate
        self.orthogonal_sample_size = orthogonal_sample_size # Number of orthogonal samples to generate
        self.archive_size = int(self.pop_size * 0.2)  # Archive size for storing successful solutions
        self.archive = []

    def __orthogonal_design(self, n, k):
        """
        Generates an orthogonal design matrix.
        n: number of runs (sample size)
        k: number of factors (dimensions)
        """
        if n == 4:
            design = np.array([
                [-1, -1],
                [ 1, -1],
                [-1,  1],
                [ 1,  1]
            ])
        elif n == 8:
            design = np.array([
                [-1, -1, -1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [-1,  1,  1],
                [ 1,  1,  1]
            ])
        else:
            raise ValueError("Orthogonal design only implemented for n=4 or n=8")

        # Repeat columns to match k
        design_extended = np.tile(design[:, :min(design.shape[1],k)], (1, int(np.ceil(k/design.shape[1]))))
        return design_extended[:, :k]


    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Evolution loop
        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]

                # Adaptive F: Perturb F based on success
                F_current = self.F + np.random.normal(0, 0.05)
                F_current = np.clip(F_current, 0.1, 1.0)

                v = x_r1 + F_current * (x_r2 - x_r3)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Orthogonal Learning
                if self.orthogonal_sample_size > 1 and self.dim > 1:
                    try:
                        orthogonal_matrix = self.__orthogonal_design(self.orthogonal_sample_size, self.dim)
                        candidates = np.zeros((self.orthogonal_sample_size, self.dim))
                        for k in range(self.orthogonal_sample_size):
                            candidate = np.copy(population[i])
                            for j in range(self.dim):
                                if orthogonal_matrix[k, j] == 1:
                                    candidate[j] = v[j]  # Use mutated component
                            candidates[k] = candidate
                        
                        candidate_fitnesses = [func(candidates[k]) for k in range(self.orthogonal_sample_size)]
                        self.budget -= self.orthogonal_sample_size
                        
                        best_candidate_index = np.argmin(candidate_fitnesses)
                        if candidate_fitnesses[best_candidate_index] < func(u):
                            u = candidates[best_candidate_index]
                            
                    except ValueError:
                        pass


                # Evaluation
                f_u = func(u)
                self.budget -= 1

                if f_u < fitness[i]:
                    # Replacement
                    fitness[i] = f_u
                    population[i] = u

                    # Update archive (if necessary)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        # Replace a random element in the archive
                        replace_index = np.random.randint(self.archive_size)
                        self.archive[replace_index] = population[i]
                    

                    # Adaptive CR: Adjust CR if this solution is better than current best
                    if f_u < self.f_opt:
                        self.CR = self.CR + 0.1 * (1-self.CR) # Increase CR
                        self.CR = np.clip(self.CR, 0.1, 0.9)
                else:
                    self.CR = self.CR - 0.1 * self.CR # Decrease CR
                    self.CR = np.clip(self.CR, 0.1, 0.9)

                # Update best solution
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u.copy() # Important to make a copy!

        return self.f_opt, self.x_opt