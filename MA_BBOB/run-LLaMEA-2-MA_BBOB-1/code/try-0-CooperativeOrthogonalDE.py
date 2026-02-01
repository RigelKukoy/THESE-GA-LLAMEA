import numpy as np

class CooperativeOrthogonalDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_subspaces=5, F_init=0.5, CR_init=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_subspaces = num_subspaces
        self.F = np.full(self.pop_size, F_init)
        self.CR = np.full(self.pop_size, CR_init)
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.subspace_dims = [np.random.choice(dim, size=dim // num_subspaces, replace=False) for _ in range(num_subspaces)]
        self.subspace_success = np.zeros(num_subspaces)  # Track success of each subspace
        self.subspace_evals = np.zeros(num_subspaces)  # Track evaluations in each subspace

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def orthogonal_learning(self, x, func, subspace_idx):
        """
        Performs orthogonal learning within a subspace.
        """
        subspace = self.subspace_dims[subspace_idx]
        num_samples = min(10, self.budget // self.num_subspaces) # Dynamic number of samples
        if num_samples <= 0:
            return x

        best_x = x.copy()
        best_f = func(x)
        self.budget -= 1
        self.subspace_evals[subspace_idx] += 1

        for _ in range(num_samples):
            x_new = x.copy()
            for dim_idx in subspace:
                x_new[dim_idx] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            self.budget -= 1
            self.subspace_evals[subspace_idx] += 1

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                self.subspace_success[subspace_idx] += 1  # Subspace was useful

        return best_x

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Choose subspace based on success rate (dynamic resource allocation)
                subspace_idx = np.argmax(self.subspace_success / (self.subspace_evals + 1e-9)) #Exploitation bias, favor subspaces with better success rates.

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                v = self.population[i] + self.F[i] * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR[i] or j == j_rand:
                        u[j] = v[j]
                u = np.clip(u, func.bounds.lb, func.bounds.ub)

                # Orthogonal learning in the selected subspace
                u = self.orthogonal_learning(u, func, subspace_idx)

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()

                    # Update F and CR (simple adaptation)
                    self.F[i] = np.clip(self.F[i] + 0.1 * np.random.normal(), 0.1, 1.0)
                    self.CR[i] = np.clip(self.CR[i] + 0.1 * np.random.normal(), 0.1, 1.0)


        return self.f_opt, self.x_opt