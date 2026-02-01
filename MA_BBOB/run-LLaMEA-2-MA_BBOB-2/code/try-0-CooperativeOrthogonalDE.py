import numpy as np

class CooperativeOrthogonalDE:
    def __init__(self, budget=10000, dim=10, num_subpopulations=5, initial_popsize=None, F=0.5, CR=0.7, orthogonal_dimension=3, dynamic_popsize=True):
        self.budget = budget
        self.dim = dim
        self.num_subpopulations = num_subpopulations
        self.initial_popsize = initial_popsize if initial_popsize is not None else 5 * self.dim
        self.F = F
        self.CR = CR
        self.orthogonal_dimension = orthogonal_dimension
        self.dynamic_popsize = dynamic_popsize
        self.subpopulations = []
        self.fitness = []
        self.popsize = []
        self.eval_counts = []
        self.lb = None
        self.ub = None
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_subpopulations(self, func):
        lb = self.lb
        ub = self.ub
        for i in range(self.num_subpopulations):
            self.popsize.append(self.initial_popsize)
            self.subpopulations.append(np.random.uniform(lb, ub, size=(self.popsize[i], self.dim)))
            self.fitness.append(np.array([func(x) for x in self.subpopulations[i]]))
            self.eval_counts.append(self.popsize[i])  # Track evaluations per subpopulation
            
            best_idx = np.argmin(self.fitness[i])
            if self.fitness[i][best_idx] < self.f_opt:
                self.f_opt = self.fitness[i][best_idx]
                self.x_opt = self.subpopulations[i][best_idx]

    def orthogonal_design(self, x_center, orthogonal_dimension):
        """Generate an orthogonal design around x_center."""
        design = np.zeros((orthogonal_dimension + 1, orthogonal_dimension))
        for i in range(1, orthogonal_dimension + 1):
            for j in range(orthogonal_dimension):
                if ((i - 1) >> j) & 1:
                    design[i, j] = 1
                else:
                    design[i, j] = -1

        return design

    def orthogonal_learning(self, func, x_best):
        """Perform orthogonal learning around the best solution."""
        orthogonal_dimension = min(self.orthogonal_dimension, self.dim)
        design = self.orthogonal_design(x_best, orthogonal_dimension)
        
        lb = self.lb
        ub = self.ub
        
        levels = np.linspace(-0.1, 0.1, orthogonal_dimension + 1)  # Adjust levels as needed
        
        trials = np.zeros((orthogonal_dimension + 1, self.dim))
        for i in range(orthogonal_dimension + 1):
            trials[i, :] = x_best.copy()
            for j in range(orthogonal_dimension):
                trials[i, j] = x_best[j] + levels[i] * (ub - lb)/2 # scale the levels
                trials[i, j] = np.clip(trials[i, j], lb, ub)

        fitness_values = np.array([func(trial) for trial in trials])
        
        self.eval_counts[0] += orthogonal_dimension + 1
        
        best_idx = np.argmin(fitness_values)
        
        if fitness_values[best_idx] < self.f_opt:
            self.f_opt = fitness_values[best_idx]
            self.x_opt = trials[best_idx]
            
        return self.f_opt, self.x_opt

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.initialize_subpopulations(func)

        while sum(self.eval_counts) < self.budget:
            for i in range(self.num_subpopulations):
                for j in range(self.popsize[i]):
                    # Mutation
                    idxs = np.random.choice(self.popsize[i], 3, replace=False)
                    x1, x2, x3 = self.subpopulations[i][idxs]
                    mutant = x1 + self.F * (x2 - x3)
                    mutant = np.clip(mutant, self.lb, self.ub)

                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover_mask, mutant, self.subpopulations[i][j])

                    # Evaluation
                    f_trial = func(trial)
                    self.eval_counts[i] += 1

                    # Selection
                    if f_trial < self.fitness[i][j]:
                        self.subpopulations[i][j] = trial
                        self.fitness[i][j] = f_trial
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial

                # Cooperation: Share best solution
                best_idx = np.argmin(self.fitness[i])
                best_solution = self.subpopulations[i][best_idx]
                for k in range(self.num_subpopulations):
                    if k != i:
                        worst_idx = np.argmax(self.fitness[k])
                        if self.fitness[i][best_idx] < self.fitness[k][worst_idx]:
                            self.subpopulations[k][worst_idx] = best_solution
                            self.fitness[k][worst_idx] = self.fitness[i][best_idx]
                            if self.fitness[i][best_idx] < self.f_opt:
                                self.f_opt = self.fitness[i][best_idx]
                                self.x_opt = best_solution
                
                # Orthogonal Learning (applied to best solution of each subpopulation)
                best_idx = np.argmin(self.fitness[i])
                self.orthogonal_learning(func, self.subpopulations[i][best_idx])

                # Dynamic Population Size
                if self.dynamic_popsize:
                    if np.std(self.fitness[i]) < 1e-6:  # Stagnation detection
                        self.popsize[i] = int(self.popsize[i] * 0.9)
                        if self.popsize[i] < 5:
                             self.popsize[i] = 5
                    else:
                        self.popsize[i] = int(self.popsize[i] * 1.1)
                        if self.popsize[i] > 20 * self.dim:
                             self.popsize[i] = 20 * self.dim

                    # Resize population
                    old_pop = self.subpopulations[i]
                    old_fitness = self.fitness[i]
                    self.subpopulations[i] = np.random.uniform(self.lb, self.ub, size=(self.popsize[i], self.dim))
                    self.fitness[i] = np.array([func(x) for x in self.subpopulations[i]])
                    self.eval_counts[i] += self.popsize[i]

                    # Keep the best individuals from previous population
                    num_keep = min(self.popsize[i], len(old_pop))
                    best_indices = np.argsort(old_fitness)[:num_keep]
                    self.subpopulations[i][:num_keep] = old_pop[best_indices]
                    self.fitness[i][:num_keep] = old_fitness[best_indices]
                    
                    best_idx = np.argmin(self.fitness[i])
                    if self.fitness[i][best_idx] < self.f_opt:
                        self.f_opt = self.fitness[i][best_idx]
                        self.x_opt = self.subpopulations[i][best_idx]

        return self.f_opt, self.x_opt