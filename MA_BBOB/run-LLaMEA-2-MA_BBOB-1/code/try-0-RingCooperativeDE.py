import numpy as np

class RingCooperativeDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_subpops=5, F=0.5, CR=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_subpops = num_subpops
        self.F = F
        self.CR = CR
        self.subpop_dims = [list(range(i * (dim // num_subpops), (i + 1) * (dim // num_subpops))) for i in range(num_subpops)]
        remaining_dims = dim % num_subpops
        for i in range(remaining_dims):
            self.subpop_dims[i].append(num_subpops * (dim // num_subpops) + i)
        self.subpops = [np.random.rand(pop_size, len(dims)) for dims in self.subpop_dims]
        self.fitness = [np.zeros(pop_size) for _ in range(num_subpops)]
        self.f_opt = np.inf
        self.x_opt = None
        self.lb = None
        self.ub = None

    def evaluate(self, func, subpop_index):
        for i in range(self.pop_size):
            x = np.zeros(self.dim)
            start = 0
            for j in range(self.num_subpops):
                if j == subpop_index:
                    x[np.array(self.subpop_dims[j])] = self.lb[np.array(self.subpop_dims[j])] + self.subpops[j][i] * (self.ub[np.array(self.subpop_dims[j])] - self.lb[np.array(self.subpop_dims[j])])
                else:
                    best_idx = np.argmin(self.fitness[j])
                    x[np.array(self.subpop_dims[j])] = self.lb[np.array(self.subpop_dims[j])] + self.subpops[j][best_idx] * (self.ub[np.array(self.subpop_dims[j])] - self.lb[np.array(self.subpop_dims[j])])
            f = func(x)
            self.fitness[subpop_index][i] = f
            self.budget -= 1

            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()

    def evolve_subpop(self, func, subpop_index):
        dims = self.subpop_dims[subpop_index]
        lb_sub = self.lb[np.array(dims)]
        ub_sub = self.ub[np.array(dims)]
        
        for i in range(self.pop_size):
            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x1, x2, x3 = self.subpops[subpop_index][idxs]
            v = self.subpops[subpop_index][i] + self.F * (x2 - x3)
            v = np.clip(v, 0, 1)

            # Crossover
            j_rand = np.random.randint(len(dims))
            u = self.subpops[subpop_index][i].copy()
            for j in range(len(dims)):
                if np.random.rand() < self.CR or j == j_rand:
                    u[j] = v[j]

            # Create full vector and evaluate fitness
            x_trial = np.zeros(self.dim)
            start = 0
            for k in range(self.num_subpops):
                if k == subpop_index:
                    x_trial[np.array(self.subpop_dims[k])] = lb_sub + u * (ub_sub - lb_sub)
                else:
                     best_idx = np.argmin(self.fitness[k])
                     x_trial[np.array(self.subpop_dims[k])] = self.lb[np.array(self.subpop_dims[k])] + self.subpops[k][best_idx] * (self.ub[np.array(self.subpop_dims[k])] - self.lb[np.array(self.subpop_dims[k])])
           
            f_trial = func(x_trial)
            self.budget -= 1

            if f_trial < self.fitness[subpop_index][i]:
                self.subpops[subpop_index][i] = u
                self.fitness[subpop_index][i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial.copy()


    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        
        # Initialize sub-populations
        for i in range(self.num_subpops):
            lb_sub = self.lb[np.array(self.subpop_dims[i])]
            ub_sub = self.ub[np.array(self.subpop_dims[i])]
            self.subpops[i] = np.random.rand(self.pop_size, len(self.subpop_dims[i]))
            
        for i in range(self.num_subpops):
            self.evaluate(func, i)
        
        while self.budget > 0:
            for i in range(self.num_subpops):
                self.evolve_subpop(func, i)

        return self.f_opt, self.x_opt