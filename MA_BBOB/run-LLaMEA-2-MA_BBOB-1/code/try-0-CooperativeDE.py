import numpy as np

class CooperativeDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_sub_pops=5, F=0.5, CR=0.7, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_sub_pops = num_sub_pops
        self.F = F
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.sub_pop_size = pop_size // num_sub_pops
        self.populations = [None] * num_sub_pops
        self.fitnesses = [None] * num_sub_pops
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.cooperation_rates = np.ones((num_sub_pops, num_sub_pops)) / num_sub_pops # Initialize cooperation rates

    def initialize_sub_population(self, func, sub_pop_index):
        self.populations[sub_pop_index] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.sub_pop_size, self.dim))
        self.fitnesses[sub_pop_index] = np.array([func(x) for x in self.populations[sub_pop_index]])
        self.budget -= self.sub_pop_size

    def local_search(self, x, func):
        """Performs a local search around x."""
        x_new = x.copy()
        for i in range(self.dim):
            # Perturb each dimension with a small random value
            x_new[i] = x[i] + np.random.normal(0, 0.05 * (func.bounds.ub[i] - func.bounds.lb[i]))
            x_new[i] = np.clip(x_new[i], func.bounds.lb[i], func.bounds.ub[i])

        f_new = func(x_new)
        self.budget -= 1
        return x_new, f_new
        
    def update_cooperation_rates(self):
        """Updates cooperation rates based on fitness improvements."""
        for i in range(self.num_sub_pops):
            for j in range(self.num_sub_pops):
                if i != j:
                    # If sub-pop i is doing well and sub-pop j is not, increase cooperation
                    if np.min(self.fitnesses[i]) < np.mean(self.fitnesses[i]) and np.min(self.fitnesses[j]) > np.mean(self.fitnesses[j]):
                        self.cooperation_rates[i, j] *= 1.05  # Increase cooperation rate
                    else:
                        self.cooperation_rates[i, j] *= 0.95  # Decrease cooperation rate
                    
                    self.cooperation_rates[i, j] = np.clip(self.cooperation_rates[i, j], 0.01, 0.5) # Keep rates in a reasonable range
        # Normalize the cooperation rates
        for i in range(self.num_sub_pops):
            self.cooperation_rates[i] /= np.sum(self.cooperation_rates[i])


    def __call__(self, func):
        for i in range(self.num_sub_pops):
            self.initialize_sub_population(func, i)

        while self.budget > 0:
            self.update_cooperation_rates()

            for i in range(self.num_sub_pops):
                for j in range(self.sub_pop_size):
                    # Choose a sub-population to cooperate with
                    coop_partner_index = np.random.choice(self.num_sub_pops, p=self.cooperation_rates[i])
                    if coop_partner_index == i:
                        coop_partner_index = (i + 1) % self.num_sub_pops

                    # Differential Evolution within the sub-population with cooperation
                    indices = np.random.choice(self.sub_pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.populations[i][indices]
                    
                    # Cooperative component
                    donor_index = np.random.randint(self.sub_pop_size)
                    x_coop = self.populations[coop_partner_index][donor_index]

                    v = self.populations[i][j] + self.F * (x_coop - self.populations[i][j]) + self.F * (x_r1 - x_r2)
                    v = np.clip(v, func.bounds.lb, func.bounds.ub)

                    # Crossover
                    j_rand = np.random.randint(self.dim)
                    u = self.populations[i][j].copy()
                    for k in range(self.dim):
                        if np.random.rand() < self.CR or k == j_rand:
                            u[k] = v[k]

                    # Local search with some probability
                    if np.random.rand() < self.local_search_prob and self.budget > 1:
                        u, f_u = self.local_search(u, func)
                    else:
                        f_u = func(u)
                        self.budget -= 1

                    # Selection
                    if f_u < self.fitnesses[i][j]:
                        self.fitnesses[i][j] = f_u
                        self.populations[i][j] = u
                        if f_u < self.f_opt:
                            self.f_opt = f_u
                            self.x_opt = u.copy()

        return self.f_opt, self.x_opt