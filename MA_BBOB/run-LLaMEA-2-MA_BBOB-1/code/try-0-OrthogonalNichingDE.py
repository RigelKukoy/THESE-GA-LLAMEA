import numpy as np

class OrthogonalNichingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, num_niches=5, niche_radius=0.5, F_adaptive=True, CR_adaptive=True):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.num_niches = num_niches
        self.niche_radius = niche_radius
        self.F_adaptive = F_adaptive
        self.CR_adaptive = CR_adaptive
        self.population = None
        self.fitness = None
        self.niches = None
        self.f_opt = np.inf
        self.x_opt = None
        self.success_F = []
        self.success_CR = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.niches = self.initialize_niches(func)
        self.update_niches(func)
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)].copy()
        self.success_F = []
        self.success_CR = []

    def initialize_niches(self, func):
        niches = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_niches, self.dim))
        return niches
    
    def assign_to_niches(self):
        assignments = np.argmin(np.linalg.norm(self.population[:, np.newaxis, :] - self.niches[np.newaxis, :, :], axis=2), axis=1)
        return assignments

    def update_niches(self, func):
        assignments = self.assign_to_niches()
        for i in range(self.num_niches):
            members = self.population[assignments == i]
            if len(members) > 0:
                self.niches[i] = np.mean(members, axis=0)

    def orthogonal_learning(self, x):
        # Generate an orthogonal array
        oa = self.generate_orthogonal_array(self.dim)
        
        # Sample points based on the orthogonal array
        samples = []
        for row in oa:
            new_x = x.copy()
            for i, val in enumerate(row):
                if val == 1:
                    new_x[i] = np.random.uniform(x[i], 5.0) # Explore upper bound
                else:
                    new_x[i] = np.random.uniform(-5.0, x[i]) # Explore lower bound
            samples.append(new_x)
        return samples

    def generate_orthogonal_array(self, dim):
        # A simple example: a 2-level orthogonal array
        oa = np.zeros((dim + 1, dim))
        for i in range(dim):
            oa[0, i] = 0
            oa[i + 1, i] = 1
        return oa

    def mutation(self, func, i, assignment):
        members = self.population[self.assign_to_niches() == assignment]
        if len(members) < 3:
            indices = np.random.choice(self.pop_size, 3, replace=True)
            x_r1, x_r2, x_r3 = self.population[indices]
        else:
             indices = np.random.choice(len(members), 3, replace=False)
             x_r1, x_r2, x_r3 = members[indices]
        
        v = self.population[i] + self.F * (x_r1 - x_r2)
        v = np.clip(v, func.bounds.lb, func.bounds.ub)
        return v

    def crossover(self, v, i):
        j_rand = np.random.randint(self.dim)
        u = self.population[i].copy()
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                u[j] = v[j]
        return u

    def update_parameters(self, success):
        if self.F_adaptive:
            if success and self.success_F:
                self.F = 0.9 * self.F + 0.1 * np.mean(self.success_F)
            else:
                self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
            self.F = np.clip(self.F, 0.1, 1.0)
        if self.CR_adaptive:
            if success and self.success_CR:
                self.CR = 0.9 * self.CR + 0.1 * np.mean(self.success_CR)
            else:
                self.CR = np.clip(np.random.normal(0.7, 0.3), 0.1, 1.0)
            self.CR = np.clip(self.CR, 0.1, 1.0)

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            assignments = self.assign_to_niches()
            for i in range(self.pop_size):
                assignment = assignments[i]

                # Mutation and Crossover
                v = self.mutation(func, i, assignment)
                u = self.crossover(v, i)

                # Orthogonal Learning
                ol_samples = self.orthogonal_learning(u)
                
                # Evaluate Orthogonal Learning Samples, use budget efficiently
                ol_fitnesses = []
                num_evals = min(len(ol_samples), self.budget)
                for k in range(num_evals):
                  f_ol = func(ol_samples[k])
                  ol_fitnesses.append(f_ol)
                  self.budget -=1
                
                # if no budget left, break
                if self.budget <= 0:
                    break

                # Best fitness from OL samples
                if ol_fitnesses: # to avoid errors if empty
                  best_ol_index = np.argmin(ol_fitnesses)
                  f_u = ol_fitnesses[best_ol_index]
                  u = ol_samples[best_ol_index]
                else: # to ensure f_u gets assigned a value, use function evaluation only if budget is left
                  f_u = func(u)
                  self.budget -=1 #ensure this is done only when needed

                # Selection
                success = False
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.success_F.append(self.F)
                        self.success_CR.append(self.CR)
                    success = True
                
                # Adapt parameters
                self.update_parameters(success)
            
            if self.budget > 0: #update only if enough budget
                self.update_niches(func)

        return self.f_opt, self.x_opt