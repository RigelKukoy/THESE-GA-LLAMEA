import numpy as np

class SelfOrganizingSpeciationDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_niches=5, initial_niche_radius=0.5, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_niches = num_niches
        self.initial_niche_radius = initial_niche_radius
        self.adaptation_rate = adaptation_rate
        self.population = None
        self.fitness = None
        self.niche_assignments = None
        self.niche_radii = None
        self.f_opt = np.inf
        self.x_opt = None
        self.mutation_strategies = [self.mutation_DE_rand1, self.mutation_DE_best1]
        self.num_strategies = len(self.mutation_strategies)
        self.strategy_successes = np.ones((self.num_niches, self.num_strategies)) #Niche-specific strategy successes
        self.lb = None
        self.ub = None

    def initialize_population(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.niche_assignments = np.random.choice(self.num_niches, size=self.pop_size)
        self.niche_radii = np.full(self.num_niches, self.initial_niche_radius)

        best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_index]
        self.x_opt = self.population[best_index].copy()

    def mutation_DE_rand1(self, population, i, F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices]
        return population[i] + F * (x_r2 - x_r3)

    def mutation_DE_best1(self, population, i, F):
        best_index = np.argmin(self.fitness)
        indices = np.random.choice(self.pop_size, 2, replace=False)
        x_r1, x_r2 = population[indices]
        return population[i] + F * (self.population[best_index] - population[i]) + F * (x_r1 - x_r2)


    def assign_to_niche(self, individual):
        distances = np.linalg.norm(self.population - individual, axis=1)
        niche_fitnesses = np.zeros(self.num_niches)
        for niche in range(self.num_niches):
            niche_members = self.population[self.niche_assignments == niche]
            if len(niche_members) > 0:
                niche_fitnesses[niche] = np.mean(self.fitness[self.niche_assignments == niche])
            else:
                niche_fitnesses[niche] = np.inf  # Penalize empty niches

        # Assign to the niche with the closest member and the best fitness
        best_niche = np.argmin(niche_fitnesses)
        return best_niche


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Niche-specific strategy selection
                niche = self.niche_assignments[i]
                strategy_probabilities = self.strategy_successes[niche]
                strategy_probabilities /= np.sum(strategy_probabilities)
                strategy_index = np.random.choice(self.num_strategies, p=strategy_probabilities)
                mutation_strategy = self.mutation_strategies[strategy_index]

                # Mutation
                F = np.random.uniform(0.2, 0.8)
                v = mutation_strategy(self.population, i, F)
                v = np.clip(v, self.lb, self.ub)

                # Crossover
                CR = np.random.uniform(0.3, 0.9)
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection and Niche Adjustment
                if f_u < self.fitness[i]:
                    self.strategy_successes[niche, strategy_index] += 1
                    self.fitness[i] = f_u
                    self.population[i] = u

                    # Re-assign niche
                    new_niche = self.assign_to_niche(u)
                    self.niche_assignments[i] = new_niche
                    
                    # Adapt niche radius (simplified - can be enhanced)
                    if np.random.rand() < self.adaptation_rate:
                        self.niche_radii[niche] *= np.random.uniform(0.9, 1.1)
                        self.niche_radii[niche] = np.clip(self.niche_radii[niche], 0.1, 1.0)

                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                else:
                     self.strategy_successes[niche, strategy_index] *= 0.9 # Reduce success if the strategy didn't improve


        return self.f_opt, self.x_opt