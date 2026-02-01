import numpy as np

class AdaptiveDETournament:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.9, F=0.5, lr_F=0.1, lr_CR=0.1, tournament_size=3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.F = F
        self.lr_F = lr_F  # Learning rate for F adaptation
        self.lr_CR = lr_CR  # Learning rate for CR adaptation
        self.tournament_size = tournament_size
        self.population = None
        self.fitness = None
        self.best_fitness_history = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness_history.append(np.min(self.fitness))
        self.budget -= self.pop_size

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        return x_r1 + self.F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i, cr):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= cr or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def tournament_selection(self):
        """Selects an individual using tournament selection."""
        indices = np.random.choice(self.pop_size, size=self.tournament_size, replace=False)
        tournament_fitnesses = self.fitness[indices]
        winner_index = indices[np.argmin(tournament_fitnesses)]
        return self.population[winner_index], self.fitness[winner_index]

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i])

                # Crossover
                u_i = self.crossover(self.population[i], v_i, self.CR)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                #Tournament selection
                winner, winner_fitness = self.tournament_selection()
                
                if f_u_i < self.fitness[i]:
                  new_population[i] = u_i
                  new_fitness[i] = f_u_i
                  # Adapt F and CR based on success
                  self.F = self.F + self.lr_F * (1 - self.F)
                  self.CR = self.CR + self.lr_CR * (1 - self.CR)
                else:
                  #Adapt F and CR based on failure
                  self.F = self.F - self.lr_F * (self.F)
                  self.CR = self.CR - self.lr_CR * (self.CR)
                  
                if f_u_i < self.f_opt:
                    self.f_opt = f_u_i
                    self.x_opt = u_i

                self.F = np.clip(self.F, 0.1, 0.9)
                self.CR = np.clip(self.CR, 0.1, 0.9)


            self.population = new_population
            self.fitness = new_fitness
            self.best_fitness_history.append(np.min(self.fitness))

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt