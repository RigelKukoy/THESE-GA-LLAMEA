import numpy as np

class AdaptiveDELearning:
    def __init__(self, budget=10000, dim=10, pop_multiplier=5, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size_initial = dim * pop_multiplier
        self.pop_size = self.pop_size_initial
        self.population = None
        self.fitness = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9 # Crossover rate
        self.learning_rate = learning_rate
        self.archive = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            self.adapt_population_size()
            for i in range(self.pop_size):
                # Mutation with learning
                mutant = self.mutation(i, func)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < self.fitness[i]:
                    self.archive.append(self.population[i].copy())
                    self.fitness[i] = f_trial
                    self.population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    self.archive.append(trial.copy())


                if self.budget <=0:
                    break

            self.adapt_parameters()

        return self.f_opt, self.x_opt

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

    def mutation(self, i, func):
        idxs = [idx for idx in range(self.pop_size) if idx != i]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        
        if self.archive:
            donor = self.archive[np.random.choice(len(self.archive))]
            mutant = np.clip(a + self.F * (b - c + donor - self.population[i]), func.bounds.lb, func.bounds.ub) #Using archive as learning
        else:
             mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

        return mutant

    def adapt_parameters(self):
         # Adapt F and CR based on success in previous generation
        successful_F = []
        successful_CR = []
        for i in range(self.pop_size):
            if len(self.archive) > 0:
                if np.random.rand() < 0.1:
                    self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
                    self.CR = np.clip(np.random.normal(0.9, 0.1), 0.1, 1.0)

    def adapt_population_size(self):
        if self.budget > 0:
            success_rate = sum(1 for i in range(self.pop_size) if self.archive and self.fitness[i] < func(self.population[i])) / self.pop_size if self.pop_size > 0 else 0
            if success_rate > 0.2 and self.pop_size < 2*self.pop_size_initial:
                self.pop_size = min(2*self.pop_size_initial, self.pop_size + 1)
                self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population]) # re-evaluate fitness with larger population
            elif success_rate < 0.1 and self.pop_size > self.dim + 2:
                self.pop_size = max(self.dim + 2, self.pop_size - 1)
                self.population = self.population[:self.pop_size]
                self.fitness = self.fitness[:self.pop_size]


if __name__ == '__main__':
    # Example Usage (replace with your actual function)
    class DummyFunction:
        def __init__(self, dim):
            self.dim = dim
            self.bounds = Bounds(-5, 5)

        def __call__(self, x):
            return np.sum(x**2)  # Example function

    class Bounds:
        def __init__(self, lb, ub):
            self.lb = np.array([lb])
            self.ub = np.array([ub])

    dim = 10
    budget = 10000
    func = DummyFunction(dim)
    optimizer = AdaptiveDELearning(budget=budget, dim=dim)
    f_opt, x_opt = optimizer(func)
    print(f"Optimal value: {f_opt}")
    print(f"Optimal solution: {x_opt}")