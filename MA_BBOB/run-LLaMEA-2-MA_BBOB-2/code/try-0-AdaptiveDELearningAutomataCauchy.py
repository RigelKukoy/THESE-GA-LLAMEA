import numpy as np

class AdaptiveDELearningAutomataCauchy:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, learning_rate=0.1, initial_exploration_probability=0.5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.learning_rate = learning_rate
        self.initial_exploration_probability = initial_exploration_probability
        self.mutation_probabilities = np.array([self.initial_exploration_probability, 1 - self.initial_exploration_probability]) # Probabilities for Cauchy and rand/1 mutation

    def cauchy_mutation(self, x, scale):
        mutation = scale * np.random.standard_cauchy(size=self.dim)
        return x + mutation

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Learning Automata to select mutation strategy
                mutation_choice = np.random.choice([0, 1], p=self.mutation_probabilities) # 0: Cauchy, 1: rand/1

                if mutation_choice == 0:
                    # Cauchy Mutation
                    mutant = self.cauchy_mutation(self.population[i], scale=self.F * (ub - lb))
                    mutant = np.clip(mutant, lb, ub)
                else:
                    # rand/1 Mutation
                    idxs = np.random.choice(self.popsize, 3, replace=False)
                    x1, x2, x3 = self.population[idxs]
                    mutant = x1 + self.F * (x2 - x3)
                    mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    # Reward the selected mutation operator
                    if mutation_choice == 0:
                        self.mutation_probabilities[0] += self.learning_rate * (1 - self.mutation_probabilities[0])
                        self.mutation_probabilities[1] -= self.learning_rate * self.mutation_probabilities[1]
                    else:
                        self.mutation_probabilities[1] += self.learning_rate * (1 - self.mutation_probabilities[1])
                        self.mutation_probabilities[0] -= self.learning_rate * self.mutation_probabilities[0]
                    
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    # Punish the selected mutation operator
                    if mutation_choice == 0:
                        self.mutation_probabilities[0] -= self.learning_rate * self.mutation_probabilities[0]
                        self.mutation_probabilities[1] += self.learning_rate * (1 - self.mutation_probabilities[1])
                    else:
                        self.mutation_probabilities[1] -= self.learning_rate * self.mutation_probabilities[1]
                        self.mutation_probabilities[0] += self.learning_rate * (1 - self.mutation_probabilities[0])

                # Normalize probabilities
                self.mutation_probabilities /= np.sum(self.mutation_probabilities)

        return self.f_opt, self.x_opt