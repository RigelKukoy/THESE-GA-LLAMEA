import numpy as np

class ToroidalDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.learning_rate = learning_rate
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.F_history = []
        self.CR_history = []
        self.automaton_states = [0, 1]  # Two states for F and CR adaptation
        self.F_probabilities = [0.5, 0.5]  # Initial probabilities for F choices
        self.CR_probabilities = [0.5, 0.5]  # Initial probabilities for CR choices
        self.F_choices = [0.3, 0.7]  # Two F values to choose from
        self.CR_choices = [0.2, 0.8]  # Two CR values to choose from


    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()


    def toroidal_mutation(self, i):
        p = self.pop_size
        r1 = (i + 1) % p
        r2 = (i - 1 + p) % p
        r3 = (i + 2) % p

        return self.population[r1], self.population[r2], self.population[r3]

    def learning_automaton_update(self, is_successful, param_type):
        if param_type == 'F':
            probabilities = self.F_probabilities
            choices = self.F_choices
            state = self.automaton_states[0] # F has only one automaton state
        elif param_type == 'CR':
            probabilities = self.CR_probabilities
            choices = self.CR_choices
            state = self.automaton_states[0] # CR has only one automaton state
        else:
            return

        if is_successful:
            probabilities[state] += self.learning_rate * (1 - probabilities[state])
        else:
            probabilities[state] -= self.learning_rate * probabilities[state]

        # Normalize probabilities
        total = sum(probabilities)
        probabilities[0] /= total
        probabilities[1] /= total

        if param_type == 'F':
            self.F_probabilities = probabilities
        elif param_type == 'CR':
            self.CR_probabilities = probabilities


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Select F and CR using learning automaton
                F_current = np.random.choice(self.F_choices, p=self.F_probabilities)
                CR_current = np.random.choice(self.CR_choices, p=self.CR_probabilities)

                # Mutation using toroidal topology
                x_r1, x_r2, x_r3 = self.toroidal_mutation(i)
                v = self.population[i] + F_current * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < CR_current or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    # Success
                    self.learning_automaton_update(True, 'F')
                    self.learning_automaton_update(True, 'CR')
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                else:
                    # Failure
                    self.learning_automaton_update(False, 'F')
                    self.learning_automaton_update(False, 'CR')

        return self.f_opt, self.x_opt