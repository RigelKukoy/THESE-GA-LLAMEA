import numpy as np

class BanditDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_arms=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_arms = num_arms
        self.arms = []
        for _ in range(num_arms):
            self.arms.append({
                'F': np.random.uniform(0.3, 0.8),
                'CR': np.random.uniform(0.3, 0.8),
                'wins': 0,
                'trials': 0
            })
        self.epsilon = 0.1  # Exploration rate
        self.archive = []
        self.archive_size = int(pop_size * 0.2)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            new_population = np.copy(self.population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Bandit selection
                if np.random.rand() < self.epsilon:
                    arm_index = np.random.randint(self.num_arms)  # Explore
                else:
                    # Exploit: choose the arm with highest win rate
                    win_rates = [arm['wins'] / (arm['trials'] + 1e-6) for arm in self.arms]
                    arm_index = np.argmax(win_rates)

                arm = self.arms[arm_index]
                arm['trials'] += 1

                # Differential Evolution with selected arm parameters
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + arm['F'] * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover = np.random.uniform(size=self.dim) < arm['CR']
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial
                    arm['wins'] += 1

                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                    else:
                        idx_to_replace = np.random.randint(0, self.archive_size)
                        self.archive[idx_to_replace] = trial

                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = self.population[i]
                
            self.population = new_population
            fitness = new_fitness

        return self.f_opt, self.x_opt