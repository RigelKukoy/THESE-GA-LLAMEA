import numpy as np

class SymbioticOptimization:
    def __init__(self, budget=10000, dim=10, host_population_size=20, symbiont_population_size=10, mutualism_factor=0.5, parasitism_factor=0.1):
        self.budget = budget
        self.dim = dim
        self.host_population_size = host_population_size
        self.symbiont_population_size = symbiont_population_size
        self.mutualism_factor = mutualism_factor
        self.parasitism_factor = parasitism_factor

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize host population
        hosts = np.random.uniform(lb, ub, size=(self.host_population_size, self.dim))
        host_fitness = np.array([func(x) for x in hosts])
        self.budget -= self.host_population_size

        # Initialize symbiont population (associated with each host)
        symbionts = np.random.uniform(lb, ub, size=(self.host_population_size, self.symbiont_population_size, self.dim))
        symbiont_fitness = np.zeros((self.host_population_size, self.symbiont_population_size))
        for i in range(self.host_population_size):
            for j in range(self.symbiont_population_size):
                symbiont_fitness[i, j] = func(symbionts[i, j])
                self.budget -= 1
                if self.budget <= 0:
                    break
            if self.budget <= 0:
                break

        best_host_index = np.argmin(host_fitness)
        self.f_opt = host_fitness[best_host_index]
        self.x_opt = hosts[best_host_index]

        while self.budget > 0:
            for i in range(self.host_population_size):
                # Mutualism: Host and symbiont benefit each other
                for j in range(self.symbiont_population_size):
                    if self.budget <= 0:
                        break

                    # Host learns from symbiont
                    new_host = hosts[i] + self.mutualism_factor * (symbionts[i, j] - hosts[i]) * np.random.uniform(-1, 1, self.dim)
                    new_host = np.clip(new_host, lb, ub)
                    new_host_fitness = func(new_host)
                    self.budget -= 1
                    if new_host_fitness < host_fitness[i]:
                        host_fitness[i] = new_host_fitness
                        hosts[i] = new_host

                    # Symbiont learns from host
                    new_symbiont = symbionts[i, j] + self.mutualism_factor * (hosts[i] - symbionts[i, j]) * np.random.uniform(-1, 1, self.dim)
                    new_symbiont = np.clip(new_symbiont, lb, ub)
                    new_symbiont_fitness = func(new_symbiont)
                    self.budget -= 1
                    if new_symbiont_fitness < symbiont_fitness[i, j]:
                        symbiont_fitness[i, j] = new_symbiont_fitness
                        symbionts[i, j] = new_symbiont

                if self.budget <= 0:
                    break


                # Parasitism: Host is harmed, symbiont benefits (or vice versa)
                if np.random.rand() < self.parasitism_factor and self.budget > 0:
                    target_host_index = np.random.randint(0, self.host_population_size)
                    #Replace host with a slightly mutated version of the current host's best symbiont
                    best_symbiont_index = np.argmin(symbiont_fitness[i])
                    parasite = symbionts[i, best_symbiont_index] + np.random.normal(0, 0.05, self.dim)
                    parasite = np.clip(parasite, lb, ub)
                    parasite_fitness = func(parasite)
                    self.budget -= 1

                    if parasite_fitness < host_fitness[target_host_index]:
                        host_fitness[target_host_index] = parasite_fitness
                        hosts[target_host_index] = parasite

                # Update global best
                current_best_index = np.argmin(host_fitness)
                if host_fitness[current_best_index] < self.f_opt:
                    self.f_opt = host_fitness[current_best_index]
                    self.x_opt = hosts[current_best_index]

        return self.f_opt, self.x_opt