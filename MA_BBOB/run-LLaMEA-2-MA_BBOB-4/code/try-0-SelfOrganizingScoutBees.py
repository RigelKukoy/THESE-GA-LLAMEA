import numpy as np

class SelfOrganizingScoutBees:
    def __init__(self, budget=10000, dim=10, num_bees=50, scout_ratio=0.1, initial_radius=1.0, radius_decay=0.95, step_size=0.1, step_size_decay=0.98, stagnation_threshold=50):
        self.budget = budget
        self.dim = dim
        self.num_bees = num_bees
        self.scout_ratio = scout_ratio
        self.initial_radius = initial_radius
        self.radius_decay = radius_decay
        self.step_size = step_size
        self.step_size_decay = step_size_decay
        self.stagnation_threshold = stagnation_threshold
        self.best_fitness_history = []
        self.last_improvement = 0
        self.radius = initial_radius
        self.num_scouts = int(self.num_bees * self.scout_ratio)

    def __call__(self, func):
        # Initialization
        bees = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_bees, self.dim))
        fitness = np.array([func(x) for x in bees])
        self.budget -= self.num_bees
        
        self.f_opt = np.min(fitness)
        self.x_opt = bees[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.num_bees:
            # Employed Bees Phase
            for i in range(self.num_bees):
                neighbor_index = np.random.choice([j for j in range(self.num_bees) if j != i])
                
                new_bee = bees[i] + self.step_size * (bees[i] - bees[neighbor_index])
                new_bee = np.clip(new_bee, func.bounds.lb, func.bounds.ub)
                
                new_fitness = func(new_bee)
                self.budget -= 1
                
                if new_fitness < fitness[i]:
                    bees[i] = new_bee
                    fitness[i] = new_fitness
                    
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = new_bee
                        self.last_improvement = generation

            # Scout Bees Phase
            worst_indices = np.argsort(fitness)[-self.num_scouts:]  # Replace worst bees with scouts

            for i in worst_indices:
                # Conduct a broader search within a radius
                new_bee = np.random.uniform(
                    np.maximum(func.bounds.lb, self.x_opt - self.radius),
                    np.minimum(func.bounds.ub, self.x_opt + self.radius),
                    size=self.dim
                )
                new_fitness = func(new_bee)
                self.budget -= 1
                
                if new_fitness < fitness[i]:
                    bees[i] = new_bee
                    fitness[i] = new_fitness
                    
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = new_bee
                        self.last_improvement = generation
            
            self.best_fitness_history.append(self.f_opt)
            
            # Stagnation Check and Adaptation
            if (generation - self.last_improvement) > self.stagnation_threshold:
                # Reduce search radius and step size if stagnating
                self.radius *= self.radius_decay
                self.step_size *= self.step_size_decay

                # Increase scouts to promote exploration if stagnating
                self.num_scouts = min(int(self.num_bees * 0.5), self.num_scouts + 1)  #Increase scout ratio gradually
                
                #Reset the worst scout bee indexes to the updated number of scout bees.
                worst_indices = np.argsort(fitness)[-self.num_scouts:] #Recalculate the worst bees with updated num_scouts

                #Explore around the best bee.
                for i in worst_indices:
                    new_bee = np.random.uniform(
                        np.maximum(func.bounds.lb, self.x_opt - self.radius),
                        np.minimum(func.bounds.ub, self.x_opt + self.radius),
                        size=self.dim
                    )
                    new_fitness = func(new_bee)
                    self.budget -= 1
                    
                    if new_fitness < fitness[i]:
                        bees[i] = new_bee
                        fitness[i] = new_fitness
                        
                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_bee
                            self.last_improvement = generation
            else:
                # Decrease scout ratio for exploitation
                 self.num_scouts = max(int(self.num_bees * 0.05), int(self.num_scouts * 0.95))

            generation += 1

        return self.f_opt, self.x_opt