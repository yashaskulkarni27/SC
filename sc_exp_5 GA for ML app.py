
import random


def fitness_function(x):
 return x**2 + 5*x + 5


population_size = 100
mutation_rate = 0.1
num_generations = 50


def initialize_population(size):
    return [random.uniform(-10, 10) for _ in range(size)]



def select_parents(population, num_parents):
    fitness_scores = [1 / (fitness_function(x) + 1e-6) for x in population]
    total_fitness = sum(fitness_scores)
    selected_parents = []
    for _ in range(num_parents):
        rand_value = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        for i, fitness in enumerate(fitness_scores):
            cumulative_fitness += fitness
            if cumulative_fitness > rand_value:
                selected_parents.append(population[i])
                break
    return selected_parents


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    ind = 0
    for i in range(len(individual)):
        if random.random() < mutation_rate and individual[i] != "." and individual[i] != "-":
            ind = float(individual[i]) + random.uniform(-0.1, 0.1)
    return ind

population = initialize_population(population_size)
for generation in range(num_generations):
    parents = select_parents(population, population_size // 2)
    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child1, child2 = crossover(str(parent1), str(parent2))
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        offspring.extend([child1, child2])
    population = offspring

best_solution = min(population, key=fitness_function)
best_fitness = fitness_function(best_solution)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
