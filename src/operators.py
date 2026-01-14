import numpy as np
from typing import Tuple
from copy import deepcopy

def tournament_selection(population: list, fitness_values: list, tournament_size: int, rng: np.random.Generator) -> list:
    """
    Select the best individual from a random tournament.
    
    Args:
        population: List of individuals in the population.
        fitness_values: List of fitness values for each individual.
        tournament_size: Number of individuals competing in the tournament.
        rng: Random number generator.
    
    Returns:
        Deep copy of the selected individual.
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = min(indices, key=lambda i: fitness_values[i])
    return deepcopy(population[best_idx])

def order_crossover(parent1: list, parent2: list, rng: np.random.Generator) -> list:
    """
    OX1 Crossover that preserves the relative order of cities.
    
    Args:
        parent1: First parent route.
        parent2: Second parent route.
        rng: Random number generator.
    
    Returns:
        Child route combining elements from both parents.
    """
    size = len(parent1)
    child = [-1] * size
    
    start, end = sorted(rng.choice(size, size=2, replace=False))
    
    # Copia segmento dal genitore 1
    child[start:end] = parent1[start:end]
    
    # Riempi dal genitore 2
    current_pos_p2 = 0
    for i in range(size):
        if i >= start and i < end:
            continue
            
        # Trova il prossimo elemento valido in P2 che non è già in Child
        while parent2[current_pos_p2] in child[start:end]:
            current_pos_p2 += 1
        
        child[i] = parent2[current_pos_p2]
        current_pos_p2 += 1
            
    return child

def inversion_mutation(individual: list, rng: np.random.Generator) -> list:
    """
    Invert a segment of the route (simulates 2-opt).
    
    Args:
        individual: Route to mutate.
        rng: Random number generator.
    
    Returns:
        Mutated route with inverted segment.
    """
    size = len(individual)
    if size < 2: return individual
    idx1, idx2 = sorted(rng.choice(size, size=2, replace=False))
    individual[idx1:idx2+1] = individual[idx1:idx2+1][::-1]
    return individual

def swap_mutation(individual: list, rng: np.random.Generator) -> list:
    """
    Swap two random cities in the route.
    
    Args:
        individual: Route to mutate.
        rng: Random number generator.
    
    Returns:
        Mutated route with two cities swapped.
    """
    size = len(individual)
    if size < 2: return individual
    idx1, idx2 = rng.choice(size, size=2, replace=False)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def scramble_mutation(individual: list, rng: np.random.Generator) -> list:
    """
    Randomly shuffle a segment of the route.
    
    Args:
        individual: Route to mutate.
        rng: Random number generator.
    
    Returns:
        Mutated route with shuffled segment.
    """
    size = len(individual)
    if size < 2: return individual
    idx1, idx2 = sorted(rng.choice(size, size=2, replace=False))
    segment = individual[idx1:idx2+1]
    rng.shuffle(segment)
    individual[idx1:idx2+1] = segment
    return individual

def insertion_mutation(individual: list, rng: np.random.Generator) -> list:
    """
    Remove a city and insert it at a random position.
    
    Args:
        individual: Route to mutate.
        rng: Random number generator.
    
    Returns:
        Mutated route with repositioned city.
    """
    size = len(individual)
    if size < 2: return individual
    idx = rng.integers(0, size)
    city = individual.pop(idx)
    new_pos = rng.integers(0, size)
    individual.insert(new_pos, city)
    return individual

def adaptive_mutation(route: list, rng: np.random.Generator, generation: int, max_generations: int) -> list:
    """
    Select mutation operator based on the generation progress

    Args:
        route: Current route (list of cities).
        rng: Random number generator.
        generation: Current generation number.
        max_generations: Total number of generations.
    
    Returns:
        Mutated route.
    """
    progress = generation / max_generations
    
    # Rimosso base_return_mutation e or_opt (complessi o inutili qui)
    if progress < 0.3:
        operators = [swap_mutation, insertion_mutation, scramble_mutation]
        weights = [0.4, 0.4, 0.2]
    elif progress < 0.7:
        # Inversion è forte (2-opt), usiamolo di più nel mezzo
        operators = [swap_mutation, inversion_mutation, insertion_mutation]
        weights = [0.2, 0.6, 0.2]
    else:
        # Raffinamento finale
        operators = [inversion_mutation, swap_mutation]
        weights = [0.8, 0.2]
    
    weights = np.array(weights) / sum(weights)
    op = rng.choice(operators, p=weights)
    return op(route, rng)