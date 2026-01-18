import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from copy import deepcopy
from tqdm import tqdm
import concurrent.futures
import functools

from .utils import (
    precompute_shortest_paths,
    evaluate_route,
    create_random_route,
    create_greedy_permutation # <--- Nuovo
)
from .operators import (
    tournament_selection,
    order_crossover,
    adaptive_mutation,
    inversion_mutation, # Usato per varianti greedy e local search
    swap_mutation
)

@dataclass
class GAConfig:
    population_size: int = 100
    generations: int = 1000
    crossover_prob: float = 0.85
    mutation_prob: float = 0.25
    elite_size: int = 2
    tournament_size: int = 5
    seed: int = 42
    verbose: bool = True
    update_interval: int = 10
    stagnation_threshold: int = 50
    restart_threshold: int = 150
    local_search_iterations: int = 50 # Ridotto leggermente

class GeneticAlgorithm:
    def __init__(self, graph, alpha: float, beta: float, config: GAConfig = None):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.config = config or GAConfig()
        self.num_cities = len(graph.nodes())
        self.rng = np.random.default_rng(self.config.seed)
        self.distances, self.beta_sums, self.paths = precompute_shortest_paths(graph, beta)


        
        self.best_route = None
        self.best_fitness = float('inf')
        self.best_path = None

        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def _evaluate(self, route: list) -> Tuple[float, list]:
        return evaluate_route(route, self.graph, self.distances, self.beta_sums, self.alpha, self.beta)
    
    def _evaluate_population_parallel(self, population: List[list], executor) -> Tuple[List[float], List[list]]:
        """Valuta la popolazione usando tutti i core disponibili."""
        # Creiamo una versione fissa della funzione evaluate che ha già grafo e matrici
        # così ai worker passiamo solo la rotta variabile
        eval_func = functools.partial(
            evaluate_route,
            graph=self.graph,
            distances=self.distances,
            beta_sums=self.beta_sums,
            alpha=self.alpha,
            beta=self.beta
        )
        
        # Mappiamo la funzione su tutta la popolazione in parallelo
        results = list(executor.map(eval_func, population))
        
        # Separiamo i risultati
        fitness_values = [r[0] for r in results]
        detailed_paths = [r[1] for r in results]
        return fitness_values, detailed_paths
    
    def _create_initial_population(self) -> List[list]:
        population = []
        # 1. Greedy Base
        greedy_base = create_greedy_permutation(self.num_cities, self.distances)
        population.append(greedy_base)
        
        # 2. Varianti Greedy
        num_greedy_variants = max(0, int(self.config.population_size * 0.10) - 1)
        for _ in range(num_greedy_variants):
            variant = list(greedy_base)
            variant = inversion_mutation(variant, self.rng)
            population.append(variant)
        
        # 3. Random
        while len(population) < self.config.population_size:
            route = create_random_route(self.num_cities, self.rng)
            population.append(route)
            
        return population
    
    def _evaluate_population(self, population: List[list]) -> Tuple[List[float], List[list]]:
        fitness_values = []
        detailed_paths = []
        for route in population:
            fitness, path = self._evaluate(route)
            fitness_values.append(fitness)
            detailed_paths.append(path)
        return fitness_values, detailed_paths
    
    def _local_search(self, route: list) -> list:
        """Semplice Hill Climbing usando inversion (2-opt)."""
        best = list(route)
        best_fit, _ = self._evaluate(best)
        
        # Proviamo N mosse casuali
        for _ in range(self.config.local_search_iterations):
            # Inversion è equivalente a 2-opt per permutazioni
            candidate = list(best)
            candidate = inversion_mutation(candidate, self.rng)
            
            fit, _ = self._evaluate(candidate)
            if fit < best_fit:
                best = candidate
                best_fit = fit
        return best
    
    def run(self) -> Tuple[list, float]:
        population = self._create_initial_population()
        fitness_values, detailed_paths = self._evaluate_population(population)
        
        best_idx = int(np.argmin(fitness_values))
        self.best_fitness = fitness_values[best_idx]
        self.best_route = deepcopy(population[best_idx])
        self.best_path = detailed_paths[best_idx]
        
        pbar = None
        if self.config.verbose:
            pbar = tqdm(total=self.config.generations, desc="GA", unit="gen")
        
        stagnation = 0
        
        for gen in range(self.config.generations):
            new_population = []
            sorted_idx = np.argsort(fitness_values)
            
            # Elitism
            for i in range(self.config.elite_size):
                new_population.append(deepcopy(population[sorted_idx[i]]))
            
            # Offspring
            while len(new_population) < self.config.population_size:
                p1 = tournament_selection(population, fitness_values, self.config.tournament_size, self.rng)
                p2 = tournament_selection(population, fitness_values, self.config.tournament_size, self.rng)
                
                if self.rng.random() < self.config.crossover_prob:
                    c1 = order_crossover(p1, p2, self.rng)
                    c2 = order_crossover(p2, p1, self.rng)
                else:
                    c1, c2 = deepcopy(p1), deepcopy(p2)
                
                if self.rng.random() < self.config.mutation_prob:
                    c1 = adaptive_mutation(c1, self.rng, gen, self.config.generations)
                if self.rng.random() < self.config.mutation_prob:
                    c2 = adaptive_mutation(c2, self.rng, gen, self.config.generations)
                
                # Niente Repair o Check Validità! Sono permutazioni.
                new_population.append(c1)
                if len(new_population) < self.config.population_size:
                    new_population.append(c2)
            
            population = new_population
            fitness_values, detailed_paths = self._evaluate_population(population)

            self.best_fitness_history.append(float(np.min(fitness_values)))
            self.avg_fitness_history.append(float(np.mean(fitness_values)))
            
            best_idx = int(np.argmin(fitness_values))
            if fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_route = deepcopy(population[best_idx])
                self.best_path = detailed_paths[best_idx]
                stagnation = 0
            else:
                stagnation += 1
            
            if pbar and (gen + 1) % self.config.update_interval == 0:
                pbar.update(self.config.update_interval)
                pbar.set_postfix({'best': f'{self.best_fitness:.2f}'})
                
            # Restart
            if stagnation >= self.config.restart_threshold:
                num_keep = self.config.elite_size
                new_pop = [deepcopy(population[sorted_idx[i]]) for i in range(num_keep)]
                while len(new_pop) < self.config.population_size:
                    new_pop.append(create_random_route(self.num_cities, self.rng))
                population = new_pop
                fitness_values, detailed_paths = self._evaluate_population(population)
                stagnation = 0

        if pbar: pbar.close()
        
        # Final Polish
        self.best_route = self._local_search(self.best_route)
        self.best_fitness, self.best_path = self._evaluate(self.best_route)

        history = {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history
        }
        
        return self.best_path, self.best_fitness, history

def solve(problem, config: GAConfig = None, verbose: bool = True) -> List[Tuple[int, float]]:
    # Configurazione automatica
    num_cities = len(problem.graph.nodes())
    if config is None:
        if num_cities <= 100:
            config = GAConfig(
                population_size=150, generations=1000, verbose=verbose
            )
        else:
            config = GAConfig(
                population_size=100, generations=500, verbose=verbose
            )
    
    ga = GeneticAlgorithm(problem.graph, problem.alpha, problem.beta, config)
    return ga.run()