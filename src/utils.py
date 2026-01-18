import networkx as nx
import numpy as np
import os
import csv
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

def precompute_shortest_paths(graph: nx.Graph, beta: float) -> Tuple[Dict, Dict, Dict]:
    """
    Precompute shortest paths and specific metrics for exact cost calculation.
    
    Args:
        graph: Problem graph
        beta: Problem beta parameter (needed to precompute d^beta sums)

    Returns:
        distances: Dict[start][end] -> Sum of distances (d1 + d2 + ...)
        beta_path_sums: Dict[start][end] -> Sum of distances^beta (d1^b + d2^b + ...)
        paths: Dict[start][end] -> List of nodes
    """
    distances = {}
    beta_path_sums = {}
    paths_dict = {}
    
    # Pre-calculate edge weights raised to beta for faster lookup
    # edge_beta_cache[(u, v)] = dist^beta
    edge_beta_cache = {}
    for u, v, data in graph.edges(data=True):
        d = data['dist']
        val = d ** beta
        edge_beta_cache[(u, v)] = val
        edge_beta_cache[(v, u)] = val

    for source in graph.nodes():
        # Compute shortest paths from source using Dijkstra
        dist_map, path_map = nx.single_source_dijkstra(graph, source, weight='dist')
        
        distances[source] = dist_map
        paths_dict[source] = path_map
        beta_path_sums[source] = {}
        
        # Now calculate the sum of powers (d^beta) for each found path
        for target, path in path_map.items():
            if source == target:
                beta_path_sums[source][target] = 0.0
                continue
                
            sum_beta = 0.0
            # Iterate over the edges in the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # If the edge exists in the original graph
                if (u, v) in edge_beta_cache:
                    sum_beta += edge_beta_cache[(u, v)]
                else:
                    #fallback
                    dist = graph[u][v]['dist']
                    sum_beta += dist ** beta
            
            beta_path_sums[source][target] = sum_beta

    return distances, beta_path_sums, paths_dict


def create_random_route(num_cities: int, rng: np.random.Generator) -> list:
    """
    Create a random permutation of cities (excluding the base city 0).

    Args: 
        num_cities: Total number of cities including base (0).
        rng: Random number generator.

    Returns:
        A list representing a random route.
    """
    cities = list(range(1, num_cities))
    rng.shuffle(cities)
    return cities

def create_greedy_permutation(num_cities: int, distances: dict) -> list:
    """
    Generate a pure Nearest Neighbor permutation.

    Args:
        num_cities: Total number of cities including base (0).
        distances: Precomputed distances dictionary.

    Returns:
        A list representing a greedy route.
    """
    unvisited = set(range(1, num_cities))
    current_node = 0
    path = []
    
    while unvisited:
        next_city = min(unvisited, key=lambda city: distances[current_node][city])
        path.append(next_city)
        unvisited.remove(next_city)
        current_node = next_city
        
    return path

def save_solution_to_file(path, problem_params, fitness, baseline, filename=None, output_dir=None):
    """
    Save solution results to CSV file.
    
    Args:
        path: Solution path (list of cities).
        problem_params: Dictionary containing problem parameters.
        fitness: Final GA cost.
        baseline: Baseline cost.
        filename: Optional filename (defaults to 'results.csv').
        output_dir: Optional output directory.
    
    Returns:
        Full path to the saved file.
    """
    if filename is None:
        filename = "results.csv"
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    
    # Calculate improvement percentage
    if baseline > 0:
        improvement = (baseline - fitness) / baseline * 100
    else:
        improvement = 0.0
    
    # Extract parameters
    problem_size = problem_params.get('num_cities', 'N/A')
    density = problem_params.get('density', 'N/A')
    alpha = problem_params.get('alpha', 'N/A')
    beta = problem_params.get('beta', 'N/A')
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        fieldnames = ['Problem_size', 'Density', 'Alpha', 'Beta', 'Baseline', 'GA_cost', 'Improvement_%', 'Path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow({
            'Problem_size': problem_size,
            'Density': density,
            'Alpha': alpha,
            'Beta': beta,
            'Baseline': f"{baseline:.2f}",
            'GA_cost': f"{fitness:.2f}",
            'Improvement_%': f"{improvement:.2f}",
            'Path': str(path)
        })
    
    return filename

def compute_exact_travel_cost(
    u: int, 
    v: int, 
    weight: float, 
    distances: Dict, 
    beta_sums: Dict, 
    alpha: float, 
    beta: float
) -> float:
    """
    Calcola il costo esatto replicando la logica hop-by-hop del professore.
    Formula implicita: Sum(dist_i) + (alpha * weight)^beta * Sum(dist_i^beta)
    """
    dist_total = distances[u][v]
    if weight == 0:
        return dist_total
    
    beta_sum_total = beta_sums[u][v]
    penalty_factor = (alpha * weight) ** beta
    return dist_total + penalty_factor * beta_sum_total

def _evaluate_greedy(genome, graph, distances, beta_sums, alpha, beta):
    total_cost = 0.0
    current_node = 0
    current_weight = 0.0
    
    physical_path = [] 
    trip_path = []      
    trip_cost = 0.0     

    for next_city in genome:
        gold_at_next = graph.nodes[next_city]['gold']
        
        # Compute cost from current_node to next_city
        cost_step_extend = compute_exact_travel_cost(current_node, next_city, current_weight, distances, beta_sums, alpha, beta)
        # Update weight
        new_weight = current_weight + gold_at_next
        # Compute cost from next_city back to base with new weight
        cost_return_future = compute_exact_travel_cost(next_city, 0, new_weight, distances, beta_sums, alpha, beta)
        
        total_continue_trip = trip_cost + cost_step_extend + cost_return_future # sum of the continued trip

        # Compute cost from current_node back to base with current weight
        cost_return_now = compute_exact_travel_cost(current_node, 0, current_weight, distances, beta_sums, alpha, beta)
        # Compute cost from base to next_city with zero weight
        cost_out_new = compute_exact_travel_cost(0, next_city, 0.0, distances, beta_sums, alpha, beta)
        # Compute cost from next_city back to base with gold_at_next weight
        cost_unload_restart = compute_exact_travel_cost(next_city, 0, gold_at_next, distances, beta_sums, alpha, beta)
        
        total_unload_restart = (trip_cost + cost_return_now) + cost_out_new + cost_unload_restart # sum of the unloaded trip

        if total_continue_trip <= total_unload_restart:
            trip_cost += cost_step_extend
            current_weight = new_weight
            current_node = next_city
            trip_path.append((next_city, gold_at_next))
        else:
            total_cost += (trip_cost + cost_return_now)
            physical_path.extend(trip_path)
            if current_node != 0: physical_path.append((0, 0))
            
            current_node = next_city
            current_weight = gold_at_next
            trip_cost = cost_out_new
            trip_path = [(next_city, gold_at_next)]

    # Compute cost to return to base at the end
    cost_final = compute_exact_travel_cost(current_node, 0, current_weight, distances, beta_sums, alpha, beta)
    total_cost += (trip_cost + cost_final)
    physical_path.extend(trip_path)
    physical_path.append((0, 0))

    return total_cost, physical_path


def _evaluate_split(genome, graph, distances, beta_sums, alpha, beta):
    """
    Evaluates the genome using the Split algorithm (Dynamic Programming)

    Args:
        genome: List of city indices (e.g., [5, 2, 8]).
        graph: NetworkX graph containing gold data.
        distances, beta_sums: Precomputed matrices for exact cost calculation.
        alpha: Parameter alpha for cost calculation.
        beta: Parameter beta for cost calculation.

    Returns:    
        Minimum total cost and the complete physical path.
    """

    n = len(genome)
    min_cost = [float('inf')] * (n + 1)
    min_cost[0] = 0.0
    predecessors = [0] * (n + 1)
    golds = [graph.nodes[c]['gold'] for c in genome]
    
    # Optimized window size based on beta
    MAX_TRIP_LENGTH = 15 if beta >= 1.5 else n

    # Dynamic Programming to compute min_cost and predecessors
    for i in range(1, n + 1):
        city_i = genome[i-1]
        dist_last_to_base = distances[city_i][0]
        
        
        limit = max(0, i - MAX_TRIP_LENGTH)
        
        # Evaluate trips ending at city_i starting from city_j
        for j in range(i - 1, limit - 1, -1):
            
            first_city = genome[j]
            trip_cost = compute_exact_travel_cost(0, first_city, 0.0, distances, beta_sums, alpha, beta)
            
            current_weight = golds[j]
            prev_node = first_city
            
            valid_trip = True

            # Iterate through intermediate cities in the trip
            for k in range(j + 1, i):
                next_node = genome[k]
                
                step_cost = compute_exact_travel_cost(prev_node, next_node, current_weight, distances, beta_sums, alpha, beta)
                trip_cost += step_cost
                
                if trip_cost > min_cost[i]: 
                    valid_trip = False
                    break
                    
                current_weight += golds[k]
                prev_node = next_node
            
            if not valid_trip: continue

            # Compute cost to return to base from city_i
            return_cost = compute_exact_travel_cost(city_i, 0, current_weight, distances, beta_sums, alpha, beta)
            trip_cost += return_cost
            
            if min_cost[j] + trip_cost < min_cost[i]:
                min_cost[i] = min_cost[j] + trip_cost
                predecessors[i] = j

    physical_path = []
    curr = n

    # Compute physical path via backtracking
    while curr > 0:
        prev = predecessors[curr]
        trip_segment = []
        for k in range(prev, curr):
            trip_segment.append((genome[k], golds[k]))
        trip_segment.append((0, 0))
        physical_path = trip_segment + physical_path
        curr = prev
        
    return min_cost[n], physical_path

def evaluate_route(genome: list, graph: nx.Graph, distances: Dict, beta_sums: Dict, alpha: float, beta: float) -> Tuple[float, list]:
    """
    Hybrid evaluation strategy selecting between Greedy and Split based on beta.

    Args:
        genome: List of city indices (e.g., [5, 2, 8]).
        graph: NetworkX graph containing gold data.
        distances, beta_sums: Precomputed matrices for exact cost calculation.
        alpha: Parameter alpha for cost calculation.
        beta: Parameter beta for cost calculation.

    Returns:    
        Total cost and the complete physical path.
    """
    if beta >= 1.5:
        return _evaluate_split(genome, graph, distances, beta_sums, alpha, beta)
    else:
        return _evaluate_greedy(genome, graph, distances, beta_sums, alpha, beta)
    
def plot_evolution(history, problem_params, save_path=None):
    """Plot fitness evolution over generations"""
    plt.figure(figsize=(10, 6))
    
    generations = range(len(history['best_history']))
    plt.plot(generations, history['best_history'], 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, history['avg_history'], 'g--', label='Average Fitness', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness (Cost)', fontsize=12)
    plt.title(f"GA Evolution (N={problem_params['num_cities']}, α={problem_params['alpha']}, β={problem_params['beta']})", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()