from Problem import Problem
from src import solve, GAConfig
from src.utils import save_solution_to_file


def solution(p: Problem):
    """
    Solve the given Problem instance using Genetic Algorithm
    
    Args:
        p: Problem instance.
        
    Returns:
        Path as [(city, gold), ..., (0, 0)].
    """
    path, fitness = solve(p, verbose=True)
    return path, fitness


if __name__ == "__main__":
    
    # Test configurations (num_cities, density, alpha, beta)
    test_problems = [
        (100, 0.2, 1, 1),
        (100, 0.2, 2, 1),
        (100, 0.2, 1, 2),
        (100, 1, 1, 1),
        (100, 1, 2, 1),
        (100, 1, 1, 2),
        # Uncomment for large problems
        # (1000, 0.2, 1, 1),
        # (1000, 0.2, 2, 1),
        # (1000, 0.2, 1, 2),
        # (1000, 1, 1, 1),
        # (1000, 1, 2, 1),
        # (1000, 1, 1, 2),
    ]
    
    print("=" * 60)
    print("GENETIC ALGORITHM")
    print("=" * 60)
    
    results = []
    
    for num_cities, density, alpha, beta in test_problems:
        print(f"\n{'='*60}")
        p = Problem(num_cities, density=density, alpha=alpha, beta=beta)
        print(f"Problem: {len(p.graph.nodes())} cities, α={p.alpha}, β={p.beta}")
        print("=" * 60)
        
        baseline = p.baseline()
        print(f"Baseline: {baseline:.2f}")
        
        path, fitness = solution(p)
        
        # Validate
        cities_visited = {c for c, g in path if c != 0}
        expected = set(range(1, num_cities))
        
        valid = cities_visited == expected and path[-1] == (0, 0)
        status = "✓" if valid else "✗"
        print(f"{status} Solution valid: {valid}")
        print(f"\nPath format: {path[:5]}...{path[-3:]}" if len(path) > 8 else f"\nPath: {path}")
        print(f"Path length: {len(path)} steps")


        problem_params = {
            'num_cities': num_cities,
            'density': density,
            'alpha': alpha,
            'beta': beta
        }
        filename = save_solution_to_file(path, problem_params, fitness, baseline)
        print(f"📄 Solution saved to: {filename}")
        
        results.append({
            'cities': num_cities,
            'alpha': alpha,
            'beta': beta,
            'baseline': baseline,
            'valid': valid
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"Problem ({r['cities']} cities, α={r['alpha']}, β={r['beta']}): "
              f"baseline={r['baseline']:.2f}, valid={r['valid']}")
