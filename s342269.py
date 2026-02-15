from Problem import Problem
from src import solve, GAConfig
from src.utils import is_valid, plot_evolution, save_solution_to_file, validate_solution


def solution(p: Problem):
    """
    Solve the given Problem instance using Genetic Algorithm
    
    Args:
        p: Problem instance.
        
    Returns:
        Path as [(city, gold), ..., (0, 0)].
    """
    path, fitness, history = solve(p, verbose=True)
    
    # Get problem parameters
    problem_params = {
        'num_cities': len(p.graph.nodes()),
        'density': p.density if hasattr(p, 'density') else 'unknown',
        'alpha': p.alpha,
        'beta': p.beta
    }
    
    # Define output directories
    output_dir = "results"
    plot_dir = "results/plots"
    
    # Save solution to file
    baseline = p.baseline()
    filename = save_solution_to_file(path, problem_params, fitness, baseline, 
                                     filename="results.csv", output_dir=output_dir)
    print(f"Solution saved to: {filename}")
    
    # Generate and save evolution plot
    plot_filename = f"p_{problem_params['num_cities']}_{problem_params['density']}_{p.alpha}_{p.beta}_evolution.png"
    plot_evolution(history, problem_params, save_path=f"{plot_dir}/{plot_filename}")
    print(f"Evolution plot saved to: {plot_dir}/{plot_filename}")
    
    return path

if __name__ == "__main__":
    
    # Test configurations (num_cities, density, alpha, beta)
    test_problems = [
        (100, 0.2, 1, 1),
        (100, 0.2, 2, 1),
        (100, 0.2, 1, 2),
        (100, 1, 1, 1),
        (100, 1, 2, 1),
        (100, 1, 1, 2),
        (1000, 0.2, 1, 1),
        (1000, 0.2, 2, 1),
        (1000, 0.2, 1, 2),
        (1000, 1, 1, 1),
        (1000, 1, 2, 1),
        (1000, 1, 1, 2),
    ]
    
    print("=" * 60)
    print("GENETIC ALGORITHM")
    print("=" * 60)
    
    for num_cities, density, alpha, beta in test_problems:
        print(f"\n{'='*60}")
        p = Problem(num_cities, density=density, alpha=alpha, beta=beta)
        p.density = density
        print(f"Problem: {len(p.graph.nodes())} cities, α={p.alpha}, β={p.beta}")
        print("=" * 60)
        
        baseline = p.baseline()
        print(f"Baseline: {baseline:.2f}")
        
        path = solution(p)
        
        # Validate with is_valid (uncomment for edge validity check)
        #valid, msg = validate_solution(p, path)
        #edge_check = all(is_valid(p, path))
        
        #status = "✓" if (valid and edge_check) else "✗"
        #print(f"{status} Solution valid: {valid} ({msg})")
        #print(f"  Edge validation: {'✓ all True' if edge_check else '❌ FAIL'}")

        print(f"\nPath format: {path[:5]}...{path[-3:]}" if len(path) > 8 else f"\nPath: {path}")
        print(f"Path length: {len(path)} steps")