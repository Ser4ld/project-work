# Project Work

## Table of Contents
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Execution](#execution)
- [Configuration](#configuration)
- [Output](#output)

---

## Problem Description

The **Thief and Gold** problem is an extension of the classic Traveling Salesman Problem (TSP) that introduces multi-objective optimization elements.

### Scenario
A thief must visit all cities in a graph to collect gold, starting from the base city (city 0) and returning to it. The complexity arises from the fact that:

- **Each city** (except city 0) contains a variable amount of gold
- **The weight carried** influences the travel cost according to a non-linear function
- The thief can **return to the base** multiple times during the route to deposit collected gold

### Cost Function
The total cost of a path is calculated as:

```
cost(path, weight) = dist + (α × dist × weight)^β
```

Where:
- `dist`: Euclidean distance between two cities
- `weight`: Weight of gold carried
- `α`, `β`: Parameters controlling the weight penalty

### Objective
Find the path that:
1. Visits all cities exactly once
2. Minimizes the total cost considering distances and gold weight
3. Starts and ends at city 0 (with the possibility of intermediate returns)

---

## Solution Approach

The implemented solution uses an advanced **Genetic Algorithm (GA)** with the following features:

### Main Components

#### 1. **Representation**
- **Genotype**: Permutation of cities (excluding city 0)
- **Phenotype**: Complete path with decisions on when to return to base to unload gold

#### 2. **Initial Population**
- **Greedy Base**: Nearest neighbor solution as starting point
- **Greedy Variants** (10%): Mutations of the greedy solution
- **Random Solutions** (90%): Random permutations for diversity

#### 3. **Genetic Operators**

**Selection**:
- Tournament Selection (tournament size: 5)

**Crossover**:
- Order Crossover (OX1) that preserves the relative order of cities
- Probability: 85%

**Adaptive Mutation**:
- **Inversion Mutation**: Inverts a path segment (simulates 2-opt)
- **Swap Mutation**: Swaps two random cities
- **Scramble Mutation**: Shuffles a random segment
- Base probability: 25% (adaptive based on stagnation)

#### 4. **Implemented Optimizations**

**Precomputation**:
- Shortest paths between all pairs of cities (Dijkstra)
- Distances and power sums pre-calculated for O(1) lookup

**Parallelization**:
- Parallel population evaluation using all available CPU cores
- Significant speedup on large problems

**Local Search**:
- Local 2-opt on elite solutions for refinement
- Applied every 50 generations on top 2 solutions

**Adaptive Mechanisms**:
- **Adaptive Mutation**: Increases mutation probability during stagnation
- **Population Restart**: Partial reset after 150 generations without improvement
- **Elite Preservation**: Best 2 solutions pass unchanged to next generation

#### 5. **Multi-Trip Strategy**
For each permutation, the algorithm automatically determines:
- When to return to base (city 0) to unload gold
- How many times to make the return trip
- Which gold to collect at each visited city

This is done through optimized dynamic programming that balances:
- Cost of carrying gold
- Cost of the return trip to base

---

## Repository Structure

```
project-work/
│
├── Problem.py              # Problem class: generates problem instances and baseline
├── s342269.py             # Main script: solution() function and tests
├── base_requirements.txt  # Basic Python dependencies
│
├── src/                   # Main GA module
│   ├── __init__.py       # Exports public interfaces
│   ├── GA.py             # GeneticAlgorithm implementation
│   ├── operators.py      # Genetic operators (selection, crossover, mutation)
│   └── utils.py          # Utilities (evaluation, precomputing, plotting, I/O)
│
├── plots/                # Fitness evolution plots (automatically generated)
├── results.csv          # Test results on various configurations
└── README.md            # This documentation
```

### Main File Descriptions

- **Problem.py**: Defines the `Problem` class that generates city graphs with gold, calculates costs, and provides a baseline solution
- **s342269.py**: Main script that runs tests on 12 different problem configurations
- **src/GA.py**: Core of the genetic algorithm with evolutionary logic, parallelization, and adaptive mechanisms
- **src/operators.py**: Implementation of genetic operators (tournament, OX1, mutations)
- **src/utils.py**: Support functions for precomputing, evaluation, result saving, and plotting

---

## Requirements

### Software
- Python 3.8+
- pip

### Python Libraries
Automatically installed from `base_requirements.txt`:
- numpy
- networkx
- matplotlib
- icecream
- tqdm

---

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd project-work
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r base_requirements.txt
```

---

## Execution

### Standard Execution
To run all predefined tests:

```bash
python s342269.py
```

This will execute 12 test configurations:
- **Sizes**: 100 and 1000 cities
- **Graph density**: 0.2 (sparse) and 1.0 (complete)
- **Parameters α, β**: (1,1), (2,1), (1,2)

## Configuration

### GA Parameters (in `src/GA.py`)

```python
@dataclass
class GAConfig:
    population_size: int = 100          # Population size
    generations: int = 1000             # Number of generations
    crossover_prob: float = 0.85        # Crossover probability
    mutation_prob: float = 0.25         # Base mutation probability
    elite_size: int = 2                 # Number of elite to preserve
    tournament_size: int = 5            # Tournament size
    seed: int = 42                      # Seed for reproducibility
    stagnation_threshold: int = 50      # Generations before increasing mutation
    restart_threshold: int = 150        # Generations before population restart
    local_search_iterations: int = 50   # Local 2-opt iterations
```

### Customization

To modify parameters:

```python
from src import GAConfig, GeneticAlgorithm

config = GAConfig(
    population_size=200,
    generations=2000,
    mutation_prob=0.3
)

ga = GeneticAlgorithm(p.graph, p.alpha, p.beta, config)
best_route, best_fitness, best_path = ga.evolve()
```

---

## Notes

### Rules
1. The thief must start and finish at city 0
2. Intermediate returns to city 0 to unload gold are allowed
3. All cities must be visited exactly once
