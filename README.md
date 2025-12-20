
# Distributed PageRank Algorithm with MPI

A parallel implementation of the PageRank algorithm using MPI-based MapReduce operations for computing webpage importance rankings in a distributed computing environment.

## Overview

This project implements the PageRank algorithm to analyze a web graph represented as edge lists across multiple files. Using `mpi4py`, the implementation distributes computation across multiple processes, with each process handling a subset of nodes. The algorithm uses the taxation method with β = 0.9 to compute webpage rankings through iterative MapReduce operations.

## Problem Statement

Given 10 data files where each row contains two webpage numbers `(source, destination)` indicating a hyperlink from the source webpage to the destination webpage, the goal is to:
- Run 4 MapReduce iterations of the PageRank algorithm
- Report the top 10 webpages with the largest PageRank values

**Example**: `(149, 35725)` means webpage 149 has a link to webpage 35725.

## Algorithm

The PageRank algorithm uses the taxation method with the following formula:

```
v' = (1 - β)/n + β(Mv + d/n)
```

Where:
- **β = 0.9**: Damping factor (taxation parameter)
- **M**: Transition matrix representing link structure
- **n**: Total number of nodes (webpages)
- **d**: Total rank from dangling nodes (nodes without outgoing links)
- **v**: Current PageRank vector

### MapReduce Workflow

**Map Step (Local Computation)**:
- Each MPI process handles a subset of source nodes (distributed round-robin by node index)
- For each owned node, distribute its current rank equally among all outgoing links
- Accumulate contributions in local NumPy arrays
- Track dangling nodes separately

**Reduce Step (Global Aggregation)**:
- Use `MPI.Allreduce` to sum partial contribution vectors across all processes
- Aggregate dangling node mass and redistribute uniformly
- Update the global PageRank vector using the taxation formula
- Broadcast updated ranks to all processes for the next iteration

## Requirements

- Python 3.x
- NumPy
- mpi4py
- Access to SLURM cluster (for HPC execution)

## Installation

```bash
# Install required packages
pip install numpy mpi4py
```

## Files

- `pagerank.py`: Main Python script implementing distributed PageRank
- `pagerank.slurm`: SLURM batch script for cluster execution
- `pagerank_iter_*.txt`: Intermediate PageRank values after each iteration
- `top10.txt`: Final top 10 webpages with the highest PageRank values

## Usage

### Running on SLURM Cluster

```bash
sbatch pagerank.slurm
```

### Running Standalone

```bash
mpiexec -n 5 python pagerank.py <data_dir> <output_dir> [--iterations 4]
```

### Command-line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `data_dir` | Directory containing input `.txt` edge files | Yes |
| `out_dir` | Directory to write output files | Yes |
| `--iterations` | Number of PageRank iterations (default: 4) | No |

### Example

```bash
mpiexec -n 5 python pagerank.py \
    /gpfs/projects/AMS598/projects2025_data/project2_data \
    /path/to/output \
    --iterations 4
```

## Implementation Details

### Key Functions

- **`scan_nodes()`**: Scans all edge files to discover unique node IDs and build a global node list
- **`build_local_adjacency()`**: Each rank builds adjacency lists only for its assigned source nodes (round-robin distribution)
- **`main()`**: Orchestrates MapReduce iterations with MPI communication and file I/O

### MPI Strategy

- **Node Distribution**: Nodes are assigned to processes using modulo operation: `node_idx % size == rank`
- **Data Structure**: NumPy arrays for efficient vectorized operations
- **Communication Pattern**: `MPI.Allreduce` for summing contributions, `MPI.Bcast` for distributing updated ranks
- **Scalability**: Supports an arbitrary number of processes (not limited to 5)

### SLURM Configuration

```bash
#SBATCH --ntasks=5              # 5 MPI processes
#SBATCH --time=01:00:00         # 1 hour time limit
#SBATCH --partition=short-40core
#SBATCH --mem-per-cpu=4000      # 4GB per CPU
```

## Output Format

### Intermediate Files

After each iteration, `pagerank_iter_N.txt` contains:
```
<node_id>    <pagerank_value>
149          0.00123456
35725        0.00234567
...
```

### Top 10 Results

`top10.txt` contains the 10 highest-ranked webpages:
```
<node_id>    <pagerank_value>
12345        0.01234567
67890        0.00987654
...
```

## Performance

- **Cluster Configuration**: 5 MPI processes, 4GB per CPU, short-40core partition
- **Iterations**: Fixed at 4 MapReduce iterations
- **Scalability**: Designed to work with an arbitrary number of processes
- **Efficiency**: Vectorized NumPy operations for fast local computations

## Algorithm Features

- ✅ **Parallel MapReduce**: Distributed computation across MPI processes
- ✅ **Efficient Communication**: Native MPI collective operations (`Allreduce`, `Bcast`)
- ✅ **Dangling Node Handling**: Proper redistribution of rank mass from nodes without outlinks
- ✅ **Vectorized Operations**: NumPy arrays for optimized numerical computations
- ✅ **Iterative Output**: Intermediate results saved after each iteration
- ✅ **Scalable Design**: Works with any number of processes and data files

## Key Implementation Highlights

### Round-Robin Node Assignment
```python
# Each rank processes nodes where: node_idx % size == rank
if (src_idx % size) != rank:
    continue
```

### Efficient Contribution Aggregation
```python
# Vectorized addition for duplicate edges
np.add.at(local_contrib, out, share)

# MPI reduction for global sum
comm.Allreduce(local_contrib, global_contrib, op=MPI.SUM)
```

### Dangling Node Treatment
```python
# Accumulate dangling mass locally
if out is None or out.size == 0:
    local_dangling += r

# Global sum and uniform redistribution
total_dangling = comm.allreduce(local_dangling, op=MPI.SUM)
new_ranks = (1.0 - BETA) / n + BETA * (global_contrib + total_dangling / n)
```

## Example Workflow

1. **Initialization**: Root process scans all files, discovers unique nodes
2. **Distribution**: Node list and file paths broadcast to all processes
3. **Local Adjacency**: Each rank builds adjacency only for its assigned nodes
4. **Iteration Loop** (4 times):
   - **Map**: Compute local PageRank contributions
   - **Reduce**: Aggregate contributions via `MPI.Allreduce`
   - **Update**: Apply PageRank formula with taxation
   - **Output**: Root writes intermediate results
5. **Finalization**: Root computes and writes the top 10 webpages

## Author

**Gayatri Sivani Susarla**  

- Institution: Stony Brook University
- Cluster: SeaWulf HPC
- Algorithm: PageRank (Page & Brin, 1998) with taxation method
