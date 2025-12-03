#!/bin/bash
#SBATCH --job-name=pagerank
#SBATCH --output=pagerank.out
#SBATCH --error=pagerank.err
#SBATCH --ntasks=5
#SBATCH --time=01:00:00
#SBATCH --partition=short-40core
#SBATCH --mem-per-cpu=4000

# Paths
DATA_DIR=/gpfs/projects/AMS598/projects2025_data/project2_data
OUT_DIR=/gpfs/projects/AMS598/class2025/Susarla_Gayatri/pagerank_output
SCRIPT=/gpfs/projects/AMS598/class2025/Susarla_Gayatri/pagerank.py

# Load modules
module purge
module load python
module load mpi4py/latest

# Ensure output directory exists
mkdir -p "$OUT_DIR"

echo "Starting PageRank job with $SLURM_NTASKS MPI processes..."
mpiexec -n $SLURM_NTASKS python "$SCRIPT" "$DATA_DIR" "$OUT_DIR"
echo "Job completed."
