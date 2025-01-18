#!/bin/bash
#SBATCH --job-name=full_cpu_utilization
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=577Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive

# Load any necessary modules or source your Python environment here
# e.g., module load python/3.8

# Run the Python script for WindowConvolution
srun --ntasks=1 --cpus-per-task=64 python3 WindowConvolution/pipe_window.py

# Run the Python script for BLAS Optimized Convolution
srun --ntasks=1 --cpus-per-task=64 python3 Im2ColBLASConvolution/pipe_blas.py