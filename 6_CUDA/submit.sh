#!/bin/bash
#SBATCH --job-name=histogram
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=histogram.%j.out
#SBATCH --error=histogram.%j.err

module load cuda/12.1.1
nvcc -O3 -arch=sm_70 -o histogram histogram_shared.cu || exit 1

./histogram
