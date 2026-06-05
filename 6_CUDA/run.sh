#!/bin/bash
#SBATCH --job-name=hello_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=hello_%j.out
#SBATCH --error=hello_%j.err

module load cuda/12.1
nvcc -arch=sm_70 hello.cu -o hello
./hello