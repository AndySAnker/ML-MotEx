#!/bin//bash
#SBATCH -p pearl
#SBATCH -t 48:00:00
#SBATCH --mem 300G
#SBATCH --output=slurm-%j.out

singularity exec ML-MotEx.sif python -u ML-MotEx_Step1_2.py
singularity exec ML-MotEx.sif python -u ML-MotEx_Step3_4.py
