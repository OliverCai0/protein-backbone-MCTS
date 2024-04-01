#!/bin/bash
#SBATCH -p short
#SBATCH --mem=12g
#SBATCH -o log_run

. /software/conda/bin/activate /home/ilutz/.conda/envs/ae_36
CMD=$(head -n $SLURM_ARRAY_TASK_ID tasks | tail -1)
exec ${CMD}

