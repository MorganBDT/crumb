#!/bin/bash
#SBATCH -c 4                               # Request 4 cores
#SBATCH -t 0-5:00                         # Runtime in D-HH:MM format
#SBATCH --mem=16000M                         # Memory total in MB (for all cores)
#SBATCH -p gpu                             # Partition to run in (e.g. short, gpu)
#SBATCH --gres=gpu:teslaK80:1
#SBATCH -o ./o2_results/o2_results_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./o2_results/o2_errors_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=FAIL

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/10.1

source activate /home/mbt10/.conda/envs/remind_proj

./run_non_imagenet_experiment_random_remind.sh $DATASET $SCENARIO 0 $RUN

