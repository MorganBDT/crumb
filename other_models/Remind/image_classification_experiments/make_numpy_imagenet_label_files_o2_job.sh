#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-010:00                         # Runtime in D-HH:MM format
#SBATCH --mem=16000M                         # Memory total in MB (for all cores)
#SBATCH -p short                             # Partition to run in (e.g. short, gpu)
#SBATCH -o ./o2_results/o2_results_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./o2_results/o2_errors_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=FAIL

module load conda2/4.2.13

source activate ml1

mkdir -p imagenet_files_run${RUN}/imagenet_indices
python make_numpy_imagenet_label_files.py --data "/n/groups/kreiman/shared_data/Imagenet2012" --labels_dir ./imagenet_files_run${RUN}/imagenet_indices --class_order_text_file "./imagenet_files/imagenet_class_order_run${RUN}.txt"
