#!/bin/bash
#SBATCH -J run
#SBATCH -D /users/kolodny/nsegal2/
#SBATCH --job-name=generate_dataset # Job name
#SBATCH --output=/users/kolodny/nsegal2/generate_dataset/generate_dataset.out # Standard output and error log
#SBATCH --error=/users/kolodny/nsegal2/generate_dataset/generate_dataset.err # Error log
#SBATCH --gres gpu:1
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks (processes)
#SBATCH --cpus-per-task=1 # Number of CPU cores per task
#SBATCH --mem=64gb # Job memory request
#SBATCH --partition=dlc # Partition (queue) name

# Run python script
srun --container-image=/users/kolodny/nsegal2/pytorch:24.07-py3.sqsh \
--container-mounts=/users/kolodny/nsegal2/generate_dataset/:/code \
--container-workdir=/code \
--no-container-entrypoint \
--gres gpu:1 \
--nodes 1 \
--ntasks 1 \
--cpus-per-task 1 \
/bin/bash -c " \
python ./generate_dataset.py --win_size 30 --data_load_path '.' --data_save_path './dataset_30/'"