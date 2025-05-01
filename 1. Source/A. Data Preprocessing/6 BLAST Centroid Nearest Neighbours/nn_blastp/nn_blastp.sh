#!/bin/bash
#SBATCH -J run
#SBATCH -D /users/kolodny/nsegal2/
#SBATCH --job-name=nn_blastp_50 # Job name
#SBATCH --output=/users/kolodny/nsegal2/blast/nn_blastp/nn_blastp_50/nn_blastp.out # Standard output and error log
#SBATCH --error=/users/kolodny/nsegal2/blast/nn_blastp/nn_blastp_50/nn_blastp.err # Error log
#SBATCH --gres gpu:1
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks (processes)
#SBATCH --cpus-per-task=1 # Number of CPU cores per task
#SBATCH --mem=16gb # Job memory request
#SBATCH --partition=dlc # Partition (queue) name

# Run blastp
srun --container-image=/users/kolodny/tsidi/image23.07_updated.sqsh \
--container-mounts=/users/kolodny/nsegal2/blast:/code \
--container-workdir=/code \
--no-container-entrypoint \
--gres gpu:1 \
--nodes 1 \
--ntasks 1 \
--cpus-per-task 1 \
/bin/bash -c " \
python ./nn_blastp/nn_blastp_50/nn_blastp.py"
