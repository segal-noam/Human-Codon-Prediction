#!/bin/bash
#SBATCH -J run
#SBATCH -D /users/kolodny/nsegal2/
#SBATCH --job-name=makeblastdb # Job name
#SBATCH --output=/users/kolodny/nsegal2/blast/makeblastdb/makeblastdb.out # Standard output and error log
#SBATCH --error=/users/kolodny/nsegal2/blast/makeblastdb/makeblastdb.err # Error log
#SBATCH --gres gpu:1
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks (processes)
#SBATCH --cpus-per-task=1 # Number of CPU cores per task
#SBATCH --mem=4gb # Job memory request
#SBATCH --partition=dlc # Partition (queue) name

# Run makeblastdb
srun --container-image=/users/kolodny/nsegal2/cuda:12.5.1-cudnn-devel-rockylinux8.sqsh \
--container-mounts=/users/kolodny/nsegal2/blast:/code \
--container-workdir=/code \
--no-container-entrypoint \
--gres gpu:1 \
--nodes 1 \
--ntasks 1 \
--cpus-per-task 1 \
/bin/bash -c " \
./ncbi-blast-2.16.0+/bin/makeblastdb -in ./proteins_human_70_representatives.fasta -dbtype prot -out ./makeblastdb/proteins_human_70_representatives_db"
