#!/bin/bash
#SBATCH -J run
#SBATCH -D /users/kolodny/nsegal2/
#SBATCH --job-name=cdhit_prot_job # Job name
#SBATCH --output=/users/kolodny/nsegal2/cdhit/cdhit_prot_0.5/cdhit_prot_job.out # Standard output and error log
#SBATCH --error=/users/kolodny/nsegal2/cdhit/cdhit_prot_0.5/cdhit_prot_job.err # Error log
#SBATCH --gres gpu:1
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks (processes)
#SBATCH --cpus-per-task=16 # Number of CPU cores per task
#SBATCH --mem=4gb # Job memory request
#SBATCH --partition=dlc # Partition (queue) name

# Run CD-HIT
srun --container-image=/users/kolodny/nsegal2/cuda:12.5.1-cudnn-devel-rockylinux8.sqsh \
--container-mounts=/users/kolodny/nsegal2/cdhit:/code \
--container-workdir=/code \
--no-container-entrypoint \
--gres gpu:1 \
--nodes 1 \
--ntasks 1 \
--cpus-per-task 16 \
/bin/bash -c " \
./cd-hit-v4.8.1-2019-0228/cd-hit -i proteins_human.fasta -o ./cdhit_prot_0.5/proteins_human_50.fasta -d 10000 -c 0.5 -n 3 -T 16 -M 4000"
