#!/bin/bash
#SBATCH -J run
#SBATCH -D /users/kolodny/nsegal2/
#SBATCH --job-name=blastp # Job name
#SBATCH --output=/users/kolodny/nsegal2/blast/blastp/blastp.out # Standard output and error log
#SBATCH --error=/users/kolodny/nsegal2/blast/blastp/blastp.err # Error log
#SBATCH --gres gpu:1
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks (processes)
#SBATCH --cpus-per-task=16 # Number of CPU cores per task
#SBATCH --mem=16gb # Job memory request
#SBATCH --partition=dlc # Partition (queue) name

# Run blastp
srun --container-image=/users/kolodny/nsegal2/cuda:12.5.1-cudnn-devel-rockylinux8.sqsh \
--container-mounts=/users/kolodny/nsegal2/blast:/code \
--container-workdir=/code \
--no-container-entrypoint \
--gres gpu:1 \
--nodes 1 \
--ntasks 1 \
--cpus-per-task 16 \
/bin/bash -c " \
./ncbi-blast-2.16.0+/bin/blastp -query proteins_human_70_representatives.fasta -db ./makeblastdb/proteins_human_70_representatives_db -evalue 0.01 -outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq' -out ./proteins_human_70_representatives_blastp.txt -num_threads 16"
