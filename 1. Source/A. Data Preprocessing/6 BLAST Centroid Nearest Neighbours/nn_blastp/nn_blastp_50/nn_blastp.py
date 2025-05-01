# Imports

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd

# Path
import os

# --------------------------------------------
c = 0.5

# Load training data
train_df = pd.read_pickle(os.path.join('.', f"0_train_{int(100 * c)}.pkl"))

# --------------------------------------------

# Find Nearest Trainset Neighbors using BLASTP results

# BLAST results path
blast_results_path = os.path.join('.', f"proteins_human_{int(100 * c)}_representatives_blastp_0.1.txt")

# Read BLAST results into a DataFrame
columns = ['query_gene', 'query_transcript', 'subject_gene', 'subject_transcript', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'qseq', 'sseq']
df = pd.read_csv(blast_results_path, sep='[|\t]', engine='python', names=columns)

# Filter rows such that subject_gene is in the training set
df = df[df['subject_gene'].isin(train_df['gene'].values)]

# Replace pident values of 100 with 0 if there are no 0 values, else replace with NaN
print('Replacing 100 values with -1')
df['pident'] = df['pident'].replace(100, -1)

# Find nearest trainset neighbor for each query
nearest_trainset_neighbors = df.loc[df.groupby('query_transcript')['pident'].idxmax()]

# --------------------------------------------

# Save closest neighbors to a file
nearest_trainset_neighbors_path = os.path.join('.', f"proteins_human_{int(100 * c)}_representatives_nearest_trainset_neighbours_0.1.pkl")

nearest_trainset_neighbors.to_pickle(nearest_trainset_neighbors_path)