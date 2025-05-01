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
train_df = pd.read_csv(os.path.join('.', f"0_train_{int(100 * c)}.csv"))

# --------------------------------------------
# load BLASTP results 

# BLAST results path
blast_results_path = os.path.join('.', f"proteins_human_{int(100 * c)}_test_vs_train_representatives_blastp_0.01.txt")

# Read BLAST results into a DataFrame
columns = ['query_gene', 'query_transcript', 'subject_gene', 'subject_transcript', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'qseq', 'sseq']
df = pd.read_csv(blast_results_path, sep='[|\t]', engine='python', names=columns)

# --------------------------------------------
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
train_df = pd.read_csv(os.path.join('.', f"0_train_{int(100 * c)}.csv"))
test_df = pd.read_csv(os.path.join('.', f"2_test_{int(100 * c)}.csv"))

# --------------------------------------------
# load BLASTP results 

# BLAST results path
blast_results_path = os.path.join('.', f"proteins_human_{int(100 * c)}_test_vs_train_representatives_blastp_0.01.txt")

# Read BLAST results into a DataFrame
columns = ['query_gene', 'query_transcript', 'subject_gene', 'subject_transcript', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'qseq', 'sseq']
df = pd.read_csv(blast_results_path, sep='[|\t]', engine='python', names=columns)

# --------------------------------------------
# Add query length to blast data

# Merge BLASTP data with test_df to add query length
df = df.merge(test_df[['transcript', 'length (aa)']], left_on='query_transcript', right_on='transcript', how='left')

# Rename the 'length (aa)' column to 'query_length'
df = df.rename(columns={'length (aa)': 'query_length'})

# Drop the redundant 'transcript' column
df = df.drop(columns=['transcript'])

# --------------------------------------------
# Find Nearest Trainset Neighbors using BLASTP results

# Group by query_transcript and subject_transcript
grouped = df.groupby(['query_transcript', 'subject_transcript'])

# Calculate the average pident over qseq length
def calculate_average_pident(group):
    query_length = group['query_length'].iloc[0]

    # Create a boolean mask for each position in qseq (qstart/qend begin with index 1)
    positions = np.arange(1, query_length + 1)
    mask = np.zeros((len(group), query_length), dtype=bool)
    mask[np.arange(len(group))[:, None], np.arange(query_length)] = (
        (positions >= group['qstart'].values[:, None]) & 
        (positions <= group['qend'].values[:, None])
    )

    # Create a 2D array of pident values
    pidents_array = np.tile(group['pident'].values[:, None], (1, query_length))

    # Apply the mask to zero out values outside the qstart/qend range
    masked_pidents_array = np.where(mask, pidents_array, 0)

    # Take the maximum of each column (position) in the masked array
    max_pidents = masked_pidents_array.max(axis=0)

    # Calculate the average pident
    return np.mean(max_pidents) if max_pidents.size > 0 else float('nan')

# Apply the function to each group
average_pidents = grouped.apply(calculate_average_pident).reset_index(name='average_pident')

# Get nearest trainset neighbors using average pidents
nearest_trainset_neighbors = average_pidents.loc[average_pidents.groupby('query_transcript')['average_pident'].idxmax()]


# --------------------------------------------

# Save closest neighbors to a file
nearest_trainset_neighbors_path = os.path.join('.', f"proteins_human_{int(100 * c)}_testset_nearest_trainset_representatives_0.01.pkl")

nearest_trainset_neighbors.to_pickle(nearest_trainset_neighbors_path)
