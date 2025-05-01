
# --------------------------------------
# imports 

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd
import json

# Path
import os
data_load_path = os.path.join('.')
data_save_path = os.path.join('window_datasets')

# Hugging face datasets library
from datasets import Dataset

# Translate codons to amino acids
from Bio.Seq import Seq


# --------------------------------------
# Load data

# Load train data from CSV and convert JSON strings back to lists
train_df = pd.read_csv(os.path.join(data_load_path, "0_train_50.json.csv"))
train_df["median"] = train_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)

# Load validation data from CSV and convert JSON strings back to lists
validation_df = pd.read_csv(os.path.join(data_load_path, "1_validation_50.json.csv"))
validation_df["median"] = validation_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)

# Load test data from CSV and convert JSON strings back to lists
test_df = pd.read_csv(os.path.join(data_load_path, "2_test_50.json.csv"))
test_df["median"] = test_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)


# --------------------------------------
# Space Out Codons

def insert_codon_spaces(seq):
    return ' '.join([seq[i:i+3] for i in range(0, len(seq), 3)])

train_df['codon_seq'] = train_df['seq'].apply(insert_codon_spaces)
validation_df['codon_seq'] = validation_df['seq'].apply(insert_codon_spaces)
test_df['codon_seq'] = test_df['seq'].apply(insert_codon_spaces)


# --------------------------------------
# Space Out Amino Acids

def insert_aa_spaces(seq):
    return ' '.join([seq[i] for i in range(0, len(seq))])

train_df['aa_seq'] = train_df['amino_acid_seq'].apply(insert_aa_spaces)
validation_df['aa_seq'] = validation_df['amino_acid_seq'].apply(insert_aa_spaces)
test_df['aa_seq'] = test_df['amino_acid_seq'].apply(insert_aa_spaces)


# --------------------------------------
# Replace NaN expressions with NaN tuples

nan_tuple = tuple([np.nan] * 54)

train_df['median'] = train_df['median'].apply(lambda x: nan_tuple if np.all(pd.isna(x)) else x)
validation_df['median'] = validation_df['median'].apply(lambda x: nan_tuple if np.all(pd.isna(x)) else x)
test_df['median'] = test_df['median'].apply(lambda x: nan_tuple if np.all(pd.isna(x)) else x)


# --------------------------------------
# Utils

def get_dna_codons():
  nt_bases = ['A', 'C', 'G', 'T']
  codons = [c1 + c2 + c3 for c1 in nt_bases for c2 in nt_bases for c3 in nt_bases]
  return codons

# Create a dictionary mapping amino acids to all possible codons
def get_amino_acid_codon_dict():
  codons = get_dna_codons()
  
  amino_acid_codon_dict = {}
  for codon in codons:
    aa = str(Seq(codon).translate())
    if aa not in amino_acid_codon_dict:
      amino_acid_codon_dict[aa] = []
    amino_acid_codon_dict[aa].append(codon)
  return amino_acid_codon_dict


# --------------------------------------
# Define CAI class

class CAI:

    def __init__(self, dna_sequences, aa_sequences, verbose=False):
        """Get aa sequences X and codon sequences Y pandas columns.

        Create Frequency Based Model."""
        self.dna_sequences = dna_sequences
        self.aa_sequences = aa_sequences

        if verbose:
            print("--- Training CAI Model ---")
        
        if verbose:
            print("    * Calculating Codon Fractions...")
        
        self._amino_acid_codon_dict, \
        self._amino_acid_codon_frac_dict = self.calculate_amino_acid_codon_frac_dict()

        if verbose:
            print("--- Training Finished! ---")

    # Create a dictionary mapping amino acids to list of codon frequencies
    def calculate_amino_acid_codon_frac_dict(self):
        amino_acid_codon_dict = get_amino_acid_codon_dict()

        amino_acid_codon_frac_dict = {aa: [0 for _ in amino_acid_codon_dict[aa]] for aa in amino_acid_codon_dict.keys()}

        total_counts = {aa: 0 for aa in amino_acid_codon_dict.keys()}

        for seq, aa_seq in zip(self.dna_sequences, self.aa_sequences):
            codon_list = [seq[i:i+3] for i in range(0, len(seq), 3)]
            aa_seq = str(Seq(seq).translate())

            for aa, codon in zip(aa_seq, codon_list):
                if codon in amino_acid_codon_dict[aa]:
                    codon_index = amino_acid_codon_dict[aa].index(codon)
                    amino_acid_codon_frac_dict[aa][codon_index] += 1
                    total_counts[aa] += 1

        for aa, codons in amino_acid_codon_frac_dict.items():
            for i in range(len(codons)):
                if total_counts[aa] > 0:
                    amino_acid_codon_frac_dict[aa][i] /= total_counts[aa]

        return amino_acid_codon_dict, amino_acid_codon_frac_dict

    # Calculate CAI for pandas row
    def calculate_sequence_CAI(self, row):
        aa_list = row['aa_seq'].split(' ')
        codon_list = row['codon_seq'].split(' ')

        cai_values = [
            self._amino_acid_codon_frac_dict[aa][self._amino_acid_codon_dict[aa].index(codon)]
            for aa, codon in zip(aa_list, codon_list)
            if aa in self._amino_acid_codon_dict
        ]

        return np.prod(cai_values) ** (1 / len(cai_values))


# --------------------------------------
# # Train CAI model

# # Unify the dataframes
# all_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)

# # Create the CAI model
# cai_model = CAI(all_df['seq'], all_df['amino_acid_seq'], verbose=True)


# --------------------------------------
# Define Sliding Windows Functions

def get_overlapping_windows(sequence, window_len, stride_len=1, is_codon=False):
    """
    Get overlapping windows of a sequence.
    """
    if is_codon:
        window_len = window_len * 4 - 1  # Adjust for spaces between codons
        stride_len = stride_len * 4  # Adjust for spaces between codons
    else:
        window_len = window_len * 2 - 1  # Adjust for spaces between amino acids
        stride_len = stride_len * 2  # Adjust for spaces between amino acids

    if len(sequence) < window_len:
        return [sequence]
    
    windows = [sequence[i:i+window_len] for i in range(0, len(sequence) - window_len + 1, stride_len)]
    
    # Ensure the last section of the sequence is included
    remainder = (len(sequence) - window_len) % stride_len
    if remainder != 0:
        windows.append(sequence[-(remainder - 1):])
    
    return windows


def df_to_windows(df, window_len, stride_len=1):
    """Get dataframe with columns 'seq' and 'amino_acid_seq', 
    
    Return those columns as a list of windows of size 'window_len' with stride length 'stride_len'.
    """
    all_transcripts = []
    all_codon_windows = []
    all_aa_windows = []
    all_expression_vectors = []
    # all_cai_values = []

    for _, row in df.iterrows():
        codon_windows = get_overlapping_windows(row['codon_seq'], window_len, stride_len, is_codon=True)
        aa_windows = get_overlapping_windows(row['aa_seq'], window_len, stride_len, is_codon=False)
        expression_vector = row['median']
        transcript_id = row['transcript']
        # cai = cai_model.calculate_sequence_CAI(row)

        for codon_window, aa_window in zip(codon_windows, aa_windows):
            all_transcripts.append(transcript_id)
            all_codon_windows.append(codon_window)
            all_aa_windows.append(aa_window)
            all_expression_vectors.append(expression_vector)
            # all_cai_values.append(cai)

    return pd.DataFrame({
        'transcript': all_transcripts,
        'expression': all_expression_vectors,
        # 'cai': all_cai_values,
        'codon_windows': all_codon_windows,
        'aa_windows': all_aa_windows
    })


# --------------------------------------
# Generate and Save Datasets

# WINDOW_SIZES = [10, 30, 50, 75, 100, 150]
WINDOW_SIZES = [30]

for window_size in WINDOW_SIZES:
    # Print window size
    print(f'Generating datasets for window size: {window_size}')

    # Overlapping train/test windows
    train_windows = df_to_windows(train_df, window_size)
    test_windows = df_to_windows(test_df, window_size)

    # Non-overlapping validation windows
    validation_windows = df_to_windows(validation_df, window_size, stride_len=window_size)

    # load windows as datasets
    train_dataset = Dataset.from_pandas(train_windows)
    validation_dataset = Dataset.from_pandas(validation_windows)
    test_dataset = Dataset.from_pandas(test_windows)

    pd.set_option('display.max_columns', None)
    print(train_windows.head())
    
    # Save datasets
    
    train_dataset.save_to_disk(os.path.join(data_save_path, f'{window_size}_windows', f'0_50_{window_size}_windows_train'))
    validation_dataset.save_to_disk(os.path.join(data_save_path, f'{window_size}_windows', f'1_50_{window_size}_windows_validation'))
    test_dataset.save_to_disk(os.path.join(data_save_path, f'{window_size}_windows', f'2_50_{window_size}_windows_test'))
