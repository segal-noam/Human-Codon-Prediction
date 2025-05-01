
# --------------------------------------
# imports 

# Install extra libraries
import subprocess
import sys

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

install_package('datasets')
install_package('Bio')

# argument parser
import argparse

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd
import json

# Path
import os

# Hugging face datasets library
from datasets import Dataset
from datasets import DatasetDict

# Translate codons to amino acids
from Bio.Seq import Seq

# --------------------------------------
# Parse arguments

parser = argparse.ArgumentParser(description='Generate Arrow DataBase for Training.')
parser.add_argument('--data_load_path', type=str, default=None,
                    help='location of the json csv data files.')
parser.add_argument('--data_save_path', type=str, default=None,
                    help='location of the output dataset')
parser.add_argument('--win_size', type=int, default=75,
                    help='window size for the dataset.')

args = parser.parse_args()
window_size = args.win_size
print(f'Generating dataset for window size: {window_size}')
data_load_path = args.data_load_path
data_save_path = args.data_save_path

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
        'codon_seq': all_codon_windows,
        'aa_seq': all_aa_windows
    })


# --------------------------------------
# Generate and Save Dataset

# Keep only necessary columns
train_df = train_df[['transcript', 'codon_seq', 'aa_seq', 'median']]
train_df.rename(columns={'median': 'expression'}, inplace=True)

test_df = test_df[['transcript', 'codon_seq', 'aa_seq', 'median']]
test_df.rename(columns={'median': 'expression'}, inplace=True)

# Fixed non-overlapping validation windows
validation_windows = df_to_windows(validation_df, window_size, stride_len=window_size)

# Convert dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_windows)
test_dataset = Dataset.from_pandas(test_df)
# Combine datasets into a single DatasetDict

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Save dataset
dataset_dict.save_to_disk(data_save_path)