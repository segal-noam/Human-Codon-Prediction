# --------------------------------------------
# imports
# Linear algebra
import numpy as np

# Dataframes
import pandas as pd

# Load JSON-stored data
import json

# Translate codons to amino acids
from Bio.Seq import Seq


# --------------------------------------------
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


# --------------------------------------------
# Harmonizer Class
class Harmonizer:

    def __init__(self, train_X, train_Y, verbose=False):
        """Get aa sequences X and codon sequences Y pandas columns.

        Create Frequency Based Model."""
        self.train_X = train_X
        self.train_Y = train_Y

        if verbose:
            print("--- Training Frequency Model ---")

        # Create a DataFrame from the nucleotide and amino acid sequences
        self.train_df = pd.DataFrame({'amino_acids': train_X, 'nucleotides': train_Y})
        
        if verbose:
            print("    * Calculating Codon Fractions...")
        
        self._amino_acid_codon_dict, \
        self._amino_acid_codon_frac_dict = self.calculate_amino_acid_codon_frac_dict()

        if verbose:
            print("    * Finding Most Frequent Codon for each Amino Acid...")
        
        self._amino_acid_most_freq_codon_dict = self.calculate_amino_acid_most_freq_codon_dict()

        if verbose:
            print("--- Training Finished! ---")

    # Create a dictionary mapping amino acids to list of codon frequencies
    def calculate_amino_acid_codon_frac_dict(self):
        amino_acid_codon_dict = get_amino_acid_codon_dict()

        amino_acid_codon_frac_dict = {aa: [0 for _ in amino_acid_codon_dict[aa]] for aa in amino_acid_codon_dict.keys()}

        total_counts = {aa: 0 for aa in amino_acid_codon_dict.keys()}

        for seq, aa_seq in zip(self.train_Y, self.train_X):
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

    # Create a dictionary mapping amino acids to most frequent codon
    def calculate_amino_acid_most_freq_codon_dict(self):
        amino_acid_most_freq_codon_dict = {}
        for aa, codons in self._amino_acid_codon_frac_dict.items():
            most_freq_index = np.argmax(codons)
            amino_acid_most_freq_codon_dict[aa] = self._amino_acid_codon_dict[aa][most_freq_index]
        return amino_acid_most_freq_codon_dict

    def predict_codons(self, amino_acid_sequences, verbose=False):
        """Predict the codon sequences for a pandas column of amino acid sequences."""
        
        if verbose:
            print("Predicting codons for amino acid sequences...")
            # Vectorized prediction
        def predict_sequence(aa_seq):
            return ''.join(self._amino_acid_most_freq_codon_dict[aa] for aa in aa_seq)

        predicted_codons = amino_acid_sequences.apply(predict_sequence)
        return predicted_codons.tolist()
    
    def calculate_accuracies(self, predictions, ground_truth, verbose=False):
        accuracies = []
        for pred, gt in zip(predictions, ground_truth):
            correct = sum([1 for p, g in zip([pred[i:i+3] for i in range(0, len(pred), 3)], [gt[i:i+3] for i in range(0, len(gt), 3)]) if p == g])
            assert len(gt) % 3 == 0
            total = len(gt) // 3
            accuracies.append(correct / total)
        return accuracies


# --------------------------------------------

if __name__ == "__main__":
    # Load data
    # Load train data from CSV and convert JSON strings back to lists
    train_df = pd.read_csv("0_train_50.json.csv")
    train_df["median"] = train_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)

    # Load test data from CSV and convert JSON strings back to lists
    test_df = pd.read_csv("2_test_50.json.csv")
    test_df["median"] = test_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)


    # --------------------------------------------
    # Make predictions and calculate accuracy
    mfc_model = Harmonizer(train_df['amino_acid_seq'], train_df['seq'], verbose=True)
    predictions = mfc_model.predict_codons(test_df['amino_acid_seq'], verbose=True)
    accuracies = mfc_model.calculate_accuracies(predictions, test_df['seq'], verbose=True)

    # save test_df predictions and accuracies columns as pickle
    print("Saving predictions and accuracies...")
    pd.DataFrame({'mfc_harmonized_prediction':predictions, 
                'mfc_accuracy':accuracies}).to_pickle("./2_50_mfc.pkl")


    avg_accuracy = np.mean(accuracies)

    print(f"Accuracy: {avg_accuracy}")