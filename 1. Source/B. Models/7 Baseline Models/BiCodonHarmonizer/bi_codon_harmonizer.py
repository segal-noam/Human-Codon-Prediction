# --------------------------------------------
# Imports

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd

# --------------------------------------------

# Bi-codon harmonizer class
class BiCodonHarmonizer:

    def __init__(self, train_X, train_Y, verbose=False):
        """Get aa sequences X and codon sequences Y pandas columns.

        Create Frequency Based Model for bi-codons."""

        if verbose:
            print("--- Training Bi-Codon Frequency Model ---")

        # Create a DataFrame from the nucleotide and amino acid sequences
        _train_df = pd.DataFrame({'amino_acids': train_X, 'nucleotides': train_Y})
        
        if verbose:
            print("    * Extracting codons...")

        # Extract codons and corresponding amino acids
        _train_df['codons'] = _train_df['nucleotides'].apply(lambda x: [x[i:i+3] for i in range(0, len(x), 3)])
        _train_df['amino_acids'] = _train_df['amino_acids'].apply(list)

        if verbose:
            print("    * Creating bi-codon pairs...")
        # Create bi-codon pairs
        _train_df['bi_codons'] = _train_df['codons'].apply(lambda x: [x[i] + " " + x[i+1] for i in range(len(x)-1)])
        _train_df['bi_amino_acids'] = _train_df['amino_acids'].apply(lambda x: [x[i] + " " + x[i+1] for i in range(len(x)-1)])

        if verbose:
            print("    * Exploding bi-codon pairs...")
            
        # Explode the lists into separate rows
        _train_df = _train_df.explode(['bi_codons', 'bi_amino_acids'])

        if verbose:
            print("    * Calculating Frequencies...")

        # Count the frequency of each bi-codon for each bi-amino acid
        _bi_codon_freq = _train_df.groupby(['bi_amino_acids', 'bi_codons']).size().unstack(fill_value=0)

        # Normalize the frequencies to get probabilities
        self.bi_codon_prob = _bi_codon_freq.div(_bi_codon_freq.sum(axis=1), axis=0)

        # Clear the training DataFrame to save memory
        del _train_df
        del _bi_codon_freq

        if verbose:
            print("--- Training Finished! ---")
        
    def predict_codons(self, amino_acid_sequences, verbose=False):
        """Predict the codon sequences for a pandas column of amino acid sequences."""
        
        if verbose:
            print("Predicting codons for amino acid sequences...")

        def predict_sequence(seq):
            predicted_codons = ['ATG']  # Start codon
            bi_aa_pairs = [seq[i-1] + " " + seq[i] for i in range(1, len(seq))]
            
            for bi_aa in bi_aa_pairs:
                prev_codon = predicted_codons[-1]
                bi_codon = self.bi_codon_prob.loc[bi_aa].filter(like=prev_codon, axis=0).idxmax()
                predicted_codons.append(bi_codon.split(" ")[1])
            
            return ''.join(predicted_codons)
        
        predicted_codons = amino_acid_sequences.apply(predict_sequence)
        return predicted_codons.tolist()

    def calculate_accuracy(self, prediction, ground_truth, verbose=False):
        """Get model list of predictions and ground truth and return average accuracy"""
        
        if verbose:
            print("Calculating accuracy...")

        # calculate accuracies
        comparison_df = pd.DataFrame({'prediction': prediction, 'ground_truth': ground_truth})
        comparison_df['accuracy'] = comparison_df.apply(lambda row: np.mean(np.array(list(row['prediction'])) == np.array(list(row['ground_truth']))), axis=1)
        return comparison_df['accuracy']

# --------------------------------------------
# Load data
train_df = pd.read_pickle("./0_train_70.pkl")
# validation_df = pd.read_pickle("1_validation_70.pkl")
test_df = pd.read_pickle("./2_test_70.pkl")

# --------------------------------------------
# Make predictions and calculate accuracy
bicodon_model = BiCodonHarmonizer(train_df['amino_acid_seq'], train_df['seq'], verbose=True)
predictions = bicodon_model.predict_codons(test_df['amino_acid_seq'], verbose=True)
accuracies = bicodon_model.calculate_accuracy(predictions, test_df['seq'], verbose=True)

# save test_df predictions and accuracies columns as pickle
print("Saving predictions and accuracies...")
pd.DataFrame({'bicodon_harmonized_prediction':predictions, 
              'bicodon_accuracy':accuracies}).to_pickle("./2_70_bi_codon.pkl")


avg_accuracy = accuracies.mean()

print(f"Accuracy: {avg_accuracy}")