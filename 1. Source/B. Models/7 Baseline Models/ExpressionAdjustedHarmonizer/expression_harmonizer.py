# --------------------------------------------
# Imports

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd
import json

# Translate codons to amino acids
from Bio.Seq import Seq

from os.path import join


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
# Harmonizer
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
# Expression Adjusted Harmonizer 
class Expression_Adjusted_Harmonizer:

    def __init__(self, train_df, verbose=False):
        """Get aa sequences X and codon sequences Y pandas columns.

        Create Frequency Based Model."""
        self.train_df = train_df

        self.labels = ['expr_low25', 'expr_pre25_50', 'expr_pre50_75', 'expr_pre75_90', 'expr_top10']
        self.tissues = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Artery_Aorta', 
                   'Artery_Coronary', 'Artery_Tibial', 'Bladder', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 
                   'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 
                   'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 
                   'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra', 'Breast_Mammary_Tissue', 'Cells_EBV-transformed_lymphocytes', 
                   'Cells_Cultured_fibroblasts', 'Cervix_Ectocervix', 'Cervix_Endocervix', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction', 
                   'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Fallopian_Tube', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle', 'Kidney_Cortex', 
                   'Kidney_Medulla', 'Liver', 'Lung', 'Minor_Salivary_Gland', 'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary', 
                   'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 
                   'Testis', 'Thyroid', 'Uterus', 'Vagina', 'Whole_Blood']
        

        self._amino_acid_codon_dict = get_amino_acid_codon_dict()

        if verbose:
            print("--- Training Model ---")
        
        # Vanilla Harmonizer trained on all of the training data
        train_X = train_df['amino_acid_seq']
        train_Y = train_df['seq']
        if verbose:
            print(" Training vanilla harmonizer:")
        self.vanilla_harmonizer = Harmonizer(train_X, train_Y, verbose=verbose)
        
        # Tissue-expression harmonizers trained on data such that tissue 'A' has expression 'B'
        if verbose:
            print(" Training tissue-expression harmonizers::")
        self.exp_harmonizers = []
        for i, tissue in enumerate(self.tissues):
            if verbose:
                print(f"  + Tissue = {tissue}: ")
            self.exp_harmonizers.append([])
            for expression in self.labels:
                if verbose:
                    print(f"    * Training {tissue} : {expression} harmonizer...")
                
                # Use a generator to avoid creating intermediate DataFrames
                tissue_expression_subset = (
                    (row['amino_acid_seq'], row['seq'])
                    for _, row in self.train_df.iterrows()
                    if isinstance(row['median'], list) and row['median'][i] == expression
                )

                tissue_expression_subset = list(tissue_expression_subset)
                if tissue_expression_subset:
                    _train_X, _train_Y = zip(*tissue_expression_subset)
                    tissue_expression_model = Harmonizer(pd.Series(_train_X), pd.Series(_train_Y))
                    self.exp_harmonizers[-1].append(tissue_expression_model)
                else:
                    if verbose:
                        print(f"      No data for {tissue} : {expression}")
                    self.exp_harmonizers[-1].append(None)


    def average_softmax_dicts(self, dict_list):
        """Average and softmax a list of dictionaries with the same keys 
        and same length list-values for a key."""
        
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        keys = dict_list[0].keys()
        averaged_dict = {key: np.mean([d[key] for d in dict_list], axis=0) for key in keys}
        return {key: softmax(values) for key, values in averaged_dict.items()}
    

    def calculate_amino_acid_most_freq_codon_dict(self, harmonizer_list):
        """Create a dictionary mapping amino acids to most frequent codon"""
        amino_acid_most_freq_codon_dict = {}

        dict_list = [harmonizer._amino_acid_codon_frac_dict for harmonizer in harmonizer_list]
        averaged_dict = self.average_softmax_dicts(dict_list)

        for aa, codon_fracs in averaged_dict.items():
            most_freq_index = np.argmax(codon_fracs)
            amino_acid_most_freq_codon_dict[aa] = self._amino_acid_codon_dict[aa][most_freq_index]

        return amino_acid_most_freq_codon_dict


    def predict_codons(self, amino_acids_expression_df, verbose=False):
        """Predict the codon sequences for a pandas dataframe 
        with a column 'amino_acid_seq' of amino acid sequences and a column 'median' of expression.
        
        By averaging the tissue-expression models' probabilities + softmax-ing"""
        
        if verbose:
            print("Predicting codons for amino acid sequences...")

        def predict_sequence(row):
            # Check if 'median' value is NaN
            if row[['median']].isna().any():
                return self.vanilla_harmonizer.predict_codons(pd.Series([row['amino_acid_seq']]))[0]

            # Get the indices for tissues and expressions
            tissue_indices = range(len(row['median']))
            expression_indices = [self.labels.index(expression) for expression in row['median']]
            
            # Predict codons for each tissue and expression combination
            harmonizer_list = [ 
            self.exp_harmonizers[tissue_idx][expression_idx] 
            for tissue_idx, expression_idx in zip(tissue_indices, expression_indices) 
            if not self.exp_harmonizers[tissue_idx][expression_idx] is None
            ]
            
            # Average the probabilities and softmax
            averaged_most_freq_dict = self.calculate_amino_acid_most_freq_codon_dict(harmonizer_list)

            return ''.join(averaged_most_freq_dict[aa] for aa in row["amino_acid_seq"])
        
        predictions = amino_acids_expression_df.apply(predict_sequence, axis=1)
        return predictions
       
    def calculate_accuracies(self, predictions, ground_truth, verbose=False):
        """Get model list of predictions and ground truth and return average accuracy"""

        if verbose:
            print("Calculating accuracy...")

        accuracies = []
        for pred, gt in zip(predictions, ground_truth):
            correct = sum([1 for p, g in zip([pred[i:i+3] for i in range(0, len(pred), 3)], [gt[i:i+3] for i in range(0, len(gt), 3)]) if p == g])
            assert len(gt) % 3 == 0
            total = len(gt) // 3
            accuracies.append(correct / total)
        return accuracies


# --------------------------------------------

# Load data
data_path = join('data')

# Load train data from CSV and convert JSON strings back to lists
train_df = pd.read_csv(join(data_path, "0_train_50.json.csv"))
train_df["median"] = train_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)

# Load test data from CSV and convert JSON strings back to lists
test_df = pd.read_csv(join(data_path, "2_test_50.json.csv"))
test_df["median"] = test_df["median"].apply(lambda x: json.loads(x) if pd.notnull(x) else x)

# --------------------------------------------

# train expression adjusted harmonizer
exp_model = Expression_Adjusted_Harmonizer(train_df, verbose=True)

# make test predictions
predictions = exp_model.predict_codons(test_df, verbose=True)

# calculate test accuracies
accuracies = exp_model.calculate_accuracies(predictions, test_df['seq'], verbose=True)

# save test_df predictions and accuracies columns as pickle
print("Saving predictions and accuracies...")
pd.DataFrame({'exp_harmonized_prediction':predictions, 
              'exp_harmonized_accuracy':accuracies}).to_pickle("./2_50_exp_harmonized.pkl")

# print average accuracies
avg_accuracy = np.mean(accuracies)
print(f"Average Accuracy: {avg_accuracy}")

