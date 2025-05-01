# --------------------------------------------
# Imports

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd

# --------------------------------------------
# Harmonizer
class Harmonizer:

    def __init__(self, _train_X, _train_Y, verbose=False):
        """Get aa sequences X and codon sequences Y pandas columns.

        Create Frequency Based Model."""

        if verbose:
            print("--- Training Frequency Model ---")

        # Create a DataFrame from the nucleotide and amino acid sequences
        _train_df = pd.DataFrame({'amino_acids': _train_X, 'nucleotides': _train_Y})
        
        if verbose:
            print("    * Extracting codons...")

        # Extract codons and corresponding amino acids
        _train_df['codons'] = _train_df['nucleotides'].apply(lambda x: [x[i:i+3] for i in range(0, len(x), 3)])
        _train_df['amino_acids'] = _train_df['amino_acids'].apply(list)

        if verbose:
            print("    * Creating aa-codon matrix...")
        # Explode the lists into separate rows
        _train_df = _train_df.explode(['codons', 'amino_acids'])
        
        if verbose:
            print("    * Calculating Frequencies...")

        # Count the frequency of each codon for each amino acid
        _codon_freq = _train_df.groupby(['amino_acids', 'codons']).size().unstack(fill_value=0)

        # Normalize the frequencies to get probabilities
        self.codon_prob = _codon_freq.div(_codon_freq.sum(axis=1), axis=0)

        # Clear the training DataFrame to save memory
        del _train_df
        del _codon_freq

        if verbose:
            print("--- Training Finished! ---")

    def predict_codons(self, amino_acid_sequences, verbose=False):
        """Predict the codon sequences for a pandas column of amino acid sequences."""
        
        if verbose:
            print("Predicting codons for amino acid sequences...")
            # Vectorized prediction
        def predict_sequence(seq):
            return ''.join(self.codon_prob.loc[aa].idxmax() for aa in seq)

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
# Expression Adjusted Harmonizer 
class Tissue_Expression_Harmonizers:

    def __init__(self, _train_df, verbose=False):
        """Get aa sequences X and codon sequences Y pandas columns.

        Create Frequency Based Model."""

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

        if verbose:
            print("--- Training Model ---")
        
        # Vanilla Harmonizer trained on all of the training data
        _train_X = _train_df['amino_acid_seq']
        _train_Y = _train_df['seq']
        if verbose:
            print(" Training vanilla harmonizer:")
        self.vanilla_harmonizer = Harmonizer(_train_X, _train_Y, verbose=verbose)
        
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
                    for _, row in _train_df.iterrows()
                    if isinstance(row['median'], tuple) and row['median'][i] == expression
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


    def predict_codons_majority_vote(self, amino_acids_expression_df, verbose=False):
        """Predict the codon sequences for a pandas column of amino acid sequences.
        
        By majority vote of the tissue and expression combination."""
        
        if verbose:
            print("Predicting codons for amino acid sequences (by majority vote)...")

        def predict_sequence(row):
            # Check if 'median' value is NaN
            if row[['median']].isna().any():
                return self.vanilla_harmonizer.predict_codons(pd.Series([row['amino_acid_seq']]))[0]
            
            # Get the indices for tissues and expressions
            tissue_indices = range(len(row['median']))
            expression_indices = [self.labels.index(expression) for expression in row['median']]
            
            # Predict codons for each tissue and expression combination
            predictions = [
            self.exp_harmonizers[tissue_idx][expression_idx].predict_codons(pd.Series([row['amino_acid_seq']]))[0]
            for tissue_idx, expression_idx in zip(tissue_indices, expression_indices)
            ]
            
            # Split predictions into codons
            prediction_codons = [list(map(''.join, zip(*[iter(pred)]*3))) for pred in predictions]
            
            # Transpose predictions to get codons at the same index together
            transposed_predictions = list(map(list, zip(*prediction_codons)))
            
            # Get the most frequent codon for each position
            consensus_codons = [max(set(codon_list), key=codon_list.count) for codon_list in transposed_predictions]
            
            return ''.join(consensus_codons)
        
        predictions = amino_acids_expression_df.apply(predict_sequence, axis=1)
        return predictions

# --------------------------------------------

# Load data
train_df = pd.read_pickle("./0_train_70.pkl")
# validation_df = pd.read_pickle("1_validation_70.pkl")
test_df = pd.read_pickle("./2_test_70.pkl")

# --------------------------------------------

# train expression adjusted harmonizer
tissue_exp_models = Tissue_Expression_Harmonizers(train_df, verbose=True)

# make test predictions
tissue_exp_predictions = tissue_exp_models.predict_for_all_tissue_exp(test_df, verbose=True)

# calculate test accuracies
accuracies_majority_vote = tissue_exp_models.calculate_accuracy(tissue_exp_predictions, test_df['seq'], verbose=True)

# save test_df predictions and accuracies columns as pickle
print("Saving predictions and accuracies...")
pd.DataFrame({'exp_harmonized_majority_vote_prediction':tissue_exp_predictions, 
              'exp_harmonized_majority_vote_accuracy':accuracies_majority_vote}).to_pickle("./2_70_exp_harmonized.pkl")

# print average accuracies
avg_accuracy_majority_vote = accuracies_majority_vote.mean()
print(f"Majority Vote Accuracy: {avg_accuracy_majority_vote}")

