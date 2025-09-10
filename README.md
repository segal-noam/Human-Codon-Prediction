# Human Codon Prediction
This project corresponds to the manuscript "LLM Reveals: Viruses Copy Silent Cell-Type Specific Patterns of Human Coding DNA".
Predicting the evolutionarily selected codons of human protein-coding genes, using a BART-based language model. The model receives the amino acids and tissue-expression values refined to percentile categories.

## Project Structure
1. Source: Comprised of both Jupyter notebooks integrated with graphs and results, as well as some Python files accompanied by SLURM SBATCH scripts - for the more computationally intensive tasks.
   - Data preprocessing:
      1. Retrieving the expression data from GTEx
      2. Cleaning the data
      3.  Refining expression to categories + Graphs
      4.  Clustering proteins with CD-HIT + Grouping same-gene protein sequences
      5.  Finding nearest trainset neighbours (to testset CD-HIT representatives) with BLASTP
   - Models:
      1. Baseline models (frequency-based)
      2. Generating the tokenizer
      3. Generating the arrow datasets
      4. Training and evaluating the BART models
   - Analysing the results:
      1. Various accuracy graphs
      2. Gene Set Enrichment Analysis for the accuracy difference metric
2. Data: The data used for each stage of the source.
3. Graphs & Results: The interesting graphs detailing the results of the project.
