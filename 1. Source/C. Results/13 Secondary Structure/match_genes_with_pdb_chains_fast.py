# -------------------------------------------------------------------------------
# Imports

# Install additional packages
import subprocess
import sys

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])

install_package('Bio')

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# biopython
# import Bio
# from Bio.Align import PairwiseAligner
# from Bio.pairwise2 import format_alignment
# Parse PDB files
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PPBuilder

# download pdb mmcif files
import requests

# Linear algebra
import numpy as np

# Dataframes
import pandas as pd

# Path and file handling
import os
data_path = os.path.join('.')

# Progress bar
from tqdm import tqdm

# -------------------------------------------------------------------------------
# Load Data

test_with_results = pd.read_csv(os.path.join(data_path, '2_test_with_bart30.json.csv'))

ensembl_to_uniprot = pd.read_csv('ensembl_to_uniprot.csv', sep=',')
id_mapping = pd.read_csv('idmapping_2025_04_02.tsv', sep='\t')

temp_data = pd.read_csv(os.path.join(data_path, 'genes_human.csv'), delimiter='\t')[['transcript', 'canonical', 'coord_start', 'coord_end', 'exp_T', 'exp_HEK293', 'exp_U2OS', 'cARS_T', 'cARS_HEK293', 'cARS_U2OS']]

merged_data = pd.merge(test_with_results, temp_data, right_on='transcript', left_on='query_transcript', how='left').drop(columns=['transcript'])

test_with_results = merged_data

# -------------------------------------------------------------------------------
# Match PDB with ENSEMBL Canonical Transcripts

id_mapping = id_mapping.merge(ensembl_to_uniprot, left_on='From', right_on='uniprot_id', how='left')
id_mapping = id_mapping.drop(columns=['uniprot_id'])

canonical_transcripts = test_with_results.loc[test_with_results['canonical'] == True][['query_gene', 'amino_acid_seq']]
id_mapping = id_mapping.merge(canonical_transcripts, left_on='ensembl_gene_id', right_on='query_gene', how='left')
id_mapping = id_mapping.drop(columns=['query_gene'])
id_mapping = id_mapping.rename(columns={'From': 'uniprot_id', 'To': 'pdb_id', 'amino_acid_seq': 'ensembl_amino_acid_seq'})

# -------------------------------------------------------------------------------
# Extract PDB Sequences

df = id_mapping[['pdb_id', 'uniprot_id']].drop_duplicates()

results = []
os.makedirs("pdb_tmp", exist_ok=True)

def get_chains_from_sifts(pdb_id, uniprot_id):
    """Query PDBe SIFTS API to get chain(s) in PDB corresponding to a UniProt ID"""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    
    data = r.json()
    chains = []
    
    mappings = data.get(pdb_id.lower(), {}).get('UniProt', {}).get(uniprot_id, {}).get("mappings", [])
    for m in mappings:
        chain_id = m.get("chain_id")
        if chain_id:
            chains.append(chain_id)
    
    return list(set(chains))  # Remove duplicates just in case

def process_pdb_entry(pdb_id, uniprot_id):
    parser = MMCIFParser(QUIET=True)  # separate instance for each thread to avoid errors
    ppb = PPBuilder()

    pdb_id_lower = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id_lower}.cif"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"Failed to download {pdb_id}")
            return None

        # Use in-memory string buffer instead of writing to disk
        handle = StringIO(r.text)
        structure = parser.get_structure(pdb_id, handle)
        model = structure[0]

        chain_ids = get_chains_from_sifts(pdb_id, uniprot_id)
        if not chain_ids:
            print(f"No matching chain found in {pdb_id} for UniProt {uniprot_id}")
            return None

        local_results = []

        for chain in model.get_chains():
            if chain.id not in chain_ids:
                continue

            seq = ''.join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
            mmcif_dict = parser._mmcif_dict
            ss_map = {}

            helices = mmcif_dict.get('_struct_conf.beg_auth_asym_id', [])
            for i, asym_id in enumerate(helices):
                if asym_id == chain.id:
                    beg = int(mmcif_dict['_struct_conf.beg_auth_seq_id'][i])
                    end = int(mmcif_dict['_struct_conf.end_auth_seq_id'][i])
                    for res_id in range(beg, end + 1):
                        ss_map[res_id] = 'H'

            strands = mmcif_dict.get('_struct_sheet_range.beg_auth_asym_id', [])
            for i, asym_id in enumerate(strands):
                if asym_id == chain.id:
                    beg = int(mmcif_dict['_struct_sheet_range.beg_auth_seq_id'][i])
                    end = int(mmcif_dict['_struct_sheet_range.end_auth_seq_id'][i])
                    for res_id in range(beg, end + 1):
                        ss_map[res_id] = 'E'

            ss = ''.join(
                ss_map.get(residue.get_id()[1], 'C')
                for residue in chain.get_residues()
                if residue.id[0] == ' '
            )

            local_results.append({
                'pdb_id': pdb_id,
                'uniprot_id': uniprot_id,
                'chain_id': chain.id,
                'sequence': seq,
                'secondary_structure': ss
            })

        return local_results

    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None

# Run in parallel
with ThreadPoolExecutor(max_workers=128) as executor:  # Adjust to suit your SLURM node capacity
    print(f"Num of threads: {os.cpu_count()}")
    print(f"Num of PDB entries: {len(df)}")
    futures = [
        executor.submit(process_pdb_entry, row['pdb_id'], row['uniprot_id'])
        for _, row in df.iterrows()
    ]

    for future in tqdm(as_completed(futures), total=len(futures)):
        res = future.result()
        if res:
            results.extend(res)

# Convert to dataframe and save
output_df = pd.DataFrame(results)

# -------------------------------------------------------------------------------
# Save the results

output_df.to_csv('uniprot_to_pdb_chain_sequences.csv', index=False)
