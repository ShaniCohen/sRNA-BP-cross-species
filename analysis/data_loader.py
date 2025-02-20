from typing import Dict
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
from utils.general import read_df, write_df
from goatools.anno.gaf_reader import GafReader
import json
import sys
import os


ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")

from analysis import data_prepro as ap


def load_goa(file_path):
    gaf_reader = GafReader(file_path)
    return gaf_reader.read_gaf()


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class DataLoader:
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info(f"initializing DataLoader")
        self.config = config
        
        self.ecoli_k12_nm = 'ecoli_k12'    
        self.ecoli_epec_nm = 'ecoli_epec' 
        self.salmonella_nm = 'salmonella'
        self.strains_data = {}
        self.srna_acc_col = 'sRNA_accession_id'
        self.mrna_acc_col = 'mRNA_accession_id'

    def load_and_process_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # 1 - RNA and interactions data (self.strains_data)
        self._load_rna_and_inter_data()
        self._align_data()

    def _load_rna_and_inter_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        # 1 - Escherichia coli K12 MG1655
        #     srna_eco: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
        #     mrna_eco: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
        k12_dir = "Escherichia_coli_K12_MG1655"
        k12_mrna = read_df(file_path=join(self.config['rna_dir'], k12_dir, "mrna_eco.csv"))
        k12_srna = read_df(file_path=join(self.config['rna_dir'], k12_dir, "srna_eco.csv"))
        k12_inter = read_df(file_path=join(self.config['interactions_dir'], k12_dir, 'sInterBase_interactions_post_processing.csv'))
        k12_annot = load_goa(file_path=join(self.config['go_annotations_dir'], k12_dir, 'e_coli_MG1655.goa'))
        k12_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], k12_dir, 'InterProScan', 'Ecoli_k12_protein_sample.fasta.json'))


        k12_mrna, k12_srna, k12_inter = \
            ap.preprocess_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna, inter_data=k12_inter)
        k12_unq_inter, k12_sum, k12_srna, k12_mrna = ap.analyze_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna,
                                                                                inter_data=k12_inter)
        # 1.1 - update info
        self.strains_data.update({
            self.ecoli_k12_nm: {
                'all_mrna': k12_mrna,
                'all_srna': k12_srna,
                'unq_inter': k12_unq_inter,
                'all_inter': k12_inter,
                'all_srna_acc_col': 'EcoCyc_accession_id',
                'all_mrna_acc_col': 'EcoCyc_accession_id',
                'all_inter_srna_acc_col': 'sRNA_accession_id_Eco',
                'all_inter_mrna_acc_col': 'mRNA_accession_id_Eco',
                'k12_annot': k12_annot,
                'k12_annot_interproscan': k12_annot_interproscan
            }
        })

        # 2 - Escherichia coli EPEC E2348/69
        epec_dir = 'Mizrahi_2021_EPEC'
        epec_mrna = read_df(file_path=join(self.config['rna_dir'], epec_dir, "mizrahi_epec_all_mRNA_molecules.csv"))
        epec_srna = read_df(file_path=join(self.config['rna_dir'], epec_dir, "mizrahi_epec_all_sRNA_molecules.csv"))
        epec_inter = read_df(file_path=join(self.config['interactions_dir'], epec_dir, "mizrahi_epec_interactions.csv"))

        epec_mrna, epec_srna, epec_inter = \
            ap.preprocess_ecoli_epec_inter(mrna_data=epec_mrna, srna_data=epec_srna, inter_data=epec_inter)
        epec_unq_inter, epec_sum, epec_srna, epec_mrna = \
            ap.analyze_ecoli_epec_inter(mrna_data=epec_mrna, srna_data=epec_srna, inter_data=epec_inter)
        # 2.1 - update info
        self.strains_data.update({
            self.ecoli_epec_nm: {
                'all_mrna': epec_mrna,
                'all_srna': epec_srna,
                'unq_inter': epec_unq_inter,
                'all_inter': epec_inter,
                'all_srna_acc_col': 'sRNA_accession_id',
                'all_mrna_acc_col': 'mRNA_accession_id',
                'all_inter_srna_acc_col': 'sRNA_accession_id_Eco',
                'all_inter_mrna_acc_col': 'mRNA_accession_id_Eco'
            }
        })

        # 3 - Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1  (Matera_2022)
        salmonella_dir = 'Matera_2022_salmonella'
        salmonella_mrna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_mRNA_molecules.csv"))
        salmonella_srna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_sRNA_molecules.csv"))
        salmonella_inter = read_df(file_path=join(self.config['interactions_dir'], salmonella_dir, "matera_salmonella_interactions.csv"))

        salmonella_mrna, salmonella_srna, salmonella_inter = \
            ap.preprocess_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna,
                                        inter_data=salmonella_inter)
        salmo_unq_inter, salmo_sum, salmo_srna, salmo_mrna = \
            ap.analyze_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna, inter_data=salmonella_inter)
        # 3.1 - update info
        self.strains_data.update({
            self.salmonella_nm: {
                'all_mrna': salmo_mrna,
                'all_srna': salmo_srna,
                'unq_inter': salmo_unq_inter,
                'all_inter': salmonella_inter,
                'all_srna_acc_col': 'sRNA_accession_id',
                'all_mrna_acc_col': 'mRNA_accession_id',
                'all_inter_srna_acc_col': 'sRNA_accession_id',
                'all_inter_mrna_acc_col': 'mRNA_accession_id'
            }
        })

        # 4 - Vibrio cholerae, NCBI Genomes:  NC_002505.1 and NC_002506.1  (Huber 2022)

        # 5 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh_2024)

    def _align_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # 1 - align RNA accession ids for all datasets
        for strain_data in self.strains_data.values():
            # 1.1 - all sRNA
            strain_data['all_srna'] = strain_data['all_srna'].rename(columns={strain_data['all_srna_acc_col']: self.srna_acc_col})
            # 1.2 - all mRNA
            strain_data['all_mrna'] = strain_data['all_mrna'].rename(columns={strain_data['all_mrna_acc_col']: self.mrna_acc_col})
            # 1.3 - all interactions
            strain_data['all_inter'] = strain_data['all_inter'].rename(columns={strain_data['all_inter_srna_acc_col']: self.srna_acc_col})
            strain_data['all_inter'] = strain_data['all_inter'].rename(columns={strain_data['all_inter_mrna_acc_col']: self.mrna_acc_col})
            # 1.4 - unique interactions
            strain_data['unq_inter'] = strain_data['unq_inter'].rename(columns={strain_data['all_inter_srna_acc_col']: self.srna_acc_col})
            strain_data['unq_inter'] = strain_data['unq_inter'].rename(columns={strain_data['all_inter_mrna_acc_col']: self.mrna_acc_col})
            