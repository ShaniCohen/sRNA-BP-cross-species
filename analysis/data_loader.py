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

from preprocessing import rna_inter_pr as ap
from preprocessing import annotations_pr as ap_annot


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
        """store all data in a self.strains_data dictionary
        """
        # 1 - RNA and interactions data
        self._load_rna_and_inter_data()
        self._align_rna_and_inter_data()
        # 2 - GO annotations
        self._load_annotations()
        self._preprocess_annotations()
    
    def _load_rna_and_inter_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        # 1 - Escherichia coli K12 MG1655
        k12_dir = self.config['k12_dir']
        k12_mrna = read_df(file_path=join(self.config['rna_dir'], k12_dir, "mrna_eco.csv"))
        k12_srna = read_df(file_path=join(self.config['rna_dir'], k12_dir, "srna_eco.csv"))
        k12_inter = read_df(file_path=join(self.config['interactions_dir'], k12_dir, 'sInterBase_interactions_post_processing.csv'))

        k12_mrna, k12_srna, k12_inter = \
            ap.preprocess_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna, inter_data=k12_inter)
        k12_unq_inter, k12_sum, k12_srna, k12_mrna = ap.analyze_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna,
                                                                                inter_data=k12_inter)
        # 1.1 - update info
        if self.ecoli_k12_nm not in self.strains_data:
            self.strains_data[self.ecoli_k12_nm] = {}
        self.strains_data[self.ecoli_k12_nm].update({
            'all_mrna': k12_mrna,
            'all_srna': k12_srna,
            'unq_inter': k12_unq_inter,
            'all_inter': k12_inter,
            'all_srna_acc_col': 'EcoCyc_accession_id',
            'all_mrna_acc_col': 'EcoCyc_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id_Eco',
            'all_inter_mrna_acc_col': 'mRNA_accession_id_Eco'
        })

        # 2 - Escherichia coli EPEC E2348/69
        epec_dir = self.config['epec_dir']
        epec_mrna = read_df(file_path=join(self.config['rna_dir'], epec_dir, "mizrahi_epec_all_mRNA_molecules.csv"))
        epec_srna = read_df(file_path=join(self.config['rna_dir'], epec_dir, "mizrahi_epec_all_sRNA_molecules.csv"))
        epec_inter = read_df(file_path=join(self.config['interactions_dir'], epec_dir, "mizrahi_epec_interactions.csv"))

        epec_mrna, epec_srna, epec_inter = \
            ap.preprocess_ecoli_epec_inter(mrna_data=epec_mrna, srna_data=epec_srna, inter_data=epec_inter)
        epec_unq_inter, epec_sum, epec_srna, epec_mrna = \
            ap.analyze_ecoli_epec_inter(mrna_data=epec_mrna, srna_data=epec_srna, inter_data=epec_inter)
        # 2.1 - update info
        if self.ecoli_epec_nm not in self.strains_data:
            self.strains_data[self.ecoli_epec_nm] = {}
        self.strains_data[self.ecoli_epec_nm].update({
            'all_mrna': epec_mrna,
            'all_srna': epec_srna,
            'unq_inter': epec_unq_inter,
            'all_inter': epec_inter,
            'all_srna_acc_col': 'sRNA_accession_id',
            'all_mrna_acc_col': 'mRNA_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id_Eco',
            'all_inter_mrna_acc_col': 'mRNA_accession_id_Eco'
        })

        # 3 - Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1  (Matera_2022)
        salmonella_dir = self.config['salmonella_dir']
        salmonella_mrna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_mRNA_molecules.csv"))
        salmonella_srna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_sRNA_molecules.csv"))
        salmonella_inter = read_df(file_path=join(self.config['interactions_dir'], salmonella_dir, "matera_salmonella_interactions.csv"))

        salmonella_mrna, salmonella_srna, salmonella_inter = \
            ap.preprocess_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna,
                                        inter_data=salmonella_inter)
        salmo_unq_inter, salmo_sum, salmo_srna, salmo_mrna = \
            ap.analyze_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna, inter_data=salmonella_inter)
        # 3.1 - update info
        if self.salmonella_nm not in self.strains_data:
            self.strains_data[self.salmonella_nm] = {}
        self.strains_data[self.salmonella_nm].update({
            'all_mrna': salmo_mrna,
            'all_srna': salmo_srna,
            'unq_inter': salmo_unq_inter,
            'all_inter': salmonella_inter,
            'all_srna_acc_col': 'sRNA_accession_id',
            'all_mrna_acc_col': 'mRNA_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id',
            'all_inter_mrna_acc_col': 'mRNA_accession_id'
        })

        # 4 - Vibrio cholerae, NCBI Genomes:  NC_002505.1 and NC_002506.1  (Huber 2022)

        # 5 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh_2024)


    def _align_rna_and_inter_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        self._align_accession_ids_between_rna_and_inter()
        self._assert_rna_df_columns()
    
    def _align_accession_ids_between_rna_and_inter(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        for data in self.strains_data.values():
            # 1 - all sRNA
            data['all_srna'] = data['all_srna'].rename(columns={data['all_srna_acc_col']: self.srna_acc_col})
            # 2 - all mRNA
            data['all_mrna'] = data['all_mrna'].rename(columns={data['all_mrna_acc_col']: self.mrna_acc_col})
            # 3 - all interactions
            data['all_inter'] = data['all_inter'].rename(columns={data['all_inter_srna_acc_col']: self.srna_acc_col})
            data['all_inter'] = data['all_inter'].rename(columns={data['all_inter_mrna_acc_col']: self.mrna_acc_col})
            # 4 - unique interactions
            data['unq_inter'] = data['unq_inter'].rename(columns={data['all_inter_srna_acc_col']: self.srna_acc_col})
            data['unq_inter'] = data['unq_inter'].rename(columns={data['all_inter_mrna_acc_col']: self.mrna_acc_col})
            # 5 - update keys
            data['srna_acc_col'] = self.srna_acc_col
            data['mrna_acc_col'] = self.mrna_acc_col
            data.pop('all_srna_acc_col', None)
            data.pop('all_mrna_acc_col', None)
            data.pop('all_inter_srna_acc_col', None)
            data.pop('all_inter_mrna_acc_col', None)
    
    def _assert_rna_df_columns(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        for data in self.strains_data.values():
            # 1 - assert
            for rna in ['sRNA', 'mRNA']:
                expected_cols = [f'{rna}_locus_tag', f'{rna}_name', f'{rna}_name_synonyms', f'{rna}_start', f'{rna}_end', f'{rna}_strand', f'{rna}_sequence']
                rna_df = data[f'all_{rna.lower()}']
                assert all([col in rna_df.columns for col in expected_cols]), f"missing columns in {rna} data"
            # 2 - set self columns
            data['all_mrna_locus_col'] = 'mRNA_locus_tag'
            data['all_mrna_name_col'] = 'mRNA_name'
            data['all_mrna_name_syn_col'] = 'mRNA_name_synonyms'
    
    def _load_annotations(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        # 1 - Escherichia coli K12 MG1655
        k12_dir = self.config['k12_dir']
        k12_annot_uniport = load_goa(file_path=join(self.config['go_annotations_dir'], k12_dir, 'e_coli_MG1655.goa'))
        k12_annot_map_uniport_to_locus = read_df(file_path=join(self.config['go_annotations_dir'], k12_dir, 'ECOLI_83333_idmapping.dat'))
        k12_annot_map_uniport_to_locus.columns = ['UniProt_ID', 'Database', 'Mapped_ID']
        k12_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], k12_dir, 'InterProScan', 'Ecoli_k12_proteins.fasta.json'))
        
        interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(k12_annot_interproscan)
        curated_annot, c_locus_col = ap_annot.preprocess_curated_annot(self.ecoli_k12_nm, k12_annot_uniport, k12_annot_map_uniport_to_locus)

        # 1.1 - update info
        if self.ecoli_k12_nm not in self.strains_data:
            self.strains_data[self.ecoli_k12_nm] = {}
        self.strains_data[self.ecoli_k12_nm].update({
            "interproscan_annot": interproscan_annot,
            "interproscan_header_col": i_header_col,
            "curated_annot": curated_annot,
            "curated_locus_col": c_locus_col
        })

        """
        # 2 - Escherichia coli EPEC E2348/69
        epec_dir = self.config['epec_dir']
        epec_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], epec_dir, 'InterProScan', ''))
    

        # 3 - Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1  (Matera_2022)
        salmonella_dir = self.config['salmonella_dir']
        salmonella_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], salmonella_dir, 'InterProScan', ''))
    

        # 4 - Vibrio cholerae, NCBI Genomes:  NC_002505.1 and NC_002506.1  (Huber 2022)

        # 5 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh_2024)
        """

    def _preprocess_annotations(self):
        for strain, data in self.strains_data.items():
            if 'curated_annot' in data:
                data['all_mrna_w_curated_annt'] = ap_annot.annotate_mrnas_w_curated_annt(strain, data['all_mrna'], data['all_mrna_locus_col'], data['curated_annot'], data['curated_locus_col'])
            if 'interproscan_annot' in data:
                data['interproscan_annot'] = ap_annot.parse_header_to_acc_locus_and_name(data['interproscan_annot'], data['interproscan_header_col'], data['mrna_acc_col'], data['all_mrna_locus_col'], data['all_mrna_name_col'])
                # stats = self.compute_unique_mrna_names()
                # TODO: get updated interproscan files for K12. 
                # (1) make GO ref list unique (in load_annotations)
                # (2) find how many mRNAs were mapped to go terms.
                data['interproscan_annot'] = ap_annot.annotate_mrnas_w_interproscan_annt(strain, data['all_mrna'], data['mrna_acc_col'], data['interproscan_annot'], data['interproscan_header_col'])

    def compute_unique_mrna_names(self) -> pd.DataFrame:
        """Compute the number of unique mRNA_name for each value of signature_library in interproscan_annot."""
        results = []
        for strain, data in self.strains_data.items():
            if 'interproscan_annot' in data:
                unique_counts = data['interproscan_annot'].groupby('signature_library')['mRNA_name'].nunique().reset_index()
                unique_counts.columns = ['signature_library', 'unique_mrna_name_count']
                unique_counts['strain'] = strain
                results.append(unique_counts)
        return pd.concat(results, ignore_index=True)

