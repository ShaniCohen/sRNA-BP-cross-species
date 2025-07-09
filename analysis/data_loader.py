from typing import Dict
import pandas as pd
import itertools
import numpy as np
import re
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
        self.clustering_config = self.config['clustering_config']
        
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
        self._match_annotations_to_mrnas()
        # 3 - clustering
        self._load_clustering_data()
    
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
        
		# TODO: check sRNA SL1344_0808
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
        
        # 2 - Escherichia coli EPEC E2348/69
        epec_dir = self.config['epec_dir']
        epec_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], epec_dir, 'InterProScan', 'EPEC_proteins.fasta.json'))
        eggnog_annot_file=join(self.config['go_annotations_dir'], epec_dir, 'EggNog', 'EPEC.annotations')

        interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(epec_annot_interproscan)
        eggnog_annot, e_header_col = ap_annot.load_and_preprocess_eggnog_annot(eggnog_annot_file)
        
		# 2.1 - update info
        if self.ecoli_epec_nm not in self.strains_data:
            self.strains_data[self.ecoli_epec_nm] = {}
        self.strains_data[self.ecoli_epec_nm].update({
            "interproscan_annot": interproscan_annot,
            "interproscan_header_col": i_header_col,
            "eggnog_annot": eggnog_annot,
            "eggnog_header_col": e_header_col
        })
    
        # 3 - Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1  (Matera_2022)
        salmonella_dir = self.config['salmonella_dir']
        salmonella_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], salmonella_dir, 'InterProScan', 'Salmonella_proteins.fasta.json'))
        eggnog_annot_file=join(self.config['go_annotations_dir'], salmonella_dir, 'EggNog', 'Salmonella.annotations')
        
        interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(salmonella_annot_interproscan)
        eggnog_annot, e_header_col = ap_annot.load_and_preprocess_eggnog_annot(eggnog_annot_file)
        # patch to adjust Salmonella headers
        _lambda = lambda x: x.split("|")[0] + "|" + x.split("|")[2] + "|" + "|".join(x.split("|")[2:])
        interproscan_annot[i_header_col] = interproscan_annot[i_header_col].apply(_lambda)
        eggnog_annot[e_header_col] = eggnog_annot[e_header_col].apply(_lambda)
        
		# 3.1 - update info
        if self.salmonella_nm not in self.strains_data:
            self.strains_data[self.salmonella_nm] = {}
        self.strains_data[self.salmonella_nm].update({
            "interproscan_annot": interproscan_annot,
            "interproscan_header_col": i_header_col,
            "eggnog_annot": eggnog_annot,
            "eggnog_header_col": e_header_col
        })
    
        # 4 - Vibrio cholerae, NCBI Genomes:  NC_002505.1 and NC_002506.1  (Huber 2022)

        # 5 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh_2024)

    def _match_annotations_to_mrnas(self):
        for strain, data in self.strains_data.items():
            if 'curated_annot' in data:
                data['all_mrna_w_curated_annot'] = ap_annot.annotate_mrnas_w_curated_annt(strain, data)
            if 'interproscan_annot' in data:
                data['all_mrna_w_ips_annot'] = ap_annot.annotate_mrnas_w_interproscan_annt(strain, data)
            if 'eggnog_annot' in data:
                data['all_mrna_w_eggnog_annot'] = ap_annot.annotate_mrnas_w_eggnog_annt(strain, data)

    def _load_clustering_data(self):
        # 1 - load sRNA clustering
        # self._load_rna_clustering(rna_type='srna')
        return
    
    def _load_n_parse_clstr_file(self, clstr_file_path: str) -> pd.DataFrame:
        col_cluster_id = 'cluster_id'
        col_counter = 'counter'
        col_seq_length = 'seq_length'
        col_strain_str = 'strain_str'
        col_acc = 'rna_accession_id'
        col_locus = 'rna_locus_tag'
        col_name = 'rna_name'
        col_is_rep = 'is_representative'
        out_cols = [col_cluster_id, col_counter, col_strain_str, col_name, col_is_rep, col_seq_length, col_acc, col_locus]

        # 1 - bacterial names mapping
        _map = {
            'eco': self.ecoli_k12_nm,
            'epec': self.ecoli_epec_nm,
            'salmonella': self.salmonella_nm
        }

        # 2 - load the clstr file
        with open(clstr_file_path, "r", encoding="utf-8") as f:
            clstr_content = f.read()
        self.logger.info(f"Loaded clustering file: {clstr_file_path}")

        # 3 - parse .clstr file into a DataFrame
        clusters = []
        cluster_id = None
        for line in clstr_content.splitlines():
            line = line.strip()
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[1])
            elif line:
                clusters.append({col_cluster_id: cluster_id, "entry": line})
        clstr_df = pd.DataFrame(clusters)

        # 4 - split the entries into columns and preprocess
        clstr_df[['counter_len', 'header', col_is_rep, 'temp']] = pd.DataFrame(list(map(lambda x: x.split(" "), clstr_df['entry'])))
        clstr_df[[col_counter, col_seq_length]] = pd.DataFrame(list(map(lambda x: x.split('nt,')[0].split("\t"), clstr_df['counter_len'])))
        clstr_df[[col_strain_str, col_acc, col_locus, col_name]] = pd.DataFrame(list(map(lambda x: x.split('...')[0].split('>')[1].split('|'), clstr_df['header'])))
        clstr_df[col_strain_str] =  clstr_df[col_strain_str].apply(lambda x: _map[x])
        clstr_df[col_is_rep] =  clstr_df[col_is_rep].apply(lambda x: True if x=='*' else False)
        clstr_df = clstr_df[out_cols]

        return clstr_df

    def _load_rna_clustering(self, rna_type: str) -> pd.DataFrame:
        dir_path = join(self.config['clustering_dir'], self.clustering_config[f'{rna_type}_dir'])
        
        # ---------------------------   PAIRS preprocessing   ---------------------------
        #TODO: add epec_slmo, save output in self
        pairs_clustering = {}
        bacteria = [self.ecoli_k12_nm, self.ecoli_epec_nm, self.salmonella_nm]
        for b1, b2 in itertools.combinations(bacteria, 2):
            # 1 - define the clustering file path
            or_pattern = f"((.*?){b1}-{b2}|(.*?){b2}-{b1}).clstr$"
            files = [f for f in os.listdir(dir_path) if re.match(or_pattern, f)]
            # assert len(files) == 1, f"Expected exactly one clustering file for {rna_type} {b1} {b2} in {dir_path}, found: {files}"
            if len(files) > 0:
                clstr_file_path = join(dir_path, files[0])

                # 2 - load and parse the clustering file
                clstr_df = self._load_n_parse_clstr_file(clstr_file_path=clstr_file_path)

                # 3 - add to the pairs_clustering dictionary
                pairs_clustering[(b1, b2)] = clstr_df

        print()

        # # 1 - Escherichia coli K12 MG1655
        # k12_dir = self.config['k12_dir']
        # k12_annot_uniport = load_goa(file_path=join(self.config['go_annotations_dir'], k12_dir, 'e_coli_MG1655.goa'))
        # k12_annot_map_uniport_to_locus = read_df(file_path=join(self.config['go_annotations_dir'], k12_dir, 'ECOLI_83333_idmapping.dat'))
        # k12_annot_map_uniport_to_locus.columns = ['UniProt_ID', 'Database', 'Mapped_ID']
        # k12_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], k12_dir, 'InterProScan', 'Ecoli_k12_proteins.fasta.json'))
        
        # interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(k12_annot_interproscan)
        # curated_annot, c_locus_col = ap_annot.preprocess_curated_annot(self.ecoli_k12_nm, k12_annot_uniport, k12_annot_map_uniport_to_locus)

        # # 1.1 - update info
        # if self.ecoli_k12_nm not in self.strains_data:
        #     self.strains_data[self.ecoli_k12_nm] = {}
        # self.strains_data[self.ecoli_k12_nm].update({
        #     "interproscan_annot": interproscan_annot,
        #     "interproscan_header_col": i_header_col,
        #     "curated_annot": curated_annot,
        #     "curated_locus_col": c_locus_col
        # })
  
        # # 2 - Escherichia coli EPEC E2348/69
        # epec_dir = self.config['epec_dir']
        # epec_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], epec_dir, 'InterProScan', 'EPEC_proteins.fasta.json'))
        # interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(epec_annot_interproscan)
        
		# # 2.1 - update info
        # if self.ecoli_epec_nm not in self.strains_data:
        #     self.strains_data[self.ecoli_epec_nm] = {}
        # self.strains_data[self.ecoli_epec_nm].update({
        #     "interproscan_annot": interproscan_annot,
        #     "interproscan_header_col": i_header_col
        # })
    
        # # 3 - Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1  (Matera_2022)
        # salmonella_dir = self.config['salmonella_dir']
        # salmonella_annot_interproscan = load_json(file_path=join(self.config['go_annotations_dir'], salmonella_dir, 'InterProScan', 'Salmonella_proteins.fasta.json'))
        # interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(salmonella_annot_interproscan)
        # # patch to adjust Salmonella headers
        # interproscan_annot[i_header_col] = interproscan_annot[i_header_col].apply(lambda x: "|".join([x.split("|")[0]] + [x.split("|")[2]] + x.split("|")[2:]))
        
		# # 3.1 - update info
        # if self.salmonella_nm not in self.strains_data:
        #     self.strains_data[self.salmonella_nm] = {}
        # self.strains_data[self.salmonella_nm].update({
        #     "interproscan_annot": interproscan_annot,
        #     "interproscan_header_col": i_header_col
        # })
    
        # # 4 - Vibrio cholerae, NCBI Genomes:  NC_002505.1 and NC_002506.1  (Huber 2022)

        # # 5 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh_2024)
        


        return 
    
    # def compute_unique_mrna_names(self) -> pd.DataFrame:
    #     """Compute the number of unique mRNA_name for each value of signature_library in interproscan_annot."""
    #     results = []
    #     for strain, data in self.strains_data.items():
    #         if 'interproscan_annot' in data:
    #             unique_counts = data['interproscan_annot'].groupby('signature_library')['mRNA_name'].nunique().reset_index()
    #             unique_counts.columns = ['signature_library', 'unique_mrna_name_count']
    #             unique_counts['strain'] = strain
    #             results.append(unique_counts)
    #     return pd.concat(results, ignore_index=True)

