from typing import Dict, List
import pandas as pd
import itertools
import numpy as np
from Bio import SeqIO
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
from preprocessing import general_pr as gp


def load_goa(file_path):
    gaf_reader = GafReader(file_path)
    return gaf_reader.read_gaf()


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_fasta(file_path) -> pd.DataFrame:
    records = []
    for rec in SeqIO.parse(file_path, "fasta"):
        records.append({
            'id': rec.id,
            'seq': str(rec.seq),
            'description': rec.description
        })
    return pd.DataFrame(records)

class DataLoader:
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info(f"initializing DataLoader")
        self.config = config
        self.clustering_config = self.config['clustering_config']
        
        self.ecoli_k12_nm = 'ecoli_k12'    
        self.ecoli_epec_nm = 'ecoli_epec' 
        self.salmonella_nm = 'salmonella'
        self.vibrio_nm = 'vibrio'
        self.klebsiella_nm = 'klebsiella'
        self.pseudomonas_nm = 'pseudomonas'

        self.strains = [self.ecoli_k12_nm, self.ecoli_epec_nm, self.salmonella_nm, self.vibrio_nm, self.klebsiella_nm, self.pseudomonas_nm]
        
        self.strains_data = {}
        self.srna_acc_col = 'sRNA_accession_id'
        self.mrna_acc_col = 'mRNA_accession_id'
        
        self.clustering_data = {}
        self.srna_seq_type = 'sRNA'
        self.protein_seq_type = 'protein'
    
    def get_strains(self) -> List[str]:
        return self.strains

    def load_and_process_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """store all data in a self.strains_data dictionary
        """
        # 1 - RNA and interactions data
        self._load_rna_and_inter_data()
        self._align_rna_and_inter_data()
        # 2 - proteins
        self._load_proteins()
        # 3 - GO annotations
        self._load_annotations()
        self._match_annotations_to_mrnas()
        # 4 - clustering
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
        k12_unq_inter, k12_sum, k12_srna, k12_mrna = ap.analyze_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna, inter_data=k12_inter)
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

        # 2 - Escherichia coli EPEC E2348/69  (Mizrahi 2021)
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

        # 3 - Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1  (Matera 2022)
        salmonella_dir = self.config['salmonella_dir']
        salmonella_mrna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_mRNA_molecules.csv"))
        salmonella_srna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_sRNA_molecules.csv"))
        salmonella_inter = read_df(file_path=join(self.config['interactions_dir'], salmonella_dir, "matera_salmonella_interactions.csv"))

        salmonella_mrna, salmonella_srna, salmonella_inter = \
            ap.preprocess_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna, inter_data=salmonella_inter)
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

        # 4 - Vibrio cholerae O1 biovar El Tor str. N16961 (NC_002505.1 and NC_002506.1)  (Huber 2022)
        vibrio_dir = self.config['vibrio_dir']
        vibrio_mrna = read_df(file_path=join(self.config['rna_dir'], vibrio_dir, "huber_vibrio_all_mRNA_molecules.csv"))
        vibrio_srna = read_df(file_path=join(self.config['rna_dir'], vibrio_dir, "huber_vibrio_all_sRNA_molecules.csv"))
        vibrio_inter = read_df(file_path=join(self.config['interactions_dir'], vibrio_dir, "huber_vibrio_interactions.csv"))

        vibrio_mrna, vibrio_srna, vibrio_inter = \
            ap.preprocess_vibrio_inter(mrna_data=vibrio_mrna, srna_data=vibrio_srna, inter_data=vibrio_inter)
        vibrio_unq_inter, vibrio_sum, vibrio_srna, vibrio_mrna = \
            ap.analyze_vibrio_inter(mrna_data=vibrio_mrna, srna_data=vibrio_srna, inter_data=vibrio_inter)
        
        # 4.1 - update info
        if self.vibrio_nm not in self.strains_data:
            self.strains_data[self.vibrio_nm] = {}
        self.strains_data[self.vibrio_nm].update({
            'all_mrna': vibrio_mrna,
            'all_srna': vibrio_srna,
            'unq_inter': vibrio_unq_inter,
            'all_inter': vibrio_inter,
            'all_srna_acc_col': 'sRNA_accession_id',
            'all_mrna_acc_col': 'mRNA_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id',
            'all_inter_mrna_acc_col': 'mRNA_accession_id'
        })

        # 5 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh 2024)
        klebsiella_dir = self.config['klebsiella_dir']
        klebsiella_mrna = read_df(file_path=join(self.config['rna_dir'], klebsiella_dir, "goh_klebsiella_all_mRNA_molecules.csv"))
        klebsiella_srna = read_df(file_path=join(self.config['rna_dir'], klebsiella_dir, "goh_klebsiella_all_sRNA_molecules.csv"))
        klebsiella_inter = read_df(file_path=join(self.config['interactions_dir'], klebsiella_dir, "goh_klebsiella_interactions.csv"))

        klebsiella_mrna, klebsiella_srna, klebsiella_inter = \
            ap.preprocess_klebsiella_inter(mrna_data=klebsiella_mrna, srna_data=klebsiella_srna, inter_data=klebsiella_inter)
        klebsiella_unq_inter, klebsiella_sum, klebsiella_srna, klebsiella_mrna = \
            ap.analyze_klebsiella_inter(mrna_data=klebsiella_mrna, srna_data=klebsiella_srna, inter_data=klebsiella_inter)
        
        # 4.1 - update info
        if self.klebsiella_nm not in self.strains_data:
            self.strains_data[self.klebsiella_nm] = {}
        self.strains_data[self.klebsiella_nm].update({
            'all_mrna': klebsiella_mrna,
            'all_srna': klebsiella_srna,
            'unq_inter': klebsiella_unq_inter,
            'all_inter': klebsiella_inter,
            'all_srna_acc_col': 'sRNA_accession_id',
            'all_mrna_acc_col': 'mRNA_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id',
            'all_inter_mrna_acc_col': 'mRNA_accession_id'
        })

        # 6 - Pseudomonas aeruginosa PAO1  (Gebhardt 2023)
        pseudomonas_dir = self.config['pseudomonas_dir']
        pseudomonas_mrna = read_df(file_path=join(self.config['rna_dir'], pseudomonas_dir, "gebhardt_pseudomonas_all_mRNA_molecules.csv"))
        pseudomonas_srna = read_df(file_path=join(self.config['rna_dir'], pseudomonas_dir, "gebhardt_pseudomonas_all_sRNA_molecules.csv"))
        pseudomonas_inter = read_df(file_path=join(self.config['interactions_dir'], pseudomonas_dir, "gebhardt_pseudomonas_interactions.csv"))

        pseudomonas_mrna, pseudomonas_srna, pseudomonas_inter = \
            ap.preprocess_pseudomonas_inter(mrna_data=pseudomonas_mrna, srna_data=pseudomonas_srna, inter_data=pseudomonas_inter)
        pseudomonas_unq_inter, pseudomonas_sum, pseudomonas_srna, pseudomonas_mrna = \
            ap.analyze_pseudomonas_inter(mrna_data=pseudomonas_mrna, srna_data=pseudomonas_srna, inter_data=pseudomonas_inter)
        
        # 4.1 - update info
        if self.pseudomonas_nm not in self.strains_data:
            self.strains_data[self.pseudomonas_nm] = {}
        self.strains_data[self.pseudomonas_nm].update({
            'all_mrna': pseudomonas_mrna,
            'all_srna': pseudomonas_srna,
            'unq_inter': pseudomonas_unq_inter,
            'all_inter': pseudomonas_inter,
            'all_srna_acc_col': 'sRNA_accession_id',
            'all_mrna_acc_col': 'mRNA_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id',
            'all_inter_mrna_acc_col': 'mRNA_accession_id'
        })

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
         
    def _load_proteins(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        for strain in self.strains: 
            if strain != self.klebsiella_nm:
                # 1 - load proteins
                proteins = load_fasta(file_path=join(self.config['proteins_dir'], f"{strain}_proteins.fasta"))
                proteins = proteins.rename(columns={'seq': 'protein_sequence', 'id': 'header'})
                # 1.1 - PATCH to adjust Salmonella headers
                if strain == self.salmonella_nm:
                    _lambda = lambda x: x.split("|")[0] + "|" + x.split("|")[2] + "|" + "|".join(x.split("|")[2:])
                    proteins['header'] = proteins['header'].apply(_lambda)
                
                # 2 - preprocess
                proteins = gp.parse_header_to_acc_locus_and_name(df=proteins, df_header_col='header', acc_col=self.mrna_acc_col, locus_col='mRNA_locus_tag', name_col='mRNA_name')
                
                # 3 - validate
                assert sum(pd.isnull(proteins[self.mrna_acc_col])) == 0, f"missing accession ids in {strain} proteins"
                assert len(proteins[self.mrna_acc_col].unique()) == len(proteins), f"duplicate accession ids in {strain} proteins"
                assert len(set(proteins[self.mrna_acc_col]) - set(self.strains_data[strain]['all_mrna'][self.mrna_acc_col])) == 0, "invalid mRNA accession ids in proteins"
                assert sum(pd.isnull(proteins['protein_sequence'])) == 0, f"missing protein sequences in {strain} proteins"
                
                # 4 - update info
                if strain not in self.strains_data:
                    self.strains_data[strain] = {}
                self.strains_data[strain].update({
                    'all_proteins': proteins
                })
    
    def _load_annotations(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        # 1 - Escherichia coli K12
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
        
        # 2 - Escherichia coli EPEC
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
    
        # 3 - Salmonella enterica
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
    
        # 4 - Vibrio cholerae

        # 5 - Klebsiella pneumoniae

        # 6 - Pseudomonas aeruginosa



    def _match_annotations_to_mrnas(self):
        for strain, data in self.strains_data.items():
            if 'curated_annot' in data:
                data['all_mrna_w_curated_annot'] = ap_annot.annotate_mrnas_w_curated_annt(strain, data)
            if 'interproscan_annot' in data:
                data['all_mrna_w_ips_annot'] = ap_annot.annotate_mrnas_w_interproscan_annt(strain, data)
            if 'eggnog_annot' in data:
                data['all_mrna_w_eggnog_annot'] = ap_annot.annotate_mrnas_w_eggnog_annt(strain, data)

    def _load_clustering_data(self):
        # 1 - load sRNA and mRNA clustering
        srna_clstr_dict = self._load_n_preprocess_bacteria_pairs_clustering(seq_type=self.srna_seq_type)
        mrna_clstr_dict = self._load_n_preprocess_bacteria_pairs_clustering(seq_type=self.protein_seq_type)

        # 2 - save clustering
        self.clustering_data['srna'] = srna_clstr_dict
        self.clustering_data['mrna'] = mrna_clstr_dict

    def _load_n_preprocess_bacteria_pairs_clustering(self, seq_type: str) -> pd.DataFrame:
        _dir = self.clustering_config[f'{seq_type.lower()}_dir']
        dir_path = join(self.config['clustering_dir'], seq_type, _dir)
        
        # ---------------------------   PAIRS preprocessing   ---------------------------
        self.logger.info(f"PAIRS preprocessing --> loading {seq_type} clustering data from {_dir}")
        pairs_clustering = {}
        for b1, b2 in itertools.combinations(self.strains, 2):
            # 1 - identify clustering files
            file_1_to_2 = [f for f in os.listdir(dir_path) if re.match(f"(.*?){b1}-{b2}.clstr$", f)][0]
            file_2_to_1 = [f for f in os.listdir(dir_path) if re.match(f"(.*?){b2}-{b1}.clstr$", f)][0]

            # 2 - load and parse the clustering files
            clstr_df_1_to_2 = self._load_n_parse_clstr_file(clstr_file_path=join(dir_path, file_1_to_2), seq_type=seq_type)
            clstr_df_2_to_1 = self._load_n_parse_clstr_file(clstr_file_path=join(dir_path, file_2_to_1), seq_type=seq_type)
            
            # 3 - filter invalid matches from the clustering data 
            clstr_df_1_to_2 = self._filter_invalid_matches(clstr_df_1_to_2, f"{b1} to {b2}", seq_type)
            clstr_df_2_to_1 = self._filter_invalid_matches(clstr_df_2_to_1, f"{b2} to {b1}", seq_type)

            # 4 - add to the pairs_clustering dictionary
            pairs_clustering[(b1, b2)] = clstr_df_1_to_2
            pairs_clustering[(b2, b1)] = clstr_df_2_to_1

        return pairs_clustering

    def _load_n_parse_clstr_file(self, clstr_file_path: str, seq_type: str, col_cluster_id: str = 'cluster_id', col_name: str = 'rna_name',
                                 col_seq_length: str = 'seq_length', col_is_rep: str = 'is_representative', col_similarity_score: str = 'similarity_score') -> pd.DataFrame:
        col_counter = 'counter'       
        col_strain_str = 'strain_str'
        col_acc = 'rna_accession_id'
        col_locus = 'rna_locus_tag'
        col_seq_location = 'seq_location'
        col_rep_location = 'rep_location'
        col_strand = 'strand'
        out_cols = [col_cluster_id, col_counter, col_strain_str, col_name, col_is_rep, col_seq_length, col_acc, col_locus, col_similarity_score, col_seq_location, col_rep_location]

        # 1 - bacterial names mapping
        _map = {
            'eco': self.ecoli_k12_nm,
            'epec': self.ecoli_epec_nm,
            'salmonella': self.salmonella_nm
        }

        # 2 - load the clstr file
        with open(clstr_file_path, "r", encoding="utf-8") as f:
            clstr_content = f.read()

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
        clstr_df[['counter_len', 'header', col_is_rep, 'footer']] = pd.DataFrame(list(map(lambda x: x.split(" "), clstr_df['entry'])))
        clstr_df[[col_counter, col_seq_length]] = pd.DataFrame(list(map(lambda x: x.split('nt,')[0].split('aa,')[0].split("\t"), clstr_df['counter_len'])))
        clstr_df[[col_strain_str, col_acc, col_locus, col_name]] = pd.DataFrame(list(map(lambda x: x.split('...')[0].split('>')[1].split('|'), clstr_df['header'])))
        clstr_df[col_strain_str] =  clstr_df[col_strain_str].apply(lambda x: _map[x])
        clstr_df[col_is_rep] =  clstr_df[col_is_rep].apply(lambda x: True if x=='*' else False)
        
        if seq_type == self.srna_seq_type:
            clstr_df[['location', col_strand, 'similarity_score_str']] =  pd.DataFrame(list(map(lambda x: x.split('/') if pd.notnull(x) else [None, None, None], clstr_df['footer'])))
            out_cols = out_cols + [col_strand]
        elif seq_type == self.protein_seq_type:
            clstr_df[['location', 'similarity_score_str']] =  pd.DataFrame(list(map(lambda x: x.split('/') if pd.notnull(x) else [None, None], clstr_df['footer'])))
        else:
            raise ValueError(f"seq_type {seq_type} is not supported")
        
        clstr_df['location'] =  pd.DataFrame(list(map(lambda x: re.findall(f"(.*?):(.*?):(.*?):(.*?)$", x)[0] if pd.notnull(x) else x, clstr_df['location'])))
        clstr_df[col_seq_location] =  pd.DataFrame(list(map(lambda x: (int(x[0]), int(x[1])) if pd.notnull(x) else x, clstr_df['location'])))
        clstr_df[col_rep_location] =  pd.DataFrame(list(map(lambda x: (int(x[2]), int(x[3])) if pd.notnull(x) else x, clstr_df['location'])))
        clstr_df[col_similarity_score] =  pd.DataFrame(list(map(lambda x: float(re.findall(f"(.*?)%$", x)[0]) if pd.notnull(x) else x, clstr_df['similarity_score_str'])))
        clstr_df = clstr_df[out_cols]

        # patch to adjust salmonella's accession id
        clstr_df[col_acc] = list(map(lambda strain, acc, locus: locus if strain==self.salmonella_nm else acc, clstr_df[col_strain_str], clstr_df[col_acc], clstr_df[col_locus]))
        
        return clstr_df
    
    def _filter_invalid_matches(self, clstr_df: pd.DataFrame, clstr_nm: str, seq_type: str, debug: bool = False,
                                col_cluster_id: str = 'cluster_id', col_name: str = 'rna_name', col_seq_length: str = 'seq_length', col_is_rep: str = 'is_representative', col_similarity_score: str = 'similarity_score') -> pd.DataFrame:
        """Filter invalid matches from the clustering df.
        Keep representatives. For matches, keep only if sequence length ratio (match/rep) >= 0.5.
        Returns filtered DataFrame.
        """
        filtered_rows = []
        for cluster_id, group in clstr_df.groupby(col_cluster_id):
            # Find representative row
            rep_row = group[group[col_is_rep] == True]
            rep_row = rep_row.iloc[0]
            rep_seq_length = float(rep_row[col_seq_length])
            # Always keep representative
            filtered_rows.append(rep_row)
            # Check matches
            match_rows = group[group[col_is_rep] == False]
            for _, match_row in match_rows.iterrows():
                match_seq_length = float(match_row[col_seq_length])
                if self._is_valid_match(seq_type, match_seq_length, rep_seq_length, match_row[col_similarity_score]):
                    filtered_rows.append(match_row)
                elif debug:
                    prfx = "**** RECHECK **** " if match_row[col_name].lower() == rep_row[col_name].lower() else ""
                    self.logger.debug(f"{prfx}{clstr_nm}: invalid match (seq ratio: {round(match_seq_length / rep_seq_length, 2)}, similarity: {match_row[col_similarity_score]}) {match_row[col_name]} to {rep_row[col_name]} in cluster id {cluster_id}")
            
        filtered_df = pd.DataFrame(filtered_rows, columns=clstr_df.columns)
        self.logger.info(f"{seq_type} clustering {clstr_nm} ---> filtered {len(clstr_df) - len(filtered_df)} invalid matches\n")
        return filtered_df
    
    def _is_valid_match(self, seq_type: str, match_seq_length: float, rep_seq_length: float, similarity_score: float):
        if seq_type == self.srna_seq_type:
            res = match_seq_length / rep_seq_length >= 0.5  
        elif seq_type == self.protein_seq_type:
            valid_1 = (match_seq_length / rep_seq_length >= 0.5) and similarity_score >= 75
            valid_2 = (match_seq_length / rep_seq_length >= 0.75) and similarity_score >= 50
            res = valid_1 or valid_2
        else:
            raise ValueError(f"seq_type {seq_type} is not supported")
        return res
    
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

