from typing import Dict, List
import pandas as pd
import itertools
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re
import pickle
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


def load_fasta(file_path, header_col = 'header', seq_col = 'seq') -> pd.DataFrame:
    records = []
    for rec in SeqIO.parse(file_path, "fasta"):
        records.append({
            header_col: rec.description,
            seq_col: str(rec.seq)
            # id_col: rec.id,
        })
    return pd.DataFrame(records)


def write_fasta(df: pd.DataFrame, out_path: str, header_col: str = 'header', seq_col: str = 'seq'):
    """
    Write a DataFrame with columns 'header' and 'seq' to a FASTA file.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():    
            header = row[header_col]
            seq = row[seq_col]
            # seq_id = row[id_col]

            header = f">{header}"
            f.write(header + "\n")
            f.write(seq.replace("\n", "") + "\n")

class DataLoader:
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info(f"initializing DataLoader")
        self.config = config
        self.clustering_config = self.config['clustering_config']
        
        self.ecoli_k12_nm = 'ecoli_k12'    
        self.ecoli_epec_nm = 'ecoli_epec' 
        self.salmonella_nm = 'salmonella'
        self.klebsiella_nm = 'klebsiella'
        self.vibrio_nm = 'vibrio'
        self.pseudomonas_nm = 'pseudomonas'

        self.strains = [self.ecoli_k12_nm, self.ecoli_epec_nm, self.salmonella_nm, self.klebsiella_nm, self.vibrio_nm, self.pseudomonas_nm]
        
        self.strains_data = {}
        self.srna_acc_col = 'sRNA_accession_id'
        self.mrna_acc_col = 'mRNA_accession_id'
        
        self.clustering_data = {}
        self.srna_seq_type = 'sRNA'
        self.protein_seq_type = 'protein'

        self.dump_rna_and_inter_data_summary = True
        self.generate_clean_rna_fasta = False
    
    def get_strains(self) -> List[str]:
        return self.strains

    def load_and_process_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """store all data in a self.strains_data dictionary
        """
        # 1 - RNA and interactions data
        self._load_rna_and_inter_data()
        self._align_rna_and_inter_data()
        if self.generate_clean_rna_fasta:
            self._generate_clean_rna_fasta_files()
        # 2 - proteins
        self._load_proteins()
        self._match_proteins_to_mrnas()
        # 3 - GO annotations
        self._load_annotations()
        self._match_annotations_to_mrnas()
        # 4 - clustering
        self._load_clustering_data(load_from_pickle=True)
    
    def _load_rna_and_inter_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        # 1 - Escherichia coli K12 MG1655
        k12_dir = self.config[f'{self.ecoli_k12_nm}_dir']
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
        epec_dir = self.config[f'{self.ecoli_epec_nm}_dir']
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
        salmonella_dir = self.config[f'{self.salmonella_nm}_dir']
        salmonella_mrna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_mRNA_molecules.csv"))
        salmonella_srna = read_df(file_path=join(self.config['rna_dir'], salmonella_dir, "matera_salmonella_all_sRNA_molecules.csv"))
        salmonella_inter = read_df(file_path=join(self.config['interactions_dir'], salmonella_dir, "matera_salmonella_interactions.csv"))

        salmonella_mrna, salmonella_srna, salmonella_inter = \
            ap.preprocess_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna, inter_data=salmonella_inter)
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
        # 4 - Klebsiella pneumoniae str. SGH10; KL1, ST23  (Goh 2024)
        klebsiella_dir = self.config[f'{self.klebsiella_nm}_dir']
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

        # 5 - Vibrio cholerae O1 biovar El Tor str. N16961 (NC_002505.1 and NC_002506.1)  (Huber 2022)
        vibrio_dir = self.config[f'{self.vibrio_nm}_dir']
        vibrio_mrna = read_df(file_path=join(self.config['rna_dir'], vibrio_dir, "huber_vibrio_all_mRNA_molecules.csv"))
        vibrio_srna = read_df(file_path=join(self.config['rna_dir'], vibrio_dir, "huber_vibrio_all_sRNA_molecules.csv"))
        vibrio_inter = read_df(file_path=join(self.config['interactions_dir'], vibrio_dir, "huber_vibrio_interactions.csv"))

        vibrio_mrna, vibrio_srna, vibrio_inter = \
            ap.preprocess_vibrio_inter(mrna_data=vibrio_mrna, srna_data=vibrio_srna, inter_data=vibrio_inter)
        vibrio_unq_inter, vibrio_sum, vibrio_srna, vibrio_mrna = \
            ap.analyze_vibrio_inter(mrna_data=vibrio_mrna, srna_data=vibrio_srna, inter_data=vibrio_inter)
        
        # 5.1 - update info
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

        # 6 - Pseudomonas aeruginosa PAO1  (Gebhardt 2023)
        pseudomonas_dir = self.config[f'{self.pseudomonas_nm}_dir']
        pseudomonas_mrna = read_df(file_path=join(self.config['rna_dir'], pseudomonas_dir, "gebhardt_pseudomonas_all_mRNA_molecules.csv"))
        pseudomonas_srna = read_df(file_path=join(self.config['rna_dir'], pseudomonas_dir, "gebhardt_pseudomonas_all_sRNA_molecules.csv"))
        pseudomonas_inter = read_df(file_path=join(self.config['interactions_dir'], pseudomonas_dir, "gebhardt_pseudomonas_interactions.csv"))

        pseudomonas_mrna, pseudomonas_srna, pseudomonas_inter = \
            ap.preprocess_pseudomonas_inter(mrna_data=pseudomonas_mrna, srna_data=pseudomonas_srna, inter_data=pseudomonas_inter)
        pseudomonas_unq_inter, pseudomonas_sum, pseudomonas_srna, pseudomonas_mrna = \
            ap.analyze_pseudomonas_inter(mrna_data=pseudomonas_mrna, srna_data=pseudomonas_srna, inter_data=pseudomonas_inter)
        
        # 6.1 - update info
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

        # ---------------------------   create and dump a summary table of interaction and RNA data   ---------------------------

        if self.dump_rna_and_inter_data_summary:
            self.logger.info(f"Dumping a summary table of all strains data")
            summary_df = pd.concat([k12_sum, epec_sum, salmo_sum, vibrio_sum, klebsiella_sum, pseudomonas_sum], ignore_index=True)
            summary_df['No.'] = list(range(1, len(summary_df) + 1))
            _rename = {
                'strain_short': 'Bacterial Strain', 
                'unq_pos_inter': 'Unique sRNA-mRNA Interactions', 
                'unique_sRNAs': 'Interacting sRNA', 
                'unique_targets': 'Interacting mRNA', 
                'total_sRNAs': 'All sRNA', 
                'total_mRNAs': 'All mRNA',
                'strain': 'Full Strain Name'
            }
            summary_df = summary_df.rename(columns=_rename)[['No.'] + list(_rename.values())]
            write_df(df=summary_df, file_path=join(self.config['interactions_dir'], 'inter_and_rna_data_summary.csv'))

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
    
    def _generate_clean_rna_fasta_files(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        for strain in self.strains:
            # 1 - load RNA fasta files
            _path = join(self.config['rna_dir'], self.config[f'{strain}_dir'], "rna_fasta")
            srna_fasta = load_fasta(file_path=join(_path, f"{strain}_sRNAs.fasta"))
            mrna_fasta = load_fasta(file_path=join(_path, f"{strain}_mRNAs.fasta"))

            srna_fasta = srna_fasta[[c for c in srna_fasta.columns if c !='description']]
            mrna_fasta = mrna_fasta[[c for c in mrna_fasta.columns if c !='description']]

            # 2 - load processed RNA data
            srna_df = self.strains_data[strain]['all_srna']
            mrna_df = self.strains_data[strain]['all_mrna']

            # 3 - clean
            srna_fasta_clean = self._clean_rna_fasta(strain=strain, rna_type='sRNA', rna_df=srna_df, rna_fasta=srna_fasta)
            mrna_fasta_clean = self._clean_rna_fasta(strain=strain, rna_type='mRNA', rna_df=mrna_df, rna_fasta=mrna_fasta)

            srna_fasta_clean = srna_fasta_clean[['cleaned_header','seq']]
            mrna_fasta_clean = mrna_fasta_clean[['cleaned_header','seq']]
            assert not pd.isnull(srna_fasta_clean).y(), f"{strain} - missing sequences in sRNA fasta after cleaning"
            assert not pd.isnull(mrna_fasta_clean).values.any(), f"{strain} - missing sequences in mRNA fasta after cleaning"

            # 4 - write clean fasta files
            write_fasta(df=srna_fasta_clean, out_path=join(_path, f"{strain}_sRNAs_clean.fasta"), header_col='cleaned_header')
            write_fasta(df=mrna_fasta_clean, out_path=join(_path, f"{strain}_mRNAs_clean.fasta"), header_col='cleaned_header')
    
    def _clean_rna_fasta(self, strain: str, rna_type: str, rna_df: pd.DataFrame, rna_fasta: pd.DataFrame) -> pd.DataFrame:
        # 1 - merge fasta with rna_df
        merged = self._merge_rna_df_n_fasta(rna_type=rna_type, rna_df=rna_df, rna_fasta=rna_fasta)
        assert len(merged) == len(merged[['header', 'seq']].drop_duplicates()), f"{strain} - duplications in {rna_type} fasta after merge"
        assert (merged[f'{rna_type}_sequence'] == merged['seq']).all(), f"{strain} - sequence mismatch in {rna_type} fasta after merge"
        
        # 2 - generate clean header for fasta
        merged['cleaned_header'] = list(map(lambda prev_header, acc, locus, nm: "|".join([prev_header.split('|')[0], acc, str(locus), str(nm)]), 
                                        merged['header'], 
                                        merged[f'{rna_type}_accession_id'],
                                        merged[f'{rna_type}_locus_tag'],
                                        merged[f'{rna_type}_name']
                                        ))
        fixed = merged[merged['cleaned_header'] != merged['header']].reset_index(drop=True)
        self.logger.info(f"{strain} - fixed {len(fixed)} {rna_type} fasta headers out of {len(merged)}")

        return merged

    def _merge_rna_df_n_fasta(self,  rna_type: str, rna_df: pd.DataFrame, rna_fasta: pd.DataFrame, rna_fasta_header_col: str = 'header') -> pd.DataFrame:
        # Parse fasta headers to extract accession, locus, name
        rna_fasta = gp.parse_header_to_acc_locus_and_name(
            df=rna_fasta,
            df_header_col=rna_fasta_header_col,
            acc_col=f'{rna_type}_accession_id_fasta',
            locus_col=f'{rna_type}_locus_tag_fasta',
            name_col=f'{rna_type}_name_fasta'
        )

        # Try merge by accession id
        merged = pd.merge(
            rna_df,
            rna_fasta.dropna(subset=[f'{rna_type}_accession_id_fasta']),
            how='left',
            left_on=f'{rna_type}_accession_id',
            right_on=f'{rna_type}_accession_id_fasta'
        )

        # For rows not matched, try locus tag
        not_matched = merged[pd.isnull(merged[rna_fasta_header_col])]
        if not not_matched.empty:
            rna_df_not_matched = not_matched[[c for c in rna_df.columns]]
            locus_merge = pd.merge(
                rna_df_not_matched,
                rna_fasta.dropna(subset=[f'{rna_type}_locus_tag_fasta']),
                how='left',
                left_on=f'{rna_type}_locus_tag',
                right_on=f'{rna_type}_locus_tag_fasta'
            )
            merged = pd.concat([merged[pd.notnull(merged[rna_fasta_header_col])], locus_merge], ignore_index=True)

        # For rows still not matched, try name
        not_matched = merged[pd.isnull(merged[rna_fasta_header_col])]
        if not not_matched.empty:
            rna_df_not_matched = not_matched[[c for c in rna_df.columns]]
            rna_fasta_w_lower_nm = rna_fasta.copy()
            rna_fasta_w_lower_nm[f'{rna_type}_lower_name_fasta'] = rna_fasta_w_lower_nm[f'{rna_type}_name_fasta'].str.lower()
            name_merge = pd.merge(
                rna_df_not_matched,
                rna_fasta_w_lower_nm.dropna(subset=[f'{rna_type}_lower_name_fasta']),
                how='left',
                left_on=f'{rna_type}_name',
                right_on=f'{rna_type}_lower_name_fasta'
            )
            merged = pd.concat([merged[pd.notnull(merged[rna_fasta_header_col])], name_merge], ignore_index=True)
    
        assert len(merged) == len(rna_df), f"mismatch in {rna_type} fasta and data"
        assert sum(pd.isnull(merged[rna_fasta_header_col])) == 0, f"missing sequences in {rna_type} fasta after all matching attempts"

        return merged
         
    def _load_proteins(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        for strain in self.strains:
            # 1 - load proteins
            proteins = load_fasta(file_path=join(self.config['proteins_dir'], f"{strain}_proteins.fasta"))
            proteins = proteins.rename(columns={'seq': 'protein_seq'})
            
            # 2 - preprocess
            proteins = gp.parse_header_to_acc_locus_and_name(df=proteins, df_header_col='header', acc_col=self.mrna_acc_col, locus_col='mRNA_locus_tag', name_col='mRNA_name')

            # 3 - validate
            assert sum(pd.isnull(proteins[self.mrna_acc_col])) == 0, f"missing accession ids in {strain} proteins"
            assert len(proteins[self.mrna_acc_col].unique()) == len(proteins), f"duplicate accession ids in {strain} proteins"
            assert len(set(proteins[self.mrna_acc_col]) - set(self.strains_data[strain]['all_mrna'][self.mrna_acc_col])) == 0, "invalid mRNA accession ids in proteins"
            assert sum(pd.isnull(proteins['protein_seq'])) == 0, f"missing protein sequences in {strain} proteins"
            
            # 4 - update info
            if strain not in self.strains_data:
                self.strains_data[strain] = {}
            self.strains_data[strain].update({
                'all_proteins': proteins
            })
    
    def _match_proteins_to_mrnas(self):
        for strain, data in self.strains_data.items():
            all_mrna_w = pd.merge(data['all_mrna'], data['all_proteins'][[self.mrna_acc_col, 'protein_seq']], how='left', on=self.mrna_acc_col)
            assert len(all_mrna_w) == len(data['all_mrna']), f"mRNA and protein data mismatch in {strain}"
            self.logger.info(f"{strain} --> {sum(pd.notnull(all_mrna_w['protein_seq']))} out of {len(all_mrna_w)} mRNAs with protein sequences ({round(sum(pd.notnull(all_mrna_w['protein_seq'])) / len(all_mrna_w) * 100, 2)}%)")
            
            data['all_mrna'] = all_mrna_w
    
    def _load_annotations(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        # ---------------------------   per dataset preprocessing   ---------------------------
        for strain in self.strains:
            self._load_uniprot_annotations(strain=strain)
            self._load_interproscan_annotations(strain=strain)
            # TODO: re-run eggnog with clean protein files
            # self._load_eggnog_annotations(strain=strain)
    
    def _load_uniprot_annotations(self, strain: str):
        _path = join(self.config['go_annotations_dir'], 'Curated')
        if os.path.exists(join(_path, f'{strain}.goa')):
            self.logger.info(f"loading curated annotations for {strain}")
            # 1 - load and preprocess
            annot_uniprot = load_goa(file_path=join(_path, f'{strain}.goa'))
            annot_map_uniprot_to_locus = read_df(file_path=join(_path, f'{strain}_idmapping.dat'))
            annot_map_uniprot_to_locus.columns = ['UniProt_ID', 'Database', 'Mapped_ID']
            curated_annot, c_locus_col = ap_annot.preprocess_curated_annot(strain, annot_uniprot, annot_map_uniprot_to_locus)
            # 2 - update info
            if strain not in self.strains_data:
                self.strains_data[strain] = {}
            self.strains_data[strain].update({
                "curated_annot": curated_annot,
                "curated_locus_col": c_locus_col
            })
    
    def _load_interproscan_annotations(self, strain: str):
        _path = join(self.config['go_annotations_dir'], 'InterProScan')
        if os.path.exists(join(_path, f'{strain}_proteins.fasta.json')):
            self.logger.info(f"loading InterProScan annotations for {strain}")
            # 1 - load and preprocess
            annot_interproscan = load_json(file_path=join(_path, f'{strain}_proteins.fasta.json'))
            interproscan_annot, i_header_col = ap_annot.preprocess_interproscan_annot(annot_interproscan)
            # 2 - update info
            if strain not in self.strains_data:
                self.strains_data[strain] = {}
            self.strains_data[strain].update({
                "interproscan_annot": interproscan_annot,
                "interproscan_header_col": i_header_col,
            })
    
    def _load_eggnog_annotations(self, strain: str):
        _path = join(self.config['go_annotations_dir'], 'EggNog')
        if os.path.exists(join(_path, f'{strain}.annotations')):
            self.logger.info(f"loading EggNog annotations for {strain}")
            # 1 - load and preprocess
            eggnog_annot_file=join(_path, f'{strain}.annotations')
            eggnog_annot, e_header_col = ap_annot.load_and_preprocess_eggnog_annot(eggnog_annot_file)
            # 2 - update info
            if strain not in self.strains_data:
                self.strains_data[strain] = {}
            self.strains_data[strain].update({
                "eggnog_annot": eggnog_annot,
                "eggnog_header_col": e_header_col
            })

    def _match_annotations_to_mrnas(self):
        for strain, data in self.strains_data.items():
            if 'curated_annot' in data:
                data['all_mrna_w_curated_annot'] = ap_annot.annotate_mrnas_w_curated_annt(strain, data)
            if 'interproscan_annot' in data:
                data['all_mrna_w_ips_annot'] = ap_annot.annotate_mrnas_w_interproscan_annt(strain, data)
            if 'eggnog_annot' in data:
                data['all_mrna_w_eggnog_annot'] = ap_annot.annotate_mrnas_w_eggnog_annt(strain, data)

    def _load_clustering_data(self, load_from_pickle: bool = False):
        # 1 - load sRNA and mRNA clustering
        if load_from_pickle:
            srna_clstr_dict = self._load_clustering_pickle(seq_type=self.srna_seq_type)
            mrna_clstr_dict = self._load_clustering_pickle(seq_type=self.protein_seq_type)
        else:
            srna_clstr_dict = self._load_n_preprocess_bacteria_pairs_clustering(seq_type=self.srna_seq_type)
            mrna_clstr_dict = self._load_n_preprocess_bacteria_pairs_clustering(seq_type=self.protein_seq_type)

        # 2 - save clustering
        self.clustering_data['srna'] = srna_clstr_dict
        self.clustering_data['mrna'] = mrna_clstr_dict
    
    def _get_clustering_path_n_nm(self, seq_type: str) -> tuple:
        _dir = self.clustering_config[f'{seq_type.lower()}_dir']
        _path = join(self.config['clustering_dir'], seq_type, _dir)
        f_name = f"{seq_type}_pairs_clustering"
        return _path, _dir, f_name
    
    def _load_clustering_pickle(self, seq_type: str) -> Dict[tuple, pd.DataFrame]:
        _path, _dir, f_name = self._get_clustering_path_n_nm(seq_type=seq_type)
        self.logger.info(f"loading {seq_type} clustering pickle ({_dir})")
        with open(join(_path, f'{f_name}.pickle'), 'rb') as handle:
            pairs_clustering = pickle.load(handle)
        return pairs_clustering

    def _load_n_preprocess_bacteria_pairs_clustering(self, seq_type: str) -> Dict[tuple, pd.DataFrame]:
        _path, _dir, f_name = self._get_clustering_path_n_nm(seq_type=seq_type)
        
        # ---------------------------   PAIRS preprocessing   ---------------------------
        self.logger.info(f"PAIRS preprocessing --> loading {seq_type} clustering data from {_dir}")
        pairs_clustering = {}
        for b1, b2 in itertools.combinations(self.strains, 2):
            # 1 - identify clustering files
            file_1_to_2 = [f for f in os.listdir(_path) if re.match(f"(.*?){b1}-{b2}.fasta.clstr$", f)][0]
            file_2_to_1 = [f for f in os.listdir(_path) if re.match(f"(.*?){b2}-{b1}.fasta.clstr$", f)][0]

            # 2 - load and parse the clustering files
            clstr_df_1_to_2 = self._load_n_parse_clstr_file(clstr_file_path=join(_path, file_1_to_2), seq_type=seq_type)
            clstr_df_2_to_1 = self._load_n_parse_clstr_file(clstr_file_path=join(_path, file_2_to_1), seq_type=seq_type)
            
            # 3 - filter invalid matches from the clustering data 
            clstr_df_1_to_2 = self._filter_invalid_matches(clstr_df_1_to_2, f"{b1} to {b2}", seq_type)
            clstr_df_2_to_1 = self._filter_invalid_matches(clstr_df_2_to_1, f"{b2} to {b1}", seq_type)

            # 4 - add to the pairs_clustering dictionary
            pairs_clustering[(b1, b2)] = clstr_df_1_to_2
            pairs_clustering[(b2, b1)] = clstr_df_2_to_1
        
        # 5 - save the pairs_clustering data
        with open(join(_path, f'{f_name}.pickle'), 'wb') as handle:
            pickle.dump(pairs_clustering, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
            'salmonella': self.salmonella_nm,
            'klebsiella': self.klebsiella_nm,
            'cholerae': self.vibrio_nm,
            'pseudomonas': self.pseudomonas_nm
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

        # 4 - parse each entry
        _df = pd.DataFrame(list(map(lambda x: x.split(" "), clstr_df['entry'])))
        if _df.shape[1] == 3:  # case of no matches (i.e.,  no footer)
            _df[3] = None

        # 5 - split the entries into columns and preprocess
        clstr_df[['counter_len', 'header', col_is_rep, 'footer']] = _df
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

        return clstr_df
    
    def _filter_invalid_matches(self, clstr_df: pd.DataFrame, clstr_nm: str, seq_type: str, debug: bool = False,
                                col_cluster_id: str = 'cluster_id', col_name: str = 'rna_name', col_seq_length: str = 'seq_length', col_is_rep: str = 'is_representative', col_similarity_score: str = 'similarity_score') -> pd.DataFrame:
        """Filter invalid matches from the clustering df.
        Keep representatives. For matches, keep only if the match is valid (see logic in self._is_valid_match).
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
        self.logger.info(f"{seq_type} clustering {clstr_nm} ---> filtered {len(clstr_df) - len(filtered_df)} invalid matches")
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

