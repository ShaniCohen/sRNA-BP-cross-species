from typing import Dict
import pandas as pd
import numpy as np
from os.path import join
from analysis.data_preprocessing import data_prepro as ap
from analysis.utils.utils_general import read_df, write_df

import logging
logger = logging.getLogger(__name__)


def load_rna_n_inter_data(conf: Dict[str, str], ecoli_k12_nm: str, ecoli_epec_nm: str, salmonella_nm: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    input_data_path = conf['input_data_path'][conf['machine']]
    inter_data_path = join(input_data_path, 'interactions')
    rna_data_path = join(input_data_path, 'rna')
    data = {}

    # ---------------------------   per dataset preprocessing   ---------------------------
    # 1 - Escherichia coli K12 MG1655
    #     srna_eco: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
    #     mrna_eco: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
    k12_dir = "Escherichia_coli_K12_MG1655"
    k12_mrna = read_df(file_path=join(rna_data_path, k12_dir, "mrna_eco.csv"))
    k12_srna = read_df(file_path=join(rna_data_path, k12_dir, "srna_eco.csv"))
    k12_inter = read_df(file_path=join(inter_data_path, k12_dir, 'sInterBase_interactions_post_processing.csv'))

    k12_mrna, k12_srna, k12_inter = \
        ap.preprocess_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna, inter_data=k12_inter)
    k12_unq_inter, k12_sum, k12_srna, k12_mrna = ap.analyze_ecoli_k12_inter(mrna_data=k12_mrna, srna_data=k12_srna,
                                                                            inter_data=k12_inter)
    # 1.1 - update info
    data.update({
        ecoli_k12_nm: {
            'all_mrna': k12_mrna,
            'all_srna': k12_srna,
            'unq_inter': k12_unq_inter,
            'all_inter': k12_inter,
            'all_srna_acc_col': 'EcoCyc_accession_id',
            'all_mrna_acc_col': 'EcoCyc_accession_id',
            'all_inter_srna_acc_col': 'sRNA_accession_id_Eco',
            'all_inter_mrna_acc_col': 'mRNA_accession_id_Eco'
        }
    })

    # 2 - Escherichia coli EPEC E2348/69
    epec_dir = 'Mizrahi_2021_EPEC'
    epec_mrna = read_df(file_path=join(rna_data_path, epec_dir, "mizrahi_epec_all_mRNA_molecules.csv"))
    epec_srna = read_df(file_path=join(rna_data_path, epec_dir, "mizrahi_epec_all_sRNA_molecules.csv"))
    epec_inter = read_df(file_path=join(inter_data_path, epec_dir, "mizrahi_epec_interactions.csv"))

    epec_mrna, epec_srna, epec_inter = \
        ap.preprocess_ecoli_epec_inter(mrna_data=epec_mrna, srna_data=epec_srna, inter_data=epec_inter)
    epec_unq_inter, epec_sum, epec_srna, epec_mrna = \
        ap.analyze_ecoli_epec_inter(mrna_data=epec_mrna, srna_data=epec_srna, inter_data=epec_inter)
    # 2.1 - update info
    data.update({
        ecoli_epec_nm: {
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
    salmonella_mrna = read_df(file_path=join(rna_data_path, salmonella_dir, "matera_salmonella_all_mRNA_molecules.csv"))
    salmonella_srna = read_df(file_path=join(rna_data_path, salmonella_dir, "matera_salmonella_all_sRNA_molecules.csv"))
    salmonella_inter = read_df(file_path=join(inter_data_path, salmonella_dir, "matera_salmonella_interactions.csv"))

    salmonella_mrna, salmonella_srna, salmonella_inter = \
        ap.preprocess_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna,
                                       inter_data=salmonella_inter)
    salmo_unq_inter, salmo_sum, salmo_srna, salmo_mrna = \
        ap.analyze_salmonella_inter(mrna_data=salmonella_mrna, srna_data=salmonella_srna, inter_data=salmonella_inter)
    # 3.1 - update info
    data.update({
        salmonella_nm: {
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
    return data


def align_rna_n_inter_data(data: Dict[str, Dict[str, pd.DataFrame]], srna_acc: str = 'sRNA_accession_id', mrna_acc: str = 'mRNA_accession_id') -> Dict[str, Dict[str, pd.DataFrame]]:
    # 1 - align RNA accession ids for all datasets
    for strain_data in data.values():
        # 1.1 - all sRNA
        strain_data['all_srna'] = strain_data['all_srna'].rename(columns={strain_data['all_srna_acc_col']: srna_acc})
        # 1.2 - all mRNA
        strain_data['all_mrna'] = strain_data['all_mrna'].rename(columns={strain_data['all_mrna_acc_col']: mrna_acc})
        # 1.3 - all interactions
        strain_data['all_inter'] = strain_data['all_inter'].rename(columns={strain_data['all_inter_srna_acc_col']: srna_acc})
        strain_data['all_inter'] = strain_data['all_inter'].rename(columns={strain_data['all_inter_mrna_acc_col']: mrna_acc})
        # 1.4 - unique interactions
        strain_data['unq_inter'] = strain_data['unq_inter'].rename(columns={strain_data['all_inter_srna_acc_col']: srna_acc})
        strain_data['unq_inter'] = strain_data['unq_inter'].rename(columns={strain_data['all_inter_mrna_acc_col']: mrna_acc})

    return data