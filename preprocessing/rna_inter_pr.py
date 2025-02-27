import pandas as pd
import numpy as np
import itertools
import re
from utils.general import order_df
from typing import Dict, List, Tuple, Set
import logging
logger = logging.getLogger(__name__)


def get_all_srna_mrna_pairs(all_srnas: List[str], all_mrnas: List[str], srna_out_col: str = 'sRNA',
                            mrna_out_col: str = 'mRNA') -> pd.DataFrame:
    assert len(all_srnas) == len(set(all_srnas)), "all_srnas not unique"
    assert len(all_mrnas) == len(set(all_mrnas)), "all_mrnas not unique"
    df = pd.DataFrame(list(itertools.product(all_srnas, all_mrnas)), columns=[srna_out_col, mrna_out_col])
    return df


def add_accessions_to_rna_names_in_inter_df(
        inder_df_1: pd.DataFrame, srna_nm_col_1: str, mrna_nm_col_1: str, label_col_1: str,
        srna_acc_df_2: pd.DataFrame, srna_nm_col_2: str, srna_acc_col_2: str,
        mrna_acc_df_3: pd.DataFrame, mrna_nm_col_3: str, mrna_acc_col_3: str,
        remove_mrna_names_with_multi_accessions: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug(f"add_accessions_to_rna_names_in_inter_df --> remove = {remove_mrna_names_with_multi_accessions}")

    out_col_multi_mrna_acc = 'is_mRNA_with_multi_acc'
    out_cols = [srna_nm_col_2, srna_acc_col_2, mrna_nm_col_3, mrna_acc_col_3, label_col_1, out_col_multi_mrna_acc]

    # 1 - keep only relevant columns in interactions df
    _len = len(inder_df_1)
    inder_df_1 = inder_df_1[[srna_nm_col_1, mrna_nm_col_1, label_col_1]]

    # 2 - add sRNAs accessions
    inder_df_1 = pd.merge(left=inder_df_1, right=srna_acc_df_2[[srna_nm_col_2, srna_acc_col_2]],
                          left_on=srna_nm_col_1, right_on=srna_nm_col_2, how='left')
    # 3 - find duplicated mRNA names
    group_by_cols = [mrna_nm_col_3]
    mrna_acc_df_3['row_count'] = 1
    dup_mrna_nm_df = mrna_acc_df_3.groupby(group_by_cols).count().reset_index()
    dup_mrna_nm_lst = sorted(dup_mrna_nm_df[dup_mrna_nm_df['row_count'] > 1][mrna_nm_col_3])
    dup_mrna_nm_df = mrna_acc_df_3[mrna_acc_df_3[mrna_nm_col_3].isin(dup_mrna_nm_lst)].sort_values(by=mrna_nm_col_3).reset_index(drop=True)
    logger.info(f"{len(dup_mrna_nm_lst)} mRNA names with multiple accessions: {dup_mrna_nm_lst}")
    # 3.1 - find interactions of mRNA names with duplicated accessions
    _len = len(inder_df_1)
    inder_df_1[out_col_multi_mrna_acc] = inder_df_1[mrna_nm_col_1].isin(dup_mrna_nm_lst)
    logger.info(f"out of {_len} interactions, "
                f"{sum(inder_df_1[out_col_multi_mrna_acc])} interactions of mRNA names with multiple accessions")

    # 4 - add mRNAs accessions
    if remove_mrna_names_with_multi_accessions:
        logger.info(f"remove interactions of mRNA names with multiple accessions")
        inder_df_1 = inder_df_1[~inder_df_1[out_col_multi_mrna_acc]].reset_index(drop=True)
    else:
        logger.info(f"keep mRNAs with duplicated names")
    inder_df_1 = pd.merge(left=inder_df_1, right=mrna_acc_df_3[[mrna_nm_col_3, mrna_acc_col_3]],
                          left_on=mrna_nm_col_1, right_on=mrna_nm_col_3, how='left')
    inder_df_1 = inder_df_1[out_cols]
    logger.info(f"{_len} interactions before, {len(inder_df_1)} interactions after merge")

    return inder_df_1


def find_nm1_interactions_of_common_srna_mrna(
        nm1: str, inter_df1: pd.DataFrame, srna_nm_col_1: str, mrna_nm_col_1: str, label_col_1: str,
        nm2: str, inter_df2: pd.DataFrame, srna_nm_col_2: str, mrna_nm_col_2: str, label_col_2: str,
        common_srnas: List[str], common_mrnas: List[str]) -> pd.DataFrame:

    logger.debug(f"find_nm1_interactions_of_common_srna_mrna: 1 = {nm1} vs. 2 = {nm2}")
    logger.debug(f"all pairs of common sRNA and mRNA = {len(common_srnas)*len(common_mrnas)}")
    # 1 - find nm1 interactions sRNAs and mRNAs that are common to nm1 and nm2
    mask_common_srna = inter_df1[srna_nm_col_1].isin(common_srnas)
    mask_common_mrna = inter_df1[mrna_nm_col_1].isin(common_mrnas)
    mask_common = mask_common_srna & mask_common_mrna
    inter_df1_common = inter_df1[mask_common].reset_index(drop=True)
    _len = len(inter_df1_common)
    logger.info(f"{_len} interactions of {nm1} with common sRNA, mRNA of {nm2}")

    # 2 - check if nm1 interactions of common already exist in nm2 interactions
    if label_col_1 == label_col_2:
        label_col_2_for_merge = f"{label_col_2}_{nm2}"
        inter_df2[label_col_2_for_merge] = inter_df2[label_col_2]
    else:
        label_col_2_for_merge = label_col_2

    inter_df1_common = pd.merge(left=inter_df1_common, right=inter_df2[[srna_nm_col_2, mrna_nm_col_2, label_col_2_for_merge]],
                                left_on=[srna_nm_col_1, mrna_nm_col_1], right_on=[srna_nm_col_2, mrna_nm_col_2], how='left')
    assert len(inter_df1_common) == _len, "duplications post merge"

    # 2.1 - exist in nm2 interactions (with or without conflict)
    mask_in_nm2 = pd.notnull(inter_df1_common[label_col_2_for_merge])
    mask_in_nm2_and_label_conflict = pd.Series(list(map(lambda l1, l2: pd.notnull(l2) and l1 != l2,
                                                        inter_df1_common[label_col_1], inter_df1_common[label_col_2_for_merge])))
    assert sum(mask_in_nm2_and_label_conflict) == 0, f"labels conflicts between {nm1} and {nm2} interactions"
    logger.info(f"{sum(mask_in_nm2)} out of {len(inter_df1_common)} interactions (common RNAs) of {nm1} exist in {nm2}")

    # 2.2 - filter interactions exist in nm2
    inter_df1_common = inter_df1_common[~mask_in_nm2].reset_index(drop=True)
    logger.info(f"{len(inter_df1_common)} interactions (common RNAs) of {nm1} post filtering")

    return inter_df1_common


def check_interacting_rna_names_included_in_all_rnas(inter_df: pd.DataFrame, inter_rna_nm_col: str,
                                                     all_rna: pd.DataFrame, all_rna_nm_col: str) -> bool:
    inter_rnas = sorted(set(inter_df[inter_rna_nm_col]))
    all_rnas = sorted(set(all_rna[all_rna_nm_col]))
    passed: bool = set(inter_rnas) <= set(all_rnas)
    if not passed:
        logger.error(f"interacting {inter_rna_nm_col} = {sorted(set(inter_rnas) - set(all_rnas))} missing in all rnas")
    return passed


def find_srna_mrna_intersection(nm1: str, all_srna1: pd.DataFrame, all_mrna1: pd.DataFrame, srna_nm_col_1: str, mrna_nm_col_1: str,
                                nm2: str, all_srna2: pd.DataFrame, all_mrna2: pd.DataFrame, srna_nm_col_2: str, mrna_nm_col_2: str,
                                dataset_out_col: str = 'dataset') \
        -> Tuple[List[str], List[str], pd.DataFrame]:
    logger.info(f"find_srna_mrna_intersection: {nm1} vs. {nm2}")
    srna1 = sorted(set(all_srna1[srna_nm_col_1]))
    mrna1 = sorted(set(all_mrna1[mrna_nm_col_1]))
    # logger.info(f"{nm1}: {len(all_srna1)} sRNAs ({len(srna1)} unique names), {len(all_mrna1)} mRNAs ({len(mrna1)} unique names)")
    srna2 = sorted(set(all_srna2[srna_nm_col_2]))
    mrna2 = sorted(set(all_mrna2[mrna_nm_col_2]))
    # logger.info(f"{nm2}: {len(all_srna2)} sRNAs ({len(srna2)} unique names), {len(all_mrna2)} mRNAs ({len(mrna2)} unique names)")

    srna_1_and_2 = sorted(np.intersect1d(srna1, srna2))
    mrna_1_and_2 = sorted(np.intersect1d(mrna1, mrna2))
    logger.info(f"{len(srna_1_and_2)} common sRNA names, {len(mrna_1_and_2)} common mRNA names")
    srna_1_not_2 = sorted(set(srna1) - set(srna_1_and_2))
    mrna_1_not_2 = sorted(set(mrna1) - set(mrna_1_and_2))
    srna_2_not_1 = sorted(set(srna2) - set(srna_1_and_2))
    mrna_2_not_1 = sorted(set(mrna2) - set(mrna_1_and_2))

    records = [
        {dataset_out_col: f'{nm1}_all', 'unique_sRNA_acc_ids': len(all_srna1), 'unique_sRNA_names': len(srna1),
         'unique_mRNA_acc_ids': len(all_mrna1), 'unique_mRNA_names': len(mrna1)},
        {dataset_out_col: f'{nm2}_all', 'unique_sRNA_acc_ids': len(all_srna2), 'unique_sRNA_names': len(srna2),
         'unique_mRNA_acc_ids': len(all_mrna2), 'unique_mRNA_names': len(mrna2)},

        {dataset_out_col: f'{nm1}_only', 'unique_sRNA_acc_ids': 0, 'unique_sRNA_names': len(srna_1_not_2),
         'unique_mRNA_acc_ids': 0, 'unique_mRNA_names': len(mrna_1_not_2)},
        {dataset_out_col: f'{nm2}_only', 'unique_sRNA_acc_ids': 0, 'unique_sRNA_names': len(srna_2_not_1),
         'unique_mRNA_acc_ids': 0, 'unique_mRNA_names': len(mrna_2_not_1)},

        {dataset_out_col: 'intersection', 'unique_sRNA_acc_ids': 0, 'unique_sRNA_names': len(srna_1_and_2),
         'unique_mRNA_acc_ids': 0, 'unique_mRNA_names': len(mrna_1_and_2)}
    ]
    df = pd.DataFrame(records)

    return srna_1_and_2, mrna_1_and_2, df


def find_interactions_intersection(nm1: str, df1: pd.DataFrame, strain_col1: str, srna_nm_col_1: str, target_nm_col_1: str,
                                   nm2: str, df2: pd.DataFrame, strain_col2: str, srna_nm_col_2: str, target_nm_col_2: str,
                                   dataset_out_col: str = 'dataset', cols_to_ignore: List[str] = None) -> Dict[str, pd.DataFrame]:
    logger.debug(f"find_interactions_intersection: {nm1} vs. {nm2}")
    df1[srna_nm_col_1] = df1[srna_nm_col_1].apply(lambda x: x.lower())
    df1[target_nm_col_1] = df1[target_nm_col_1].apply(lambda x: x.lower())

    df2[srna_nm_col_2] = df2[srna_nm_col_2].apply(lambda x: x.lower())
    df2[target_nm_col_2] = df2[target_nm_col_2].apply(lambda x: x.lower())

    out_dfs = {}
    for st1 in sorted(set(df1[strain_col1])):
        for st2 in sorted(set(df2[strain_col2])):
            logger.debug(f"comparing {st1} and {st2}")
            curr_df1 = df1[df1[strain_col1] == st1]
            curr_df2 = df2[df2[strain_col2] == st2]

            srna1 = sorted(set(curr_df1[srna_nm_col_1]))
            srna2 = sorted(set(curr_df2[srna_nm_col_2]))
            srna_1_and_2 = sorted(np.intersect1d(srna1, srna2))
            srna_1_not_2 = sorted(set(srna1) - set(srna_1_and_2))
            srna_2_not_1 = sorted(set(srna2) - set(srna_1_and_2))


            targets1 = sorted(set(curr_df1[target_nm_col_1]))
            targets2 = sorted(set(curr_df2[target_nm_col_2]))
            targets_1_and_2 = sorted(np.intersect1d(targets1, targets2))
            targets_1_not_2 = sorted(set(targets1) - set(targets_1_and_2))
            targets_2_not_1 = sorted(set(targets2) - set(targets_1_and_2))

            inter1 = set(zip(curr_df1[srna_nm_col_1], curr_df1[target_nm_col_1]))
            inter2 = set(zip(curr_df2[srna_nm_col_2], curr_df2[target_nm_col_2]))
            inter_1_and_2 = sorted(inter1 - (inter1 - inter2))
            inter_1_not_2 = sorted(inter1 - set(inter_1_and_2))
            inter_2_not_1 = sorted(inter2 - set(inter_1_and_2))

            records = [
                {dataset_out_col: f"{nm1}_all", 'strain': st1, 'unique_interacting_sRNA_names': len(srna1), 'unique_interacting_target_names': len(targets1), 'unique_interactions': len(inter1)},
                {dataset_out_col: f"{nm2}_all", 'strain': st2, 'unique_interacting_sRNA_names': len(srna2), 'unique_interacting_target_names': len(targets2), 'unique_interactions': len(inter2)},

                {dataset_out_col: f"{nm1}_only", 'strain': st1, 'unique_interacting_sRNA_names': len(srna_1_not_2), 'unique_interacting_target_names': len(targets_1_not_2), 'unique_interactions': len(inter_1_not_2)},
                {dataset_out_col: f"{nm2}_only", 'strain': st2, 'unique_interacting_sRNA_names': len(srna_2_not_1), 'unique_interacting_target_names': len(targets_2_not_1), 'unique_interactions': len(inter_2_not_1)},


                {dataset_out_col: 'intersection', 'strain': None, 'unique_interacting_sRNA_names': len(srna_1_and_2), 'unique_interacting_target_names': len(targets_1_and_2), 'unique_interactions': len(inter_1_and_2)}
            ]
            df = pd.DataFrame(records)
            df['num'] = np.arange(1, len(df)+1)
            if cols_to_ignore is not None:
                df = df[[c for c in df.columns.values if c not in cols_to_ignore]]

            st1_nm = st1.replace('/', ' ').replace(':', ' ')
            st2_nm = st2.replace('/', ' ').replace(':', ' ')

            out_dfs.update({f"{nm1}_{st1_nm}_vs_{nm2}_{st2_nm}": df})

    return out_dfs


def _set_agnodice_target(rna_nm: str, rna_ncbi_id: str, rna_biocyc_id: str) -> str:
    if pd.notnull(rna_nm) and type(rna_nm) == str:
        tar = rna_nm
    elif pd.notnull(rna_ncbi_id) and type(rna_ncbi_id) == str:
        tar = rna_ncbi_id
    elif pd.notnull(rna_biocyc_id) and type(rna_biocyc_id) == str:
        tar = rna_biocyc_id
    else:
        logger.error("no rna info")
        tar = None
    return tar


def analyze_agnodice_inter(inter_data: pd.DataFrame, out_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    "Escherichia coli str. K-12 substr. MG1655",
    "Escherichia coli O127:H6 str. E2348/69",
    "Escherichia coli O157:H7 str. Sakai",
    "Escherichia coli O157:H7 str. EDL933",
    "Salmonella enterica subsp. enterica serovar Typhimurium str. SL1344",
    "Vibrio cholerae C6706"

    :param inter_data:
    :return:
    """
    logger.info(f"agnodice - interactions: {len(inter_data)}")
    # logger.info(f"unique interactions (sRNA-mRNA-rbp): {len(data.groupby(['microbe_strain_name', 'srna_name', 'rna_name', 'rbp']))}")
    # logger.info(f"unique interactions (sRNA-mRNA-experimental_method_group): {len(data.groupby(['microbe_strain_name', 'srna_name', 'rna_name', 'experimental_method_group']))}")
    # logger.info(f"unique interactions (sRNA-mRNA-type_of_regulation): {len(data.groupby(['microbe_strain_name', 'srna_name', 'rna_name', 'type_of_regulation']))}")
    # logger.info(f"unique interactions (sRNA-mRNA): {len(data.groupby(['microbe_strain_name', 'srna_name', 'rna_name']))}")

    mask_srna_id = pd.notnull(inter_data['srna_name'])
    mask_rna_id = pd.notnull(inter_data['rna_name']) | pd.notnull(inter_data['rna_ncbi_id']) | pd.notnull(inter_data['rna_biocyc_id'])

    # -------------- analysis
    """
    data['count'] = 1
    data['target'] = list(map(_set_agnodice_target, data['rna_name'], data['rna_ncbi_id'], pd.notnull(data['rna_biocyc_id'])))

    g_cols = ['microbe_strain_name', 'experimental_method_name', 'experimental_method_group']
    df_exp = data[g_cols + ['count']].groupby(g_cols).count().reset_index()
    df_exp = df_exp.sort_values(by=['microbe_strain_name', 'count'], ascending=False).reset_index(drop=True)

    g_cols = ['microbe_strain_name', 'srna_name', 'target']
    unq_inter = data.groupby(g_cols, as_index=False).count()
    unq_inter = unq_inter[g_cols + ['count']].reset_index(drop=True)

    df_sum = unq_inter.groupby(['microbe_strain_name']).agg(
        unique_sRNAs=('srna_name', lambda x: len(set(x))),
        unique_targets=('target', lambda x: len(set(x))),
        unq_pos_inter=('count', 'count'),
        unq_neg_inter=('count', lambda x: 0),
        unq_inter=('count', 'count'),
        all_inter=('count', 'sum')
    ).reset_index(drop=False)

    df_sum = df_sum.sort_values(by=['unq_inter'], ascending=False).reset_index(drop=True)
    
    write_df(df=ag_exp, file_path=join(out_path, "ag_exp.csv"))
    write_df(df=ag_sum, file_path=join(out_path, "ag_sum.csv"))
    """

    # -------------- filters
    # 1 - strains
    strains = [
        "Escherichia coli str. K-12 substr. MG1655",
        "Escherichia coli O127:H6 str. E2348/69",
        "Escherichia coli O157:H7 str. Sakai",
        "Escherichia coli O157:H7 str. EDL933",
        "Salmonella enterica subsp. enterica serovar Typhimurium str. SL1344",
        "Vibrio cholerae C6706"
    ]
    mask_strain = pd.Series([x in strains for x in inter_data['microbe_strain_name']])

    # 2 - experimental_method_group
    mask_exp_met_grp = inter_data['experimental_method_group'] == "Direct"

    # 3 - filer
    mask_keep = mask_strain & mask_exp_met_grp
    inter_data = inter_data[mask_keep].reset_index(drop=True)

    # -------------- complete cols
    inter_data['count'] = 1
    inter_data['target'] = list(map(_set_agnodice_target, inter_data['rna_name'], inter_data['rna_ncbi_id'], pd.notnull(inter_data['rna_biocyc_id'])))
    inter_data = inter_data[pd.notnull(inter_data['target'])].reset_index(drop=True)

    # -------------- unique interactions intersection
    g_cols = ['microbe_strain_name', 'srna_name', 'target']
    unq_inter = inter_data[g_cols + ['count']].groupby(g_cols, as_index=False).count().reset_index(drop=True)

    # -------------- summary
    strain_col = 'microbe_strain_name'
    df_sum = unq_inter.groupby([strain_col]).agg(
        dataset=('count', lambda x: 'Agnodice'),
        unique_sRNAs=('srna_name', lambda x: len(set(x))),
        unique_targets=('target', lambda x: len(set(x))),
        unq_pos_inter=('count', 'count'),
        unq_neg_inter=('count', lambda x: 0),
        unq_inter=('count', 'count'),
        all_inter=('count', 'sum')
    ).reset_index(drop=False)

    df_sum = df_sum.sort_values(by=['unq_inter'], ascending=False).reset_index(drop=True)
    df_sum = df_sum.rename(columns={strain_col: 'strain'})

    return unq_inter, df_sum


def analyze_target_rna3_inter(inter_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    "Escherichia coli str. K-12 substr. MG1655"                          (4162) srna = 61
    "Salmonella enterica subsp. enterica serovar Typhimurium str. SL1344"  (34) srna = 2
    "Salmonella enterica subsp. enterica serovar Typhimurium str. 14028S"  (11) srna = 1
    "Salmonella enterica subsp. enterica serovar Typhimurium str. LT2"      (9) srna = 1
    "Vibrio cholerae O1 biovar El Tor str. N16961"                         (17) srna = 2

    :param inter_data:
    :return:
    """
    logger.info(f"target_rna3 - interactions: {len(inter_data)}")
    # logger.info(f"unique interactions: {len(inter_data.groupby(['Genome', 'sRNA', 'Target']))}")

    # -------------- filters
    # 1 - strains
    strains = [
        "Escherichia coli str. K-12 substr. MG1655",
        "Salmonella enterica subsp. enterica serovar Typhimurium str. SL1344",
        "Salmonella enterica subsp. enterica serovar Typhimurium str. 14028S",
        "Salmonella enterica subsp. enterica serovar Typhimurium str. LT2",
        "Vibrio cholerae O1 biovar El Tor str. N16961"
    ]
    mask_strain = pd.Series([x in strains for x in inter_data['Genome']])

    # 2 - filer
    mask_keep = mask_strain
    inter_data = inter_data[mask_keep].reset_index(drop=True)

    # -------------- complete cols
    inter_data['count'] = 1

    # -------------- unique interactions intersection
    g_cols = ['Genome', 'sRNA Accession', 'sRNA', 'Target Accession', 'Target', 'Evinced Interaction']
    unq_inter = inter_data[g_cols + ['count']].groupby(g_cols, as_index=False).count().reset_index(drop=True)

    # -------------- summary
    strain_col = 'Genome'
    df_sum = unq_inter.groupby([strain_col]).agg(
        dataset=('count', lambda x: 'TargetRNA3'),
        unique_sRNAs=('sRNA', lambda x: len(set(x))),
        unique_targets=('Target', lambda x: len(set(x))),
        unq_pos_inter=('Evinced Interaction', 'sum'),
        unq_neg_inter=('Evinced Interaction', lambda x: sum(x == 0)),
        unq_inter=('Evinced Interaction', 'count'),
        all_inter=('count', 'sum')
    ).reset_index(drop=False)

    df_sum = df_sum.sort_values(by='unq_inter', ascending=False).reset_index(drop=True)
    df_sum = df_sum.rename(columns={strain_col: 'strain'})

    return unq_inter, df_sum


def analyze_salmonella_inter(mrna_data: pd.DataFrame, srna_data: pd.DataFrame, inter_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1

    :param mrna_data:
    :param srna_data:
    :param inter_data:
    :return:
    """
    srna_acc_col = 'sRNA_accession_id'
    srna_nm_col = 'sRNA'
    mrna_acc_col = 'mRNA_accession_id'
    mrna_nm_col = 'mRNA'
    # --------------  validations  --------------
    # 1 - check sRNA acc is mapped to a single sRNA name
    srna_map = inter_data[[srna_acc_col, srna_nm_col]].groupby([srna_acc_col]).agg(
        sRNA_count=(srna_nm_col, lambda x: len(set(x))),
        sRNA_names=(srna_nm_col, lambda x: sorted(set(x)))
    ).reset_index(drop=False)
    srna_map = srna_map.sort_values(by='sRNA_count', ascending=False).reset_index(drop=True)
    assert list(srna_map['sRNA_count'])[0] == 1

    # 2 - check mRNA acc is mapped to a single mRNA name
    mrna_map = inter_data[[mrna_acc_col, mrna_nm_col]].groupby([mrna_acc_col]).agg(
        mRNA_count=(mrna_nm_col, lambda x: len(set(x))),
        mRNA_names=(mrna_nm_col, lambda x: sorted(set(x)))
    ).reset_index(drop=False)
    mrna_map = mrna_map.sort_values(by='mRNA_count', ascending=False).reset_index(drop=True)
    assert list(mrna_map['mRNA_count'])[0] == 1
    # write_df(df=mrna_map, file_path=join(output_data_path, "mrna_map.csv"))

    # 3 - check all sRNA in interactions table are included in "all sRNA"
    missing_srnas = set(inter_data[srna_acc_col]) - set(srna_data['sRNA_accession_id'])
    assert len(missing_srnas) == 0, f"sRNAs {missing_srnas} are missing in srna_data"
    # 4 - check all mRNA in interactions table are included in "all mRNA"
    missing_mrnas = set(inter_data[mrna_acc_col]) - set(mrna_data['mRNA_accession_id'])
    assert len(missing_mrnas) == 0, f"{len(missing_mrnas)} mRNAs {missing_mrnas} are missing in mrna_data"

    # 4 - check structure of mRNA and sRNA data
    assert len(srna_data) == len(set(srna_data['sRNA_accession_id']))
    assert len(mrna_data) == len(set(mrna_data['mRNA_accession_id']))

    # --------------  interactions  --------------
    logger.info(f"Salmonella enterica - interactions: {len(inter_data)}, "
                f"unique interactions: {len(inter_data.groupby([srna_acc_col, mrna_acc_col]).count())}")
    # -------------- complete cols
    inter_data['count'] = 1

    # -------------- unique interactions intersection
    strain_col = 'strain_name'
    inter_data[strain_col] = "Salmonella enterica serovar Typhimurium strain SL1344"
    g_cols = [strain_col, srna_acc_col, srna_nm_col, mrna_acc_col, mrna_nm_col, 'interaction_label']
    unq_inter = inter_data.copy()[g_cols + ['count']].groupby(g_cols, as_index=False).count().reset_index(drop=True)
    assert list(set(inter_data[strain_col])) == list(set(unq_inter[strain_col]))

    # -------------- convert sRNA names

    # -------------- summary
    strain_col = 'strain_name'
    df_sum = unq_inter.groupby([strain_col]).agg(
        dataset=('count', lambda x: 'Matera 2022'),
        unique_sRNAs=(srna_acc_col, lambda x: len(set(x))),
        unique_targets=(mrna_acc_col, lambda x: len(set(x))),
        unq_pos_inter=('interaction_label', 'sum'),
        unq_neg_inter=('interaction_label', lambda x: sum(x == 0)),
        unq_inter=('interaction_label', 'count'),
        all_inter=('count', 'sum')
    ).reset_index(drop=False)

    df_sum = df_sum.sort_values(by='unq_inter', ascending=False).reset_index(drop=True)
    df_sum = df_sum.rename(columns={strain_col: 'strain'})
    df_sum['total_sRNAs'] = len(srna_data)
    df_sum['total_mRNAs'] = len(mrna_data)

    return unq_inter, df_sum, srna_data, mrna_data


def analyze_ecoli_epec_inter(mrna_data: pd.DataFrame, srna_data: pd.DataFrame, inter_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    "Enteropathogenic Escherichia coli - EPEC E2348/69"

    :param mrna_data:
    :param srna_data:
    :param inter_data:
    :return:
    """
    # --------------  validations  --------------
    # 1 - check sRNA acc is mapped to a single sRNA name
    srna_map = inter_data[['sRNA_accession_id_Eco', 'sRNA']].groupby(['sRNA_accession_id_Eco']).agg(
        sRNA_count=('sRNA', lambda x: len(set(x))),
        sRNA_names=('sRNA', lambda x: sorted(set(x)))
    ).reset_index(drop=False)
    srna_map = srna_map.sort_values(by='sRNA_count', ascending=False).reset_index(drop=True)
    assert list(srna_map['sRNA_count'])[0] == 1

    # 2 - check mRNA acc is mapped to a single mRNA name
    mrna_map = inter_data[['mRNA_accession_id_Eco', 'mRNA']].groupby(['mRNA_accession_id_Eco']).agg(
        mRNA_count=('mRNA', lambda x: len(set(x))),
        mRNA_names=('mRNA', lambda x: sorted(set(x)))
    ).reset_index(drop=False)
    mrna_map = mrna_map.sort_values(by='mRNA_count', ascending=False).reset_index(drop=True)
    assert list(mrna_map['mRNA_count'])[0] == 1
    # write_df(df=mrna_map, file_path=join(output_data_path, "mrna_map.csv"))

    # 3 - check all sRNA in interactions table are included in "all sRNA"
    missing_srnas = set(inter_data['sRNA_accession_id_Eco']) - set(srna_data['sRNA_accession_id'])
    assert len(missing_srnas) == 0, f"sRNAs {missing_srnas} are missing in srna_data"
    # 4 - check all mRNA in interactions table are included in "all mRNA"
    missing_mrnas = set(inter_data['mRNA_accession_id_Eco']) - set(mrna_data['mRNA_accession_id'])
    assert len(missing_mrnas) == 0, f"{len(missing_mrnas)} mRNAs {missing_mrnas} are missing in mrna_data"

    # 4 - check structure of mRNA and sRNA data
    assert len(srna_data) == len(set(srna_data['sRNA_accession_id']))
    assert len(mrna_data) == len(set(mrna_data['mRNA_accession_id']))

    # --------------  interactions  --------------
    logger.info(f"mizrahi_epec_inter - interactions: {len(inter_data)}, "
                f"unique interactions: {len(inter_data.groupby(['sRNA_accession_id_Eco', 'mRNA_accession_id_Eco']).count())}")
    # -------------- complete cols
    inter_data['count'] = 1

    # -------------- unique interactions intersection
    g_cols = ['strain_name', 'sRNA_accession_id_Eco', 'sRNA', 'mRNA_accession_id_Eco', 'mRNA', 'interaction_label']
    unq_inter = inter_data.copy()[g_cols + ['count']].groupby(g_cols, as_index=False).count().reset_index(drop=True)
    strain_col = 'strain_name'
    assert list(set(inter_data[strain_col])) == list(set(unq_inter[strain_col]))

    # -------------- convert sRNA names

    # -------------- summary
    strain_col = 'strain_name'
    df_sum = unq_inter.groupby([strain_col]).agg(
        dataset=('count', lambda x: 'Mizrahi_2021'),
        unique_sRNAs=('sRNA_accession_id_Eco', lambda x: len(set(x))),
        unique_targets=('mRNA_accession_id_Eco', lambda x: len(set(x))),
        unq_pos_inter=('interaction_label', 'sum'),
        unq_neg_inter=('interaction_label', lambda x: sum(x == 0)),
        unq_inter=('interaction_label', 'count'),
        all_inter=('count', 'sum')
    ).reset_index(drop=False)

    df_sum = df_sum.sort_values(by='unq_inter', ascending=False).reset_index(drop=True)
    df_sum = df_sum.rename(columns={strain_col: 'strain'})
    df_sum['total_sRNAs'] = len(srna_data)
    df_sum['total_mRNAs'] = len(mrna_data)

    return unq_inter, df_sum, srna_data, mrna_data


def analyze_ecoli_k12_inter(mrna_data: pd.DataFrame, srna_data: pd.DataFrame, inter_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    "Escherichia coli str. K-12 substr. MG1655"    (4332) srna = 61, mrna = 2095

    :param mrna_data:
    :param srna_data:
    :param inter_data:
    """
    # --------------  validations  --------------
    # 1 - check sRNA acc is mapped to a single sRNA name
    srna_map = inter_data[['sRNA_accession_id_Eco', 'sRNA']].groupby(['sRNA_accession_id_Eco']).agg(
        sRNA_count=('sRNA', lambda x: len(set(x))),
        sRNA_names=('sRNA', lambda x: sorted(set(x)))
    ).reset_index(drop=False)
    srna_map = srna_map.sort_values(by='sRNA_count', ascending=False).reset_index(drop=True)
    assert list(srna_map['sRNA_count'])[0] == 1

    # 2 - check mRNA acc is mapped to a single mRNA name
    mrna_map = inter_data[['mRNA_accession_id_Eco', 'mRNA']].groupby(['mRNA_accession_id_Eco']).agg(
        mRNA_count=('mRNA', lambda x: len(set(x))),
        mRNA_names=('mRNA', lambda x: sorted(set(x)))
    ).reset_index(drop=False)
    mrna_map = mrna_map.sort_values(by='mRNA_count', ascending=False).reset_index(drop=True)
    assert list(mrna_map['mRNA_count'])[0] == 1
    # write_df(df=mrna_map, file_path=join(output_data_path, "mrna_map.csv"))

    # 3 - check all sRNA in interactions table are included in "all sRNA"
    missing_srnas = set(inter_data['sRNA_accession_id_Eco']) - set(srna_data['EcoCyc_accession_id'])
    assert len(missing_srnas) == 0, f"sRNAs {missing_srnas} are missing in srna_data"
    # 4 - check all mRNA in interactions table are included in "all mRNA"
    missing_mrnas = set(inter_data['mRNA_accession_id_Eco']) - set(mrna_data['EcoCyc_accession_id'])
    assert len(missing_mrnas) == 0, f"{len(missing_mrnas)} mRNAs {missing_mrnas} are missing in mrna_data"

    # 4 - check structure of mRNA and sRNA data
    assert len(srna_data) == len(set(srna_data['EcoCyc_accession_id']))
    assert len(mrna_data) == len(set(mrna_data['EcoCyc_accession_id']))

    # --------------  interactions  --------------
    logger.info(f"ecoli_inter - interactions: {len(inter_data)}, "
                f"unique interactions: {len(inter_data.groupby(['sRNA_accession_id_Eco', 'mRNA_accession_id_Eco']).count())}")
    # -------------- complete cols
    inter_data['count'] = 1

    # -------------- unique interactions intersection
    g_cols = ['strain_name', 'chromosome', 'sRNA_accession_id_Eco', 'sRNA', 'mRNA_accession_id_Eco', 'mRNA',
              'interaction_label']
    unq_inter = inter_data.copy()[g_cols + ['count']].groupby(g_cols, as_index=False).count().reset_index(drop=True)
    strain_col = 'strain_name'
    assert list(set(inter_data[strain_col])) == list(set(unq_inter[strain_col]))

    # -------------- summary
    df_sum = unq_inter.groupby([strain_col]).agg(
        dataset=('count', lambda x: 'GraphRNA'),
        unique_sRNAs=('sRNA_accession_id_Eco', lambda x: len(set(x))),
        unique_targets=('mRNA_accession_id_Eco', lambda x: len(set(x))),
        unq_pos_inter=('interaction_label', 'sum'),
        unq_neg_inter=('interaction_label', lambda x: sum(x == 0)),
        unq_inter=('interaction_label', 'count'),
        all_inter=('count', 'sum')
    ).reset_index(drop=False)

    df_sum = df_sum.sort_values(by='unq_inter', ascending=False).reset_index(drop=True)
    df_sum = df_sum.rename(columns={strain_col: 'strain'})
    df_sum['total_sRNAs'] = len(srna_data)
    df_sum['total_mRNAs'] = len(mrna_data)

    # -------------- align columns
    for rna in ['sRNA', 'mRNA']:
        rename_map = {
            'EcoCyc_locus_tag': f'{rna}_locus_tag',
            'EcoCyc_rna_name': f'{rna}_name',
            'EcoCyc_rna_name_synonyms': f'{rna}_name_synonyms',
            'EcoCyc_start': f'{rna}_start',
            'EcoCyc_end': f'{rna}_end',
            'EcoCyc_strand': f'{rna}_strand',
            'EcoCyc_sequence': f'{rna}_sequence',
            'EcoCyc_accession-2': f'{rna}_accession_2'
        }
        if rna == 'sRNA':
            srna_data = srna_data.rename(columns=rename_map)
        else:
            mrna_data = mrna_data.rename(columns=rename_map)

    return unq_inter, df_sum, srna_data, mrna_data


def get_van_diagram_info(ecoli_k12_all_srna, ecoli_k12_all_mrna, ecoli_k12_inter, ecoli_k12_nm,
                         other_bacteria_all_srna, other_bacteria_all_mrna, other_bacteria_inter,
                         other_bacteria_nm: str, other_bacteria_srna_nm_col: str = 'sRNA_name',
                         other_bacteria_mrna_nm_col: str = 'mRNA_name', sfx: str = '') -> pd.DataFrame:
    dataset_out_col = 'dataset'
    _, _, van_rna_df = find_srna_mrna_intersection(
        nm1=f'ecoli_k12 {sfx}', all_srna1=ecoli_k12_all_srna, all_mrna1=ecoli_k12_all_mrna,
        srna_nm_col_1='EcoCyc_rna_name', mrna_nm_col_1='EcoCyc_rna_name',
        nm2=f'{other_bacteria_nm} {sfx}', all_srna2=other_bacteria_all_srna, all_mrna2=other_bacteria_all_mrna,
        srna_nm_col_2=other_bacteria_srna_nm_col, mrna_nm_col_2=other_bacteria_mrna_nm_col,
        dataset_out_col=dataset_out_col)

    van_inter_dict = find_interactions_intersection(
        nm1=f'ecoli_k12 {sfx}', df1=ecoli_k12_inter, strain_col1='strain_name', srna_nm_col_1='sRNA', target_nm_col_1='mRNA',
        nm2=f'{other_bacteria_nm} {sfx}', df2=other_bacteria_inter, strain_col2='strain_name', srna_nm_col_2='sRNA', target_nm_col_2='mRNA',
        dataset_out_col=dataset_out_col, cols_to_ignore=['num', 'strain'])
    van_inter_df = list(van_inter_dict.values())[0]

    van_df = pd.merge(van_inter_df, van_rna_df, on=dataset_out_col, how='inner')

    return van_df


def get_ecoli_srna_to_nm() -> Dict[str, str]:
    # dicF_1 ?
    _3etsleuz_str = "3'etsleuz"
    ecoli_srna_nm_to_nm = {
        "3ZZESTleuZ": _3etsleuz_str,
        "3zzestleuz": _3etsleuz_str,
        "3'ets-<i>leuz</i>": _3etsleuz_str,
        "3'ets-leuz": _3etsleuz_str
    }
    return ecoli_srna_nm_to_nm


def preprocess_ecoli_k12_inter(mrna_data: pd.DataFrame, srna_data: pd.DataFrame, inter_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    "Escherichia coli str. K-12 substr. MG1655"    (4332) srna = 61, mrna = 2095

    :param mrna_data:
    :param srna_data:
    :param inter_data:
    """

    logger.info(f"Preprocess - ecoli_k12_inter - interactions: {len(inter_data)}, "
                f"unique interactions: {len(inter_data.groupby(['sRNA_accession_id_Eco', 'mRNA_accession_id_Eco']).count())}")

    # --------------  all mRNAs  --------------
    # 1 - no redundant spaces in names + lower
    assert sum([" " in x for x in mrna_data['EcoCyc_rna_name']]) == 0, "redundant spaces in K12 mRNA name"
    mrna_data['EcoCyc_rna_name'] = mrna_data['EcoCyc_rna_name'].apply(lambda x: x.lower())

    # --------------  all sRNAs  --------------
    # 1 - sRNA name - use accession if sRNA name is Null
    srna_data['EcoCyc_rna_name'] = list(map(lambda nm, acc: acc if pd.isnull(nm) else nm,
                                            srna_data['EcoCyc_rna_name'], srna_data['EcoCyc_accession_id']))

    # 2 - no redundant spaces in names + lower
    assert sum([" " in x for x in srna_data['EcoCyc_rna_name']]) == 0, "redundant spaces in K12 sRNA name"
    srna_data['EcoCyc_rna_name'] = srna_data['EcoCyc_rna_name'].apply(lambda x: x.lower())

    # 3 - align sRNA names of e.coli (to match, e.g. e.coli EPEC)
    srna_nm_to_nm = get_ecoli_srna_to_nm()
    srna_data['EcoCyc_rna_name'] = srna_data['EcoCyc_rna_name'].apply(lambda x: srna_nm_to_nm.get(x, x))

    # --------------  interactions  --------------
    # 1 - no redundant spaces in names + lower
    assert sum([" " in x for x in inter_data['sRNA']]) == 0, "redundant spaces in K12 inter sRNA name"
    assert sum([" " in x for x in inter_data['mRNA']]) == 0, "redundant spaces in K12 inter mRNA name"

    inter_data['sRNA'] = inter_data['sRNA'].apply(lambda x: x.lower())
    inter_data['mRNA'] = inter_data['mRNA'].apply(lambda x: x.lower())

    # 3 - align sRNA names of e.coli (to match, e.g. e.coli EPEC)
    srna_nm_to_nm = get_ecoli_srna_to_nm()
    inter_data['sRNA'] = inter_data['sRNA'].apply(lambda x: srna_nm_to_nm.get(x, x))

    # 3 - select experiments
    experiments = ['Gelhausen 2019', 'Iosub 2020 (CLASH)', 'Melamed 2016 (RIL-seq)', 'Melamed 2020 (RIL-seq)',
                   'Pain 2015', 'Wright 2013', 'sRNATarBase3.0']  # 'sRNATarBase3.0'
    logger.info(f"using interaction from: {experiments}")
    inter_data = inter_data[inter_data['dir'].isin(experiments)].reset_index(drop=True)

    # 4 - filter out negative interactions
    mask_negative_interactions = inter_data['interaction_label'] == 0
    inter_data = inter_data[~mask_negative_interactions].reset_index(drop=True)
    logger.info(f"filtered out {sum(mask_negative_interactions)} negative interactions")

    # --------------  assert  --------------
    assert check_interacting_rna_names_included_in_all_rnas(inter_df=inter_data,
                                                            inter_rna_nm_col='sRNA',
                                                            all_rna=srna_data, all_rna_nm_col='EcoCyc_rna_name')
    assert check_interacting_rna_names_included_in_all_rnas(inter_df=inter_data,
                                                            inter_rna_nm_col='mRNA',
                                                            all_rna=mrna_data, all_rna_nm_col='EcoCyc_rna_name')

    return mrna_data, srna_data, inter_data


def preprocess_ecoli_epec_inter(mrna_data: pd.DataFrame, srna_data: pd.DataFrame, inter_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    "Escherichia coli EPEC E2348/69"

    :param mrna_data:
    :param srna_data:
    :param inter_data:

    """
    logger.info(f"Preprocess - ecoli_epec_inter - interactions: {len(inter_data)}, "
                f"unique interactions: {len(inter_data.groupby(['sRNA_accession_id_Eco', 'mRNA_accession_id_Eco']).count())}")

    # --------------  all mRNAs  --------------
    # 1 - mRNA name - use mRNA accession instead of synonyms (as done in the interactions table)
    mrna_data['prev_mRNA_name'] = mrna_data['mRNA_name']
    mrna_data['mRNA_name'] = \
        list(map(lambda nm, acc, syn: acc if nm == syn else nm,
                 mrna_data['prev_mRNA_name'], mrna_data['mRNA_accession_id'], mrna_data['mRNA_name_synonyms']))

    # 2 - no redundant spaces in names + lower
    assert sum([" " in x for x in mrna_data['mRNA_name']]) == 0, "redundant spaces in EPEC mRNA_name"
    mrna_data['mRNA_name'] = mrna_data['mRNA_name'].apply(lambda x: x.lower())

    # --------------  all sRNAs  --------------
    # 1 - no redundant spaces in names + lower
    assert sum([" " in x for x in srna_data['sRNA_name']]) == 0, "redundant spaces in EPEC sRNA_name"
    srna_data['sRNA_name'] = srna_data['sRNA_name'].apply(lambda x: x.lower())

    # 2 - align sRNA names of e.coli (to match, e.g. e.coli K12)
    srna_nm_to_nm = get_ecoli_srna_to_nm()
    srna_data['sRNA_name'] = srna_data['sRNA_name'].apply(lambda x: srna_nm_to_nm.get(x, x))

    # --------------  interactions  --------------
    # 1 - no redundant spaces in names + lower
    assert sum([" " in x for x in inter_data['sRNA']]) == 0, "redundant spaces in EPEC inter sRNA name"
    assert sum([" " in x for x in inter_data['mRNA']]) == 0, "redundant spaces in EPEC inter mRNA name"

    inter_data['sRNA'] = inter_data['sRNA'].apply(lambda x: x.lower())
    inter_data['mRNA'] = inter_data['mRNA'].apply(lambda x: x.lower())

    # 2 - align sRNA names of e.coli (to match, e.g. e.coli K12)
    srna_nm_to_nm = get_ecoli_srna_to_nm()
    inter_data['sRNA'] = inter_data['sRNA'].apply(lambda x: srna_nm_to_nm.get(x, x))

    # 3 - save data source in 'dir' column
    inter_data['dir'] = inter_data['Data_source']

    # --------------  assert  --------------
    assert check_interacting_rna_names_included_in_all_rnas(inter_df=inter_data,
                                                            inter_rna_nm_col='sRNA',
                                                            all_rna=srna_data, all_rna_nm_col='sRNA_name')
    assert check_interacting_rna_names_included_in_all_rnas(inter_df=inter_data,
                                                            inter_rna_nm_col='mRNA',
                                                            all_rna=mrna_data, all_rna_nm_col='mRNA_name')

    return mrna_data, srna_data, inter_data


def preprocess_salmonella_inter(mrna_data: pd.DataFrame, srna_data: pd.DataFrame, inter_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Salmonella enterica serovar Typhimurium strain SL1344,  Genome: NC_016810.1

    :param mrna_data:
    :param srna_data:
    :param inter_data:

    """
    logger.info(f"Preprocess - salmonella - interactions: {len(inter_data)}, "
                f"unique interactions: {len(inter_data.groupby(['sRNA_accession_id', 'mRNA_accession_id']).count())}")

    # --------------  all mRNAs  --------------
    # 1 - mRNA accession - use mRNA locus tag as accession id (as done in the interactions table)
    mrna_data['mRNA_accession_id'] = mrna_data['mRNA_locus_tag']

    # 2 - no redundant spaces in names + lower
    assert sum([" " in x for x in mrna_data['mRNA_name']]) == 0, "redundant spaces in Salmonella mRNA_name"
    mrna_data['mRNA_name'] = mrna_data['mRNA_name'].apply(lambda x: x.lower())

    # --------------  all sRNAs  --------------
    # 1 - no redundant spaces in names + lower
    assert sum([" " in x for x in srna_data['sRNA_name']]) == 0, "redundant spaces in  sRNA_name"
    srna_data['sRNA_name'] = srna_data['sRNA_name'].apply(lambda x: x.lower())

    # 2 - align sRNA names of Salmonella (to match, e.g. e.coli K12)
    # todo  ---------- check if names alignment is needed
    # srna_nm_to_nm = get_ecoli_srna_to_nm()
    # srna_data['sRNA_name'] = srna_data['sRNA_name'].apply(lambda x: srna_nm_to_nm.get(x, x))

    # --------------  interactions  --------------
    # 1 - no redundant spaces in names + lower
    assert sum([" " in x for x in inter_data['sRNA_name']]) == 0, "redundant spaces in Salmonella inter sRNA name"
    assert sum([" " in x for x in inter_data['mRNA_name']]) == 0, "redundant spaces in Salmonella inter mRNA name"

    inter_data['sRNA'] = inter_data['sRNA_name'].apply(lambda x: x.lower())
    inter_data['mRNA'] = inter_data['mRNA_name'].apply(lambda x: x.lower())

    # 2 - align sRNA names of e.coli (to match, e.g. e.coli K12)
    # todo  ---------- check if names alignment is needed
    # srna_nm_to_nm = get_ecoli_srna_to_nm()
    # inter_data['sRNA'] = inter_data['sRNA'].apply(lambda x: srna_nm_to_nm.get(x, x))

    # 3 - save 'dir' and 'file_name' columns
    inter_data['dir'] = inter_data['Data_source']
    inter_data['file_name'] = inter_data['Experiment']
    inter_data['interaction_label'] = inter_data['Interaction_label'].apply(lambda x: 1 if x == 'interaction' else 0)

    # --------------  assert  --------------
    assert check_interacting_rna_names_included_in_all_rnas(inter_df=inter_data,
                                                            inter_rna_nm_col='sRNA',
                                                            all_rna=srna_data, all_rna_nm_col='sRNA_name')
    assert check_interacting_rna_names_included_in_all_rnas(inter_df=inter_data,
                                                            inter_rna_nm_col='mRNA',
                                                            all_rna=mrna_data, all_rna_nm_col='mRNA_name')
    return mrna_data, srna_data, inter_data
