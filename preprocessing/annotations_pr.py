import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set
import logging
logger = logging.getLogger(__name__)


def _add_library_to_go_xrefs(go_xrefs: List[Dict[str, str]], library):
    for go_xref in go_xrefs:
        go_xref['library'] = library
    return go_xrefs


def _remove_duplicated_go_xrefs(go_xrefs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplications in GO xrefs.
    """
    go_xrefs_no_dup, go_ids = [], []
    for d in go_xrefs:
        if d['id'] not in go_ids:
            go_xrefs_no_dup.append(d)
            go_ids.append(d['id'])
    return go_xrefs_no_dup


def _split_go_xrefs_to_categories(go_xrefs: List[Dict[str, str]]) -> Tuple[list, list, list]:
    # 'category': 'BIOLOGICAL_PROCESS', 'MOLECULAR_FUNCTION', 'CELLULAR_COMPONENT'
    bp, mf, cc = [], [], []
    for d in go_xrefs:
        if d['category'] == 'BIOLOGICAL_PROCESS':
            bp.append(d)
        elif d['category'] == 'MOLECULAR_FUNCTION':
            mf.append(d)
        elif d['category'] == 'CELLULAR_COMPONENT':
            cc.append(d)
    return bp, mf, cc


def preprocess_interproscan_annot(raw_interproscan_annot: Dict[str, Set[str]]) -> Tuple[pd.DataFrame, str]:
    interproscan_annot, header_col, protein_seq_col, lib_col = _preprocess_raw_interproscan_annot(raw_interproscan_annot)
    interproscan_annot = _filter_interproscan_annot(interproscan_annot, lib_col)
    interproscan_annot = _group_interproscan_annot_per_protein(interproscan_annot, header_col, protein_seq_col)
    return interproscan_annot, header_col


def _preprocess_raw_interproscan_annot(d: Dict[str, Set[str]]) -> Tuple[pd.DataFrame, str, str, str, str]:
    """_summary_

    Args:
        d (Dict[str, Set[str]]): dict in the following format: 
            {
                'results': [
                    {
                        'sequence': str <the protein sequence>,
                        'xref': {'name': str <the header of the input fasta file>},
                         ... ,
                        'matches': [
                            {   
                                'goXRefs': [],
                                'signature': {
                                    'accession': ,
                                    'name': ,
                                    'description': ,
                                    'signatureLibraryRelease': {'library': str <library name>},
                                    'entry': {
                                        "accession" : "IPR009006",
                                        "name" : "Ala_racemase/Decarboxylase_C",
                                        "description" : "Alanine racemase/group IV decarboxylase, C-terminal",
                                        "type" : "HOMOLOGOUS_SUPERFAMILY",
                                        "goXRefs" : [ 
                                            {
                                                "name" : "catalytic activity",
                                                "databaseName" : "GO",
                                                "category" : "MOLECULAR_FUNCTION",
                                                "id" : "GO:0003824"
                                            },
                                            ...
                                        ],
                                        "pathwayXRefs" : [ ]
                                    }
                                },    
                                } 
                                'locations', 
                                'evalue', 
                                'model-ac'
                            }
                        ]
                        
                    },
                    ...
                ],
            }

    Returns:
        _type_: _description_
    """
    logger.info(f"Preprocess {len(d['results'])} interproscan results. interproscan-version = {d['interproscan-version']}")
    headers, seqs = [], []
    out_col_header = 'input_header'
    out_col_protein_seq = 'protein_sequence'
    out_col_lib = 'signature_library'

    records = []
    for res in d['results']:
        input_header = res['xref'][0]['id']
        protein_sequence = res['sequence']
        headers.append(input_header)
        seqs.append(protein_sequence)
        
        for match in res['matches']:
            go_xrefs = match.get('goXRefs', [])
            sig = match['signature']
            entry = sig['entry']
            go_xrefs_entry = entry.get('goXRefs', []) if pd.notnull(entry) else []
            library = sig['signatureLibraryRelease']['library']

            if len(go_xrefs) > 0 and len(go_xrefs_entry) > 0:
                logger.warning(f"both goXRefs and goXRefs_entry are not null")
            
            go_xrefs += go_xrefs_entry

            if len(go_xrefs) > 0:
                go_xrefs = _remove_duplicated_go_xrefs(go_xrefs)
                bp, mf, cc = _split_go_xrefs_to_categories(go_xrefs)
                # go_xrefs = _add_library_to_go_xrefs(go_xrefs, library)  # Note: if this is ebabled, revisit the grouping function (where duplications are removed for BP, MF and CC)

                rec = {
                    out_col_header: input_header,
                    out_col_protein_seq: protein_sequence,
                    # 'signature_accession': sig['accession'],
                    # 'signature_name': sig['name'],
                    # 'signature_description': sig['description'],
                    out_col_lib: library,
                    # 'entry_accession': entry['accession'],
                    # 'entry_name': entry['name'],
                    # 'entry_description': entry['description'],
                    # 'entry_type': entry['type'],
                    'go_xrefs': go_xrefs,
                    'BP_go_xrefs': bp,
                    'MF_go_xrefs': mf,
                    'CC_go_xrefs': cc
                }
                records.append(rec)

    input_df = pd.DataFrame({out_col_header: headers, out_col_protein_seq: seqs}).drop_duplicates().reset_index(drop=True)
    assert len(input_df) == len(d['results']), "some results are duplicated / missing"
    matches_df = pd.DataFrame(records)
    df = pd.merge(input_df, matches_df, on=[out_col_header, out_col_protein_seq], how='left')
    return df, out_col_header, out_col_protein_seq, out_col_lib


def _filter_interproscan_annot(df: pd.DataFrame, lib_col: str) -> pd.DataFrame:
    # 1 - filter out certain libraries
    mask = pd.Series([True] * len(df))
    # valid_libs = ['Pfam', 'TIGRFAM', 'Gene3D', 'SUPERFAMILY', 'SMART', 'CDD', 'HAMAP', 'ProSiteProfiles', 'ProSitePatterns', 'PRINTS', 'PIRSF', 'PANTHER', 'SFLD', 'CATH-Gene3D', 'SFLD', 'InterPro']
    # mask = df[lib_col].isin(valid_libs)
    _len = len(df)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Libraries filtering: Filter out {(_len - len(df))} interproscan annotations")

    return df


def _group_interproscan_annot_per_protein(df: pd.DataFrame, header_col: str, protein_seq_col: str) -> pd.DataFrame:
    """
    Group InterProScan annotations per protein (optional add library information to GO xrefs).
    
    Args:
        df (pd.DataFrame): DataFrame containing the InterProScan annotations.
        header_col (str): Name of the column containing the header.
        protein_seq_col (str): Name of the column containing the protein sequence.
    
    Returns:
        pd.DataFrame: DataFrame with grouped annotations per protein.
    """
    def ensure_list(x):
        return x if isinstance(x, list) else []

    df['BP_go_xrefs'] = df['BP_go_xrefs'].apply(ensure_list)
    df['MF_go_xrefs'] = df['MF_go_xrefs'].apply(ensure_list)
    df['CC_go_xrefs'] = df['CC_go_xrefs'].apply(ensure_list)

    grouped_df = df.groupby([header_col, protein_seq_col]).agg({
        'BP_go_xrefs': lambda x: _remove_duplicated_go_xrefs([item for sublist in x for item in sublist]),
        'MF_go_xrefs': lambda x: _remove_duplicated_go_xrefs([item for sublist in x for item in sublist]),
        'CC_go_xrefs': lambda x: _remove_duplicated_go_xrefs([item for sublist in x for item in sublist])
    }).reset_index()
    
    return grouped_df


def preprocess_curated_annot(strain_nm: str, annot_uniport: Dict[str, Set[str]], annot_map_uniport_to_locus: pd.DataFrame):
    """
    Args:
        annot_uniport (Dict[str, Set[str]]): _description_
        annot_map_uniport_to_locus (pd.DataFrame): table mapping UniProt to other IDs (columns = ['UniProt_ID', 'Database', 'Mapped_ID'])
    """
    # 1 - prepare a dict mapping UniProt_ID to the following keys
    # ['Gene_OrderedLocusName', 'Gene_Name', 'Gene_Synonym', 'BioCyc']
    col_uniport_id = 'UniProt_ID'
    databases = ['Gene_OrderedLocusName', 'Gene_Name', 'Gene_Synonym', 'BioCyc']
    mask = annot_map_uniport_to_locus['Database'].isin(databases)
    annot_map = annot_map_uniport_to_locus[mask].groupby([col_uniport_id, 'Database'], as_index=False).agg(lambda x: sorted(set(x))).reset_index(drop=True)
    
    out_col_map = 'Map'
    annot_map[out_col_map] = list(map(lambda x: (x[0], x[1]), annot_map[['Database', 'Mapped_ID']].values))
    annot_map = annot_map[[col_uniport_id, out_col_map]].groupby([col_uniport_id], as_index=False).agg(lambda x: dict(sorted(x))).reset_index(drop=True)

    # 2
    out_col_go_terms = 'GO_Terms'
    uniport_go_terms = pd.DataFrame({
        col_uniport_id: list(annot_uniport.keys()),
        out_col_go_terms: list(annot_uniport.values())
    })

    out_col_locus_nm = 'Locus_Name'
    df = pd.merge(uniport_go_terms, annot_map, on=col_uniport_id, how='left')
    df['Locus_Name_list'] = df[out_col_map].apply(lambda x: x.get('Gene_OrderedLocusName', []))
    pattern = r'b\d+'  # 'b' followed by one or more digits, e.g., b0001
    valid_locus_nm = df['Locus_Name_list'].apply(lambda x: [s for s in x if re.match(pattern, s)])
    df[out_col_locus_nm] = valid_locus_nm.apply(lambda x: x[0] if len(x) == 1 else None)
    df = df[pd.notnull(df[out_col_locus_nm])][[out_col_locus_nm, out_col_go_terms, col_uniport_id, out_col_map]].reset_index(drop=True)

    missing_locus_nm = sum(valid_locus_nm.str.len() == 0)
    multi_locus_nm = sum(valid_locus_nm.str.len() > 1)
    logger.info(f"{strain_nm}: out of {len(uniport_go_terms)} GO annotations ({col_uniport_id}) - {missing_locus_nm} are missing locus names, {multi_locus_nm} has multi locus names")

    return df, out_col_locus_nm


def annotate_mrnas_w_curated_annt(strain_nm: str, data: dict) -> pd.DataFrame:
    """
    Preprocess GO terms per locus name.
    """
    all_mrna, all_mrna_locus_col = data['all_mrna'], data['all_mrna_locus_col'], 
    curated_annot, curated_annot_locus_col = data['curated_annot'], data['curated_locus_col']

    # 1 - merge GO terms with all mRNAs
    all_mrna_w_curated_annt = pd.merge(all_mrna, curated_annot, left_on=all_mrna_locus_col, right_on=curated_annot_locus_col, how='left')
    assert len(all_mrna) == len(all_mrna_w_curated_annt), "duplicate locus names"

    # 2 - log statistics
    num_mrna = len(all_mrna)
    num_mrna_w_annt = sum(pd.notnull(all_mrna_w_curated_annt['Locus_Name']))
    logger.info(f"{strain_nm}: out of {num_mrna} mRNAs {num_mrna_w_annt} have curated GO annotations ({(num_mrna_w_annt/num_mrna)*100:.2f}%)")

    return all_mrna_w_curated_annt


def _parse_header_to_acc_locus_and_name(df: pd.DataFrame, df_header_col: str, acc_col: str, locus_col: str, name_col: str) -> pd.DataFrame:
    """
    Parse the header column in the InterProScan annotations dataframe to extract accession, locus, and name.
    
    Args:
        df (pd.DataFrame): DataFrame containing the InterProScan annotations.
        df_header_col (str): Name of the column containing the header.
        acc_col (str): Name of the column to store the accession.
        locus_col (str): Name of the column to store the locus.
        name_col (str): Name of the column to store the name.
    
    Returns:
        pd.DataFrame: DataFrame with the new columns added.
    """
    def parse_header(header: str):
        parts = header.split('|')
        if len(parts) == 4:
            return parts[1], parts[2], parts[3]
        return None, None, None

    df[[acc_col, locus_col, name_col]] = df[df_header_col].apply(lambda x: pd.Series(parse_header(x)))
    return df


def annotate_mrnas_w_interproscan_annt(strain_nm: str, data: dict) -> pd.DataFrame:
    """
    """
    all_mrna = data['all_mrna']
    mrna_locus_col, mrna_name_col = data['all_mrna_locus_col'], data['all_mrna_name_col']
    mrna_acc_col = data['mrna_acc_col']

    # 1 - parse header to extract accession, locus, and name (use the same columns used data['all_mrna'])
    data['interproscan_annot'] = _parse_header_to_acc_locus_and_name(data['interproscan_annot'], data['interproscan_header_col'], mrna_acc_col, mrna_locus_col, mrna_name_col)
    
    # 1 - merge GO terms with all mRNAs
    ips_annot = data['interproscan_annot'][[mrna_acc_col, 'protein_sequence', 'BP_go_xrefs', 'MF_go_xrefs', 'CC_go_xrefs']]
    all_mrna_w_ips_annt = pd.merge(all_mrna, ips_annot, on=mrna_acc_col, how='left')
    assert len(all_mrna) == len(all_mrna_w_ips_annt), "duplicate accession numbers"

    # 1.1
    for a in ['BP_go_xrefs', 'MF_go_xrefs', 'CC_go_xrefs']:
        all_mrna_w_ips_annt[a] = all_mrna_w_ips_annt[a].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else None)

    # 2 - log statistics
    num_mrna = len(all_mrna)
    num_mrna_w_annt = sum(pd.notnull(all_mrna_w_ips_annt['protein_sequence']))
    num_mrna_w_bp_annt = sum(pd.notnull(all_mrna_w_ips_annt['BP_go_xrefs']))
    logger.info(f"{strain_nm}: out of {num_mrna} mRNAs {num_mrna_w_annt} have interproscan GO annotations ({(num_mrna_w_annt/num_mrna)*100:.2f}%) {num_mrna_w_bp_annt} have interproscan BP GO annotations ({(num_mrna_w_bp_annt/num_mrna)*100:.2f}%)")

    return all_mrna_w_ips_annt