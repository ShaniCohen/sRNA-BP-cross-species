import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set
import logging
logger = logging.getLogger(__name__)


def preprocess_interproscan_annot(d: Dict[str, Set[str]]):
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
    out_col_seq = 'protein_sequence'

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

            if len(go_xrefs) > 0 and len(go_xrefs_entry) > 0:
                logger.warning(f"both goXRefs and goXRefs_entry are not null")
            
            go_xrefs += go_xrefs_entry
            # TOOD: make it unique
            # go_xrefs = list(set(go_xrefs))
            if len(go_xrefs) > 0:
                rec = {
                    out_col_header: input_header,
                    out_col_seq: protein_sequence,
                    # 'signature_accession': sig['accession'],
                    # 'signature_name': sig['name'],
                    # 'signature_description': sig['description'],
                    'signature_library': sig['signatureLibraryRelease']['library'],
                    # 'entry_accession': entry['accession'],
                    # 'entry_name': entry['name'],
                    # 'entry_description': entry['description'],
                    # 'entry_type': entry['type'],
                    'go_xrefs': go_xrefs
                }
                records.append(rec)

    input_df = pd.DataFrame({out_col_header: headers, out_col_seq: seqs}).drop_duplicates().reset_index(drop=True)
    assert len(input_df) == len(d['results']), "some results are duplicated / missing"
    matches_df = pd.DataFrame(records)
    df = pd.merge(input_df, matches_df, on=[out_col_header, out_col_seq], how='left')
    return df, out_col_header


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

def annotate_mrnas_w_curated_annt(strain_nm: str, all_mrna: pd.DataFrame, all_mrna_locus_col: str, curated_annot: pd.DataFrame, 
                                  curated_annot_locus_col: str) -> pd.DataFrame:
    """
    Preprocess GO terms per locus name.
    """
    # 1 - merge GO terms with all mRNAs
    all_mrna_w_curated_annt = pd.merge(all_mrna, curated_annot, left_on=all_mrna_locus_col, right_on=curated_annot_locus_col, how='left')
    assert len(all_mrna) == len(all_mrna_w_curated_annt), "duplicate locus names"

    # 2 - log statistics
    num_mrna = len(all_mrna)
    num_mrna_w_annt = sum(pd.notnull(all_mrna_w_curated_annt['Locus_Name']))
    logger.info(f"{strain_nm}: out of {num_mrna} mRNAs {num_mrna_w_annt} have curated GO annotations ({(num_mrna_w_annt/num_mrna)*100:.2f})%")

    return all_mrna_w_curated_annt

def annotate_mrnas_w_interproscan_annt(strain_nm: str, all_mrna: pd.DataFrame, all_mrna_acc_col: str, interproscan_annot: pd.DataFrame, interproscan_header_col: str):
    """
    Preprocess InterProScan annotations (GO terms per header).
    """
    # Example preprocessing steps
    processed_go_terms = {}
    for header, terms in interproscan_annot.items():
        # Example processing: filter out certain terms or modify structure
        processed_go_terms[header] = [term for term in terms if 'example_filter' not in term]
    return processed_go_terms

def parse_header_to_acc_locus_and_name(df: pd.DataFrame, df_header_col: str, acc_col: str, locus_col: str, name_col: str) -> pd.DataFrame:
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