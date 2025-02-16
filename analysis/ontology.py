from typing import Dict
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
from utils.general import read_df, write_df
import json
import sys
import os


ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")


class Ontology:
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info(f"initializing Ontology")
        self.config = config

    def load_go_ontology(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        self._load_go_ontology()
        # TODO: read GOA data  (should not be here but for testing)
        self._load_goa()
        return 

    def _load_go_ontology(self):
        gene_ontology_json_file = join(self.config['input_data_path'], self.config['gene_ontology_dir'], self.config['gene_ontology_json_file'])
        with open(gene_ontology_json_file, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        ontology = ontology['graphs'][0]
        
        id = ontology['id']
        
        # process nodes
        nodes = ontology['nodes']
        class_nodes = [n for n in nodes if n['type'] == 'CLASS']
        property_nodes = [n for n in nodes if n['type'] == 'PROPERTY']
        property_id_to_lbl = {p['id']: p['lbl'] for p in property_nodes}
        assert len(nodes) == len(class_nodes) + len(property_nodes)
        self.logger.info(f"Gene Ontology: {id}, nodes: {len(nodes)} (class: {len(class_nodes)}, property: {len(property_nodes)})")
        
        edges = ontology['edges']
        edge_vals, count = np.unique([e['pred'] for e in edges], return_counts=True)
        edge_val_to_lbl = {ev: property_id_to_lbl.get(ev) for ev in edge_vals}
        print()

    def _load_goa(self):
        from goatools.anno.gaf_reader import GafReader

        # Path to your GOA file
        gaf_file = "/sise/home/shanisa/PhD/sRNA_BP_analysis/Data/main_analysis/gene_annot/Escherichia_coli_K12_MG1655/e_coli_MG1655.goa"
        # Create a GafReader object
        gaf_reader = GafReader(gaf_file)

        # Read the GOA file
        annotations = gaf_reader.read_gaf()

        # Print the annotations
        for gene, go_terms in annotations.items():
            print(f"Gene: {gene}, GO Terms: {go_terms}")