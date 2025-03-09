from typing import Dict
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
from utils.general import read_df, write_df
from typing import List, Dict
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
        self.id: str = None
        self.class_nodes: List[dict] = None
        self.property_nodes: List[dict] = None
        self.edges = None

    def load_go_ontology(self):
        gene_ontology_json_file = join(self.config['gene_ontology_dir'], self.config['gene_ontology_json_file'])
        with open(gene_ontology_json_file, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        ontology = ontology['graphs'][0]
        # id
        self.id = ontology['id']
        
        # nodes
        nodes = self._remove_deprecated_nodes(ontology['nodes'])
        self.class_nodes = [n for n in nodes if n['type'] == 'CLASS']
        self.property_nodes = [n for n in nodes if n['type'] == 'PROPERTY']
        assert len(nodes) == len(self.class_nodes) + len(self.property_nodes)
        # remode deprecated nodes
        self.class_nodes = [n for n in self.class_nodes if 'deprecated' not in n]

        # edges
        self.edges = ontology['edges']
        self.logger.info(f"Gene Ontology: {self.id}, nodes: {len(ontology['nodes'])} (class: {len(self.class_nodes)}, property: {len(self.property_nodes)}), edges: {len(self.edges)}")
        
        # analysis
        self._log_stats()
    
    def _remove_deprecated_nodes(self, nodes):
        vaild_nodes = []
        for n in nodes:
            if n.get("meta") and n['meta'].get("deprecated") is True:
                continue
            vaild_nodes.append(n)
        self.logger.info(f"removed {len(nodes) - len(vaild_nodes)} deprecated nodes")
        return vaild_nodes
    
    def _log_stats(self):
        # edge values
        property_node_id_to_lbl = {p['id']: p['lbl'] for p in self.property_nodes}
        edge_vals, counts = np.unique([e['pred'] for e in self.edges], return_counts=True)
        edge_val_to_lbl = {str(edge_vals[i]): {'lbl': property_node_id_to_lbl.get(edge_vals[i]), 'count': int(counts[i])} 
                           for i in range(len(edge_vals))}
        for k, v in edge_val_to_lbl.items():
            self.logger.info(f"  {k} ({v['lbl'] if v['lbl'] else ''}): {v['count']} edges")
