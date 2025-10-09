from typing import Dict, Set
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
from preprocessing.general_pr import convert_count_to_val
from typing import List, Dict
import pickle
import json
import sys
import os
import networkx as nx
# import matplotlib.pyplot as plt


ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

class Ontology:
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.info(f"initializing Ontology")
        self.config = config
        self.id: str = None
        self.class_nodes: List[dict] = None
        self.deprecated_nodes: List[dict] = []
        self.property_id_to_info: Dict[str, dict] = None
        self.edges = None
        
		# ---- Nodes properties ----
		# 	type
        self.type_bp = 'biological_process'
        self.type_mf = 'molecular_function'
        self.type_cc = 'cellular_component'
        # 	lbl
		# 	meta
        #   po2vec_embeddings (optional)
        self.emb_type_po2vec = 'po2vec_embeddings'

		# ---- Edge properties ----
		# 	type
        self.type_part_of = 'part_of'
        self.type_regulates = 'regulates'
        self.type_neg_regulates = 'negatively_regulates' 
        self.type_pos_regulates = 'positively_regulates'
        self.type_is_a = 'is_a'
        self.type_sub_property_of = 'sub_property_of'
        self.all_edge_types = [self.type_part_of, self.type_regulates, self.type_neg_regulates, self.type_pos_regulates, self.type_is_a, self.type_sub_property_of]
        
    @staticmethod
    def _go_number_from_id(original_id):
        return original_id.split('/')[-1].split('_')[-1]

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
        property_nodes = [n for n in nodes if n['type'] == 'PROPERTY']
        assert len(nodes) == len(self.class_nodes) + len(property_nodes)
        self.property_id_to_info = {pn['id']: {'lbl': pn['lbl'], 'meta': pn.get('meta', {})} for pn in property_nodes}

        # edges
        self.edges = ontology['edges']
        self.logger.info(f"Gene Ontology: {self.id}, nodes: {len(ontology['nodes'])} (class: {len(self.class_nodes)}, property: {len(property_nodes)}), edges: {len(self.edges)}")
        
        # analysis
        self._log_stats()
        
    def load_go_embeddings(self):
        self._load_po2vec_go_embeddings()
        return

    def _load_po2vec_go_embeddings(self):
        self.logger.info(f"loading PO2Vec GO embeddings...")
        embeddings_pkl_file = join(self.config['go_embeddings_dir'], self.config['po2vec_go_embeddings_pkl_file'])
        embeddings_df = pd.read_pickle(embeddings_pkl_file)
        go_id_to_emb = dict(zip(embeddings_df['terms'].apply(lambda x: x.split(':')[1]), 
                                embeddings_df['embeddings'].apply(lambda x: np.array(x))))
        # add embeddings to nodes
        self._add_embeddings_to_graph_nodes(self.emb_type_po2vec, go_id_to_emb)
    
    def _add_embeddings_to_graph_nodes(self, emb_type: str, node_id_to_emb: Dict[str, np.ndarray]):
        """
        Iterates over all nodes in self.BP, self.MF, and self.CC and add their embeddings vectors.
        """
        for graph, graph_name in [(self.BP, "BP"), (self.MF, "MF"), (self.CC, "CC")]:
            emb_count = 0
            for node_id in graph.nodes:
                emb = node_id_to_emb.get(node_id, None)
                if emb is not None:
                    graph.nodes[node_id][emb_type] = emb
                    emb_count += 1
            self.logger.info(f"{graph_name}: out of {len(graph.nodes)} nodes, {emb_count} have {emb_type} ({(emb_count/len(graph.nodes))*100:.2f}%)")
        
    def get_deprecated_node_ids(self) -> List[str]:
        dep_go_ids = []
        for n in self.deprecated_nodes:
            dep_go_ids.append(self._go_number_from_id(n['id']))
        return dep_go_ids
    
    def _remove_deprecated_nodes(self, nodes):
        vaild_nodes = []
        for n in nodes:
            if n.get("meta") and n['meta'].get("deprecated") is True:
                self.deprecated_nodes.append(n)
                continue
            vaild_nodes.append(n)
        self.logger.info(f"removed {len(nodes) - len(vaild_nodes)} deprecated nodes")
        return vaild_nodes
    
    def _log_stats(self):
        # edge values
        edge_vals, counts = np.unique([e['pred'] for e in self.edges], return_counts=True)
        edge_val_to_info = {str(edge_vals[i]): {'lbl': self.property_id_to_info.get(edge_vals[i], {}).get('lbl'), 'count': int(counts[i])} 
                           for i in range(len(edge_vals))}
        for k, v in edge_val_to_info.items():
            self.logger.info(f"  {k} ({v['lbl'] if v['lbl'] else ''}): {v['count']} edges")

    def create_ontology_nx_graphs(self):
        self._map_edge_id_to_type()
        self.BP = self._create_nx_graph(self.type_bp)
        self.MF = self._create_nx_graph(self.type_mf)
        self.CC = self._create_nx_graph(self.type_cc)
        # self.G = nx.compose_all([self.BP, self.MF, self.CC])
        # Example in self.BP:
        #        '0099590'  neurotransmitter receptor internalization
        # <is a> '0031503'  protein-containing complex localization
        # <is a> '0031623'  receptor internalization

    def _map_edge_id_to_type(self):
        self.edge_id_to_type = {
       "http://purl.obolibrary.org/obo/BFO_0000050": self.type_part_of,  # 6583 edges
       "http://purl.obolibrary.org/obo/RO_0002211": self.type_regulates,  # 2996 edges
       "http://purl.obolibrary.org/obo/RO_0002212": self.type_neg_regulates,  # 2618 edges
       "http://purl.obolibrary.org/obo/RO_0002213": self.type_pos_regulates,  # 2624 edges
        "is_a": self.type_is_a,  # 62678 edges
        "subPropertyOf": self.type_sub_property_of  # 2 edges
        }

    def _create_nx_graph(self, go_node_type):
        """
        Note: 2 types of meta['basicPropertyValues']['pred']
            "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"  -->   meta['basicPropertyValues']['val'] is BP, MF, CC 
            "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId"
        """
        G = nx.MultiDiGraph()
        # 1 - add nodes
        for n in self.class_nodes:
            go_number = self._go_number_from_id(n['id'])
            lbl = n['lbl']
            meta = n['meta']
            for v in meta['basicPropertyValues']:
                if v['val'] == go_node_type:
                    G.add_node(go_number, type=go_node_type, lbl=lbl, meta=meta)
                    break
        # 2 - add edges
        edge_type_to_count = dict(zip(self.all_edge_types, [0]*len(self.all_edge_types)))            
        for e in self.edges:
            sub_go_number = self._go_number_from_id(e['sub'])
            obj_go_number = self._go_number_from_id(e['obj'])
            edge_id = e['pred']
            edge_type = self.edge_id_to_type[edge_id]
            # add edge if both nodes exist
            if G.has_node(sub_go_number) and G.has_node(obj_go_number):
                G.add_edge(sub_go_number, obj_go_number, type=edge_type)
                edge_type_to_count[edge_type] = edge_type_to_count[edge_type] + 1
            # warn if missing a single node
            if sum([G.has_node(sub_go_number), G.has_node(obj_go_number)]) == 1:
                self.logger.warning(f"missing nodes: {[go for go in (sub_go_number, obj_go_number) if not G.has_node(go)]}")    
        
        assert G.number_of_edges() == sum(edge_type_to_count.values())
        
        # 3 - log

        rec = {
            "latex_symbolic_x_coords": "",
            "latex_coordinates": ""
        }
        


        self.logger.info(f"GO type: {go_node_type}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, edge types: {sorted(edge_types)}")

        return G
