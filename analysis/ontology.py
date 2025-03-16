from typing import Dict, Set
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
from utils.general import read_df, write_df
from typing import List, Dict
import json
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt


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

        self.type_bp = 'biological_process'
        self.type_mf = 'molecular_function'
        self.type_cc = 'cellular_component'

        self.type_part_of = 'part_of'
        self.type_regulates = 'regulates'
        self.type_neg_regulates = 'negatively_regulates' 
        self.type_pos_regulates = 'positively_regulates'
        self.type_is_a = 'is_a'
        self.type_sub_property_of = 'sub_property_of'
    
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
        G = nx.Graph()
        # add nodes
        for n in self.class_nodes:
            go_number = self._go_number_from_id(n['id'])
            lbl = n['lbl']
            meta = n['meta']
            for v in meta['basicPropertyValues']:
                if v['val'] == go_node_type:
                    G.add_node(go_number, type=go_node_type, lbl=lbl, meta=meta)
                    break

        # add edges
        edge_types = set()            
        for e in self.edges:
            sub_go_number = self._go_number_from_id(e['sub'])
            obj_go_number = self._go_number_from_id(e['obj'])
            edge_id = e['pred']
            edge_type = self.edge_id_to_type[edge_id]
            # add edge if both nodes exist
            if G.has_node(sub_go_number) and G.has_node(obj_go_number):
                G.add_edge(sub_go_number, obj_go_number, type=edge_type)
                edge_types.add(edge_type)
            # warn if missing a node
            if sum([G.has_node(sub_go_number), G.has_node(obj_go_number)]) == 1:
                self.logger.warning(f"missing node: {sub_go_number if not G.has_node(sub_go_number) else obj_go_number}")     
        
        self.logger.info(f"GO type: {go_node_type}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, edge types: {sorted(edge_types)}")
        return G 

    # def _compose_all_nx_graphs(self):
    #     # Combine all graphs into one
        

    #     # add edges between different types of nodes (e.g., BP to MF)
    #     edge_types = []           
    #     for e in self.edges:
    #         sub_go_number = self._go_number_from_id(e['sub'])
    #         obj_go_number = self._go_number_from_id(e['obj'])
    #         edge_id = e['pred']
    #         edge_type = self.edge_id_to_type[edge_id]
    #         # if edge already exists in BP, MF, or CC then skip
    #         if G.has_edge(sub_go_number, obj_go_number):
    #             continue
    #         # add edge if both nodes exist
    #         if G.has_node(sub_go_number) and G.has_node(obj_go_number):
    #             G.add_edge(sub_go_number, obj_go_number, type=edge_type)
    #             edge_types.append(edge_type)
    #         else:
    #             self.logger.warning(f"missing nodes: {sub_go_number if not G.has_node(sub_go_number) else ''}, {obj_go_number if not G.has_node(obj_go_number) else ''}")
    #     self.logger.info(f"Added {len(edge_types)} edges between BP, MF, CC. Edge types added: {sorted(set(edge_types))}")
    #     len(self.edges) == len(self.BP.number_of_edges) + len(self.MF.number_of_edges) + len(self.CC.number_of_edges) + len(edge_types)
    #     return G
    
        
    def ref_create_ontology_nx_graph(self):
        G = nx.Graph()

        # Add nodes with different types
        G.add_node("Person1", type="person")
        G.add_node("Person2", type="person")
        G.add_node("Company1", type="company")
        G.add_node("Company2", type="company")
        G.add_node("Event1", type="event")
        G.add_node("Event2", type="event")

        # Add edges between the nodes
        G.add_edge("Person1", "Company1")
        G.add_edge("Person2", "Company2")
        G.add_edge("Person1", "Event1")
        G.add_edge("Person2", "Event2")
        G.add_edge("Company1", "Event1")
        G.add_edge("Company2", "Event2")

        # Create another graph H
        H = nx.Graph()

        # Add nodes to the new graph H
        H.add_node("Person3", type="person")
        H.add_node("Company3", type="company")
        H.add_node("Event3", type="event")

        # Add edges within graph H
        H.add_edge("Person3", "Company3")
        H.add_edge("Company3", "Event3")
        H.add_edge("Event3", "Person3")

        # Add edges connecting nodes from H to nodes in G
        H.add_edge("Person3", "Company1")
        H.add_edge("Company3", "Event1")

        # Combine both graphs into a new graph
        F = nx.compose(G, H)

        # Draw the combined graph with different colors for different types of nodes
        pos = nx.spring_layout(F)
        node_colors = []
        for node in F.nodes(data=True):
            if node[1]['type'] == 'person':
                node_colors.append('blue')
            elif node[1]['type'] == 'company':
                node_colors.append('green')
            elif node[1]['type'] == 'event':
                node_colors.append('red')

        nx.draw(F, pos, with_labels=True, node_color=node_colors, node_size=3000, font_size=12, font_color='white')
        plt.show()
        return
