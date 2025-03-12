from typing import Dict
import pandas as pd
import numpy as np
from pathlib import Path
from utils.general import read_df, write_df
import networkx as nx
from pyvis.network import Network
import json
import sys
import os

ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")


class GraphBuilder:
    def __init__(self, config, logger, data_loader, ontology):
        self.logger = logger
        self.logger.info(f"initializing GraphBuilder")
        self.config = config
        
        self.strains_data = data_loader.strains_data
        self.srna_acc_col = data_loader.srna_acc_col
        self.mrna_acc_col = data_loader.mrna_acc_col
        self.ontology = ontology

    def build_graph(self):
        self.logger.info(f"building graph")
        self._create_3D_visualization(self.ontology.BP)
        self._map_annotations_to_ontology()
    
    def _create_3D_visualization(self, nx_graph):
        # nx_graph = nx.Graph()

        # # Add nodes with different types
        # nx_graph.add_node("Person1", type="person")
        # nx_graph.add_node("Person2", type="person")
        # nx_graph.add_node("Company1", type="company")
        # nx_graph.add_node("Company2", type="company")
        # nx_graph.add_node("Event1", type="event")
        # nx_graph.add_node("Event2", type="event")

        # # Add edges between the nodes
        # nx_graph.add_edge("Person1", "Company1")
        # nx_graph.add_edge("Person2", "Company2")
        # nx_graph.add_edge("Person1", "Event1")
        # nx_graph.add_edge("Person2", "Event2")
        # nx_graph.add_edge("Company1", "Event1")
        # nx_graph.add_edge("Company2", "Event2")
        self.logger.info("Network...")
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

        # Add nodes to the Pyvis network with different colors for different types
        self.logger.info("Add nodes")
        for node, data in nx_graph.nodes(data=True):
            if data['type'] == 'person':
                net.add_node(node, label=node, color='blue')
            elif data['type'] == 'company':
                net.add_node(node, label=node, color='green')
            elif data['type'] == 'event':
                net.add_node(node, label=node, color='red')
            else:
                net.add_node(node, label=node, color='gray')

        # Add edges to the Pyvis network
        self.logger.info("Add edges")
        for edge in nx_graph.edges():
            net.add_edge(edge[0], edge[1])

        # Generate the network visualization and save it as an HTML file
        self.logger.info("Dump html")
        net.show("nx_graph.html")

        return

    def _map_annotations_to_ontology(self):
        for strain, data in self.strains_data.items():
            self.logger.info(f"mapping annotations to ontology for {strain}")
            if 'all_mrna_w_curated_annot' in data.keys():
                self._map_curated_annot_to_ontology(data['all_mrna_w_curated_annot'])
                # TODO: 
                # (1) match curated annotations to GO terms ontology (check with are PB, MF and CC)
                # (2) report stats
                self.logger.info(f"    all_mrna_w_curated_annot: {data['all_mrna_w_curated_annot'].shape}")
            if 'all_mrna_w_ips_annot' in data.keys():
                # TODO: 
                # (3) match curated annotations to GO terms ontology (check with are PB, MF and CC)
                # (4) report stats
                self.logger.info(f"    all_mrna_w_ips_annot: {data['all_mrna_w_ips_annot'].shape}")
            # TODO: 
            # (5) compare curated and IPS annotations (optional)
            # (6) start thinking about the analysis (Sahar PPT)
            # (7) describe proprocessing in the latex paper
    
    def _map_curated_annot_to_ontology(self, currated_annot: pd.DataFrame):
        print()
  