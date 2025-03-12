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
        
        self.G = nx.Graph()

        self.strains_data = data_loader.strains_data
        self.srna_acc_col = data_loader.srna_acc_col
        self.mrna_acc_col = data_loader.mrna_acc_col
        
        self.ontology = ontology

        # node types
        # GO
        self._bp = ontology.type_bp
        self._mf = ontology.type_mf
        self._cc = ontology.type_cc
        # mRNA
        self._mrna = "mrna"
        # sRNA
        self._srna = "srna"

        # edge types
        # GO <--> GO  
        self._part_of = ontology.type_part_of
        self._regulates = ontology.type_regulates
        self._neg_regulates = ontology.type_neg_regulates
        self._pos_regulates = ontology.type_pos_regulates
        self._is_a = ontology.type_is_a
        self._sub_property_of = ontology.type_sub_property_of
        # mRNA <--> GO
        self._annot = "annotation"
        # sRNA <--> mRNA     
        self._inter = "interaction"

    def build_graph(self):
        """
        each GO node is represented as a dict item:
            <id_str> : {'type': <str>, 'lbl': <str>, 'meta': <dict>}
        each mRNA node is represented as a dict item:
            <id_str> : {'type': <str>, 'name': <str>, 'sequence': <str>}
        each edge is represented as a dict item:
            (<id_str>, <id_str>) : {'type': <str>}
        """
        self.logger.info(f"building graph")
        # self._create_3D_visualization(self.ontology.BP)
        self._process_curated_annot()
        self._add_mrna_nodes_and_annotation_edges()
    
    def _process_curated_annot(self):
        for strain, data in self.strains_data.items():
            if 'all_mrna_w_curated_annot' in data.keys():
                cu_annot = data['all_mrna_w_curated_annot']  
                cu_annot[['GO_BP', 'GO_MF', 'GO_CC']] = pd.DataFrame(list(map(lambda x: self._split(x), cu_annot["GO_Terms"])))

                has_go = sum(pd.notnull(cu_annot["GO_Terms"]))
                has_bp = sum(cu_annot['GO_BP'].apply(lambda x: len(x) > 0))
                has_mf = sum(cu_annot['GO_MF'].apply(lambda x: len(x) > 0))
                has_cc = sum(cu_annot['GO_CC'].apply(lambda x: len(x) > 0))
                self.logger.info(f"{strain}: out of {has_go} currated annotations, BP = {has_bp} ({(has_bp/has_go)*100:.2f}%), MF = {has_mf} ({(has_mf/has_go)*100:.2f}%), CC = {has_cc} ({(has_cc/has_go)*100:.2f}%)")

    def _add_mrna_nodes_and_annotation_edges(self):
        for strain, data in self.strains_data.items():
            self.logger.info(f"adding mRNA annotations to ontology for {strain}")
            if 'all_mrna_w_curated_annot' in data.keys():
                self._add_mrna_and_curated_annot(strain, data['all_mrna_w_curated_annot'])
            elif 'all_mrna_w_ips_annot' in data.keys():
                # TODO: 
                # (3) match curated annotations to GO terms ontology (check with are PB, MF and CC)
                # (4) report stats
                self.logger.info(f"    all_mrna_w_ips_annot: {data['all_mrna_w_ips_annot'].shape}")
            # TODO: 
            # (5) compare curated and IPS annotations (optional)
            # (6) start thinking about the analysis (Sahar PPT)
            # (7) describe proprocessing in the latex paper
    
    def _add_rna_node(self, id, type, strain, locus_tag, name, synonyms, start, end, strand, sequence):

        print()
    
    def _add_mrna_annot(self, id):

        print()

    def _add_mrna_and_curated_annot(self, strain, cu_annot):
        self.logger.info(f"added mRNAs and curated annotations for {strain}")
        for i, r in cu_annot.iterrows():
            print()
            go_bp = r['GO_BP']
            # if pd.notnull(go_bp) and len(go_bp)
            self._add_rna_node(
                id=r['mRNA_accession_id'],
                type=self._mrna,
                strain=strain,
                locus_tag=r['mRNA_locus_tag'],
                name=r['mRNA_name'], 
                synonyms=['mRNA_name_synonyms'],
                start=['mRNA_start'],
                end=['mRNA_end'],
                strand=['mRNA_strand'],
                sequence=['mRNA_sequence']
                )

    
    def _split(self, go_terms: set):
        BP_go_ids, MF_go_ids, CC_go_ids = [], [], []
        
        if pd.notnull(go_terms):   
            for t in go_terms:
                go_id = t.split(':')[1]
                print()
                if go_id in self.ontology.BP.nodes:
                    BP_go_ids.append(go_id)
                elif go_id in self.ontology.MF.nodes:
                    MF_go_ids.append(go_id)
                elif go_id in self.ontology.CC.nodes:
                    CC_go_ids.append(go_id)
                else:
                    self.logger.warning(f"{go_id} is missing in ontology")

        return BP_go_ids, MF_go_ids, CC_go_ids
        
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
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line')

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
        net.show("nx_graph_new.html")

        return

