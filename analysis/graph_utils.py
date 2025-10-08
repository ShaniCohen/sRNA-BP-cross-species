from typing import List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from utils.general import read_df, write_df
import networkx as nx
from pyvis.network import Network
import json
import sys
import os

ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

class GraphUtils:
    def __init__(self, config, logger, data_loader, ontology):
        self.logger = logger
        self.logger.info(f"initializing GraphUtils")
        self.config = config

        # strains
        self.strains = data_loader.get_strains()
        self.strain_nm_to_short = {
            'ecoli_k12': 'E.coli K12', 
            'ecoli_epec': 'E.coli EPEC', 
            'salmonella': 'Salmonella', 
            'klebsiella': "Klebsiella", 
            'vibrio': "Vibrio", 
            'pseudomonas': "Pseudomonas"
        }
        assert len(set(self.strains) - set(self.strain_nm_to_short.keys())) == 0, "misalignment between strain names in GraphUtils & data_loader"

        # ----- Graph properties
        # node types
        # GO
        self.bp = ontology.type_bp
        self.mf = ontology.type_mf
        self.cc = ontology.type_cc
        self.go_types = [self.bp, self.mf, self.cc]
        # mRNA
        self.mrna = "mrna"
        # sRNA
        self.srna = "srna"

        # edge types
        # GO --> GO  
        self.part_of = ontology.type_part_of
        self.regulates = ontology.type_regulates
        self.neg_regulates = ontology.type_neg_regulates
        self.pos_regulates = ontology.type_pos_regulates
        self.is_a = ontology.type_is_a
        self.sub_property_of = ontology.type_sub_property_of
        # mRNA --> GO
        self.annotated = "annotated"
        # sRNA --> mRNA     
        self.targets = "targets"
        # RNA <--> RNA
        # paralogs: same strain
        self.paralog = "paralog"    
        # orthologs: different strains
        self.ortholog_by_seq = "ortholog_by_seq"
        self.ortholog_by_name = "ortholog_by_name"

        # edge annot types (annot_type)
        self.curated = "curated"
        self.ips = "interproscan"
        self.eggnog = "eggnog"
        self.annot_types = [self.curated, self.ips, self.eggnog]

        # ----- Methods
        # BP similarity methods
        self.exact_bp = 'exact_bp'

    def add_node_rna(self, G, id, type, strain, locus_tag, name, synonyms, start, end, strand, rna_seq, protein_seq: str = None, log_warning=True):
        if not G.has_node(id):
            assert type in [self.srna, self.mrna], f"invalid RNA type: {type}"
            assert strain in self.strains, f"strain {strain} is not in the list of strains: {self.strains}"
            if pd.notnull(protein_seq):
                G.add_node(id, type=type, strain=strain, locus_tag=locus_tag, name=name, synonyms=synonyms, start=start, end=end, strand=strand, rna_seq=rna_seq, protein_seq=protein_seq)
            else:
                G.add_node(id, type=type, strain=strain, locus_tag=locus_tag, name=name, synonyms=synonyms, start=start, end=end, strand=strand, rna_seq=rna_seq)
        elif log_warning:
            self.logger.warning(f"{type} node {id} already in graph G")
        return G

    def add_edge_srna_mrna_inter(self, G, srna_node_id, mrna_node_id):
        """ Add "targets" edge from the sRNA node to the mRNA node

        Args:
            srna_node_id (str): the sRNA node id (accession id)
            mrna_node_id (str): the mRNA node id (accession id)
        """
        assert G.has_node(srna_node_id) and G.has_node(mrna_node_id)
        assert G.nodes[srna_node_id]['strain'] == G.nodes[mrna_node_id]['strain']
        assert G.nodes[srna_node_id]['type'] == self.srna
        assert G.nodes[mrna_node_id]['type'] == self.mrna
        G.add_edge(srna_node_id, mrna_node_id, type=self.targets)
        return G
        
    def add_edge_mrna_go_annot(self, G, mrna_node_id, go_id, annot_type) -> bool:
        """ Add "annotated" edge from the mRNA node to the GO node.
        Args:
            mrna_node_id (str): the mRNA node id (accession id)
            go_id (str): the GO id

        Returns:
            bool: whtether the go_id is missing in the ontology
        """
        assert G.has_node(mrna_node_id), f"mRNA id {mrna_node_id} is missing in the graph"
        if G.has_node(go_id):
            assert G.nodes[mrna_node_id]['type'] == self.mrna
            assert G.nodes[go_id]['type'] in self.go_types
            assert annot_type in self.annot_types
            G.add_edge(mrna_node_id, go_id, type=self.annotated, annot_type=annot_type)
            return G, False
        return G, True
    
    def add_edges_rna_rna_paralogs(self, G, rna_node_id_1, rna_node_id_2):
        """ Add "targets" edge from the sRNA node to the mRNA node

        Args:
            srna_node_id (str): the sRNA node id (accession id)
            mrna_node_id (str): the mRNA node id (accession id)
        """
        assert G.has_node(rna_node_id_1) and G.has_node(rna_node_id_2)
        assert G.nodes[rna_node_id_1]['strain'] == G.nodes[rna_node_id_2]['strain']
        both_srna = (G.nodes[rna_node_id_1]['type'] == self.srna) & (G.nodes[rna_node_id_2]['type'] == self.srna)
        both_mrna = (G.nodes[rna_node_id_2]['type'] == self.mrna) & (G.nodes[rna_node_id_2]['type'] == self.mrna)
        assert both_srna or both_mrna
        G.add_edge(rna_node_id_1, rna_node_id_2, type=self.paralog)
        G.add_edge(rna_node_id_2, rna_node_id_1, type=self.paralog)
        return G
    
    def add_edges_rna_rna_orthologs_by_seq(self, G, rna_node_id_1, rna_node_id_2):
        """ Add "ortholog_by_seq" edge between two RNA nodes of different strains

        Args:
            rna_node_id_1 (str): the first RNA node id (accession id)
            rna_node_id_2 (str): the second RNA node id (accession id)
        """
        self._validate_rna_rna_orthologs(G, rna_node_id_1, rna_node_id_2)
        G.add_edge(rna_node_id_1, rna_node_id_2, type=self.ortholog_by_seq)
        G.add_edge(rna_node_id_2, rna_node_id_1, type=self.ortholog_by_seq)
        return G

    def add_edges_rna_rna_orthologs_by_name(self, G, rna_node_id_1, rna_node_id_2):
        """ Add "ortholog_by_name" edge between two RNA nodes of different strains

        Args:
            rna_node_id_1 (str): the first RNA node id (accession id)
            rna_node_id_2 (str): the second RNA node id (accession id)
        """
        self._validate_rna_rna_orthologs(G, rna_node_id_1, rna_node_id_2)
        G.add_edge(rna_node_id_1, rna_node_id_2, type=self.ortholog_by_name)
        G.add_edge(rna_node_id_2, rna_node_id_1, type=self.ortholog_by_name)
        return G
    
    def _validate_rna_rna_orthologs(self, G, rna_node_id_1, rna_node_id_2):
        assert G.has_node(rna_node_id_1) and G.has_node(rna_node_id_2)
        assert G.nodes[rna_node_id_1]['strain'] != G.nodes[rna_node_id_2]['strain']
        both_srna = (G.nodes[rna_node_id_1]['type'] == self.srna) & (G.nodes[rna_node_id_2]['type'] == self.srna)
        both_mrna = (G.nodes[rna_node_id_2]['type'] == self.mrna) & (G.nodes[rna_node_id_2]['type'] == self.mrna)
        assert both_srna or both_mrna

    def is_target(self, G, srna_node_id, mrna_node_id):
        """ Check if there is an interaction edge between sRNA and mRNA nodes """
        assert G.has_node(srna_node_id) and G.has_node(mrna_node_id)
        if \
            G.nodes[srna_node_id]['strain'] == G.nodes[mrna_node_id]['strain'] and \
            G.nodes[srna_node_id]['type'] == self.srna and \
            G.nodes[mrna_node_id]['type'] == self.mrna:
            for d in G[srna_node_id][mrna_node_id].values():
                if d['type'] == self.targets:
                    return True
        return False
    
    def is_annotated(self, G, mrna_node_id, go_node_id, go_node_type, annot_type=None):
        """ Check if there is an annotation edge from mRNA to GO node."""
        assert G.has_node(mrna_node_id) and G.has_node(go_node_id)
        assert go_node_type in self.go_types, f"invalid GO node type: {go_node_type}"
        if annot_type is not None:
            assert annot_type in self.annot_types, f"invalid annotation type: {annot_type}"
        if \
            G.nodes[mrna_node_id]['type'] == self.mrna and \
            G.nodes[go_node_id]['type'] == go_node_type:
            for d in G[mrna_node_id][go_node_id].values():
                if d['type'] == self.annotated and (annot_type is None or d['annot_type'] == annot_type):
                    return True
        return False
    
    def are_paralogs(self, G, rna_node_id_1, rna_node_id_2, strain):
        """ Check if there are paralog edges between two RNA nodes of the same strain """
        assert G.has_node(rna_node_id_1) and G.has_node(rna_node_id_2)
        assert strain in self.strains, f"strain {strain} is not in the list of strains: {self.strains}"

        same_strain = G.nodes[rna_node_id_1]['strain'] == G.nodes[rna_node_id_2]['strain'] == strain
        both_srna = (G.nodes[rna_node_id_1]['type'] == self.srna) & (G.nodes[rna_node_id_2]['type'] == self.srna)
        both_mrna = (G.nodes[rna_node_id_2]['type'] == self.mrna) & (G.nodes[rna_node_id_2]['type'] == self.mrna)
        
        if same_strain and (both_srna or both_mrna):
            is_paralog_1_2 = False
            if G.has_edge(rna_node_id_1, rna_node_id_2):
                for d in G[rna_node_id_1][rna_node_id_2].values():
                    if d['type'] == self.paralog:
                        is_paralog_1_2 = True
                        break
            is_paralog_2_1 = False
            if G.has_edge(rna_node_id_2, rna_node_id_1):
                for d in G[rna_node_id_2][rna_node_id_1].values():
                    if d['type'] == self.paralog:
                        is_paralog_2_1 = True
                        break
            assert is_paralog_1_2 == is_paralog_2_1, "Paralog edge should be symmetric"
            return is_paralog_1_2 and is_paralog_2_1
        return False
    
    def _are_orthologs(self, G, rna_node_id_1, rna_node_id_2, strain_1, strain_2, ortholog_edge_type):
        """ Check if there are ortholog edges between two RNA nodes of different strains """
        assert G.has_node(rna_node_id_1) and G.has_node(rna_node_id_2)
        assert strain_1 in self.strains, f"strain {strain_1} is not in the list of strains: {self.strains}"
        assert strain_2 in self.strains, f"strain {strain_2} is not in the list of strains: {self.strains}"
        assert G.nodes[rna_node_id_1]['strain'] == strain_1
        assert G.nodes[rna_node_id_2]['strain'] == strain_2

        different_strains = strain_1 != strain_2
        both_srna = (G.nodes[rna_node_id_1]['type'] == self.srna) & (G.nodes[rna_node_id_2]['type'] == self.srna)
        both_mrna = (G.nodes[rna_node_id_2]['type'] == self.mrna) & (G.nodes[rna_node_id_2]['type'] == self.mrna)
        
        if different_strains and (both_srna or both_mrna):
            is_ortholog_1_2 = False
            if G.has_edge(rna_node_id_1, rna_node_id_2):
                for d in G[rna_node_id_1][rna_node_id_2].values():
                    if d['type'] == ortholog_edge_type:
                        is_ortholog_1_2 = True
                        break
            is_ortholog_2_1 = False
            if G.has_edge(rna_node_id_2, rna_node_id_1):
                for d in G[rna_node_id_2][rna_node_id_1].values():
                    if d['type'] == ortholog_edge_type:
                        is_ortholog_2_1 = True
                        break
            assert is_ortholog_1_2 == is_ortholog_2_1, f"Ortholog edge {ortholog_edge_type} should be symmetric"
            return is_ortholog_1_2 and is_ortholog_2_1
        return False
    
    def are_orthologs_by_seq(self, G, rna_node_id_1, rna_node_id_2, strain_1, strain_2):
        """ Check if there are ortholog_by_seq edges between two RNA nodes of different strains """
        edge_type = self.ortholog_by_seq
        return self._are_orthologs(G, rna_node_id_1, rna_node_id_2, strain_1, strain_2, edge_type)
    
    def are_orthologs_by_name(self, G, rna_node_id_1, rna_node_id_2, strain_1, strain_2):
        """ Check if there are ortholog_by_name edges between two RNA nodes of different strains """
        edge_type = self.ortholog_by_name
        return self._are_orthologs(G, rna_node_id_1, rna_node_id_2, strain_1, strain_2, edge_type)
    
    def get_orthologs_cluster(self, G, rna_node_id, cluster) -> set:
        """ Get all orthologs of a given RNA node in the graph G. ortholog is a transitive relation, so all orthologs of the orthologs are also included. """
        assert G.has_node(rna_node_id)
        assert G.nodes[rna_node_id]['type'] in [self.srna, self.mrna], f"RNA node {rna_node_id} has invalid type: {G.nodes[rna_node_id]['type']}"
        
        if rna_node_id not in cluster:
            cluster.add(rna_node_id)
            rna_strain = G.nodes[rna_node_id]['strain']
            # get direct orthologs
            direct_orthologs = [neighbor for neighbor in G.neighbors(rna_node_id) if G.nodes[neighbor]['type'] in [self.srna, self.mrna] and \
                                (self.are_orthologs_by_seq(G, rna_node_id, neighbor, rna_strain, G.nodes[neighbor]['strain']) or \
                                 self.are_orthologs_by_name(G, rna_node_id, neighbor, rna_strain, G.nodes[neighbor]['strain']))]
            for o in direct_orthologs:
                cluster = self.get_orthologs_cluster(G, o, cluster)
        
        return cluster
    
    def get_paralogs_only_cluster(self, G, strain, rna_node_id, cluster) -> set:
        """ Get all paralogs of a given RNA node in the graph G. paralogs is a transitive relation, so all paralogs of the paralogs are also included. """
        assert G.has_node(rna_node_id)
        assert G.nodes[rna_node_id]['type'] in [self.srna, self.mrna], f"RNA node {rna_node_id} has invalid type: {G.nodes[rna_node_id]['type']}"
        
        if rna_node_id not in cluster:
            cluster.add(rna_node_id)
            # get direct paralogs
            direct_paralogs = [neighbor for neighbor in G.neighbors(rna_node_id) if G.nodes[neighbor]['type'] in [self.srna, self.mrna] and \
                               self.are_paralogs(G, rna_node_id, neighbor, strain)]
            for p in direct_paralogs:
                cluster = self.get_paralogs_only_cluster(G, strain, p, cluster)

        return cluster
    
    def get_short_strain_nm(self, strain_nm: str) -> str:
        return self.strain_nm_to_short[strain_nm]
    
    def get_common_bps(self, bps1: List[str], bps2: List[str], bp_similiarity_method: str) -> List[str]:
        if bp_similiarity_method == self.exact_bp:
            return np.intersect1d(bps1, bps2).tolist()
        else:
            raise ValueError(f"invalid BP similiarity method -> {bp_similiarity_method}")
        
    def _create_3D_visualization(self, G):
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
        for node, data in G.nodes(data=True):
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
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        # Generate the network visualization and save it as an HTML file
        self.logger.info("Dump html")
        net.show("nx_graph_new.html")

        return

