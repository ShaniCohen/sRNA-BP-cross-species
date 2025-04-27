from typing import List
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

class GraphBuilder:
    def __init__(self, config, logger, data_loader, ontology):
        self.logger = logger
        self.logger.info(f"initializing GraphBuilder")
        self.config = config

        self.strains_data = data_loader.strains_data
        self.srna_acc_col = data_loader.srna_acc_col
        self.mrna_acc_col = data_loader.mrna_acc_col
        
        self.ontology = ontology
        self.curated_go_ids_missing_in_ontology = set()
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(ontology.BP.nodes(data=True))
        self.G.add_edges_from(ontology.BP.edges(data=True))
        self.graph_is_built = False

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
        # GO --> GO  
        self._part_of = ontology.type_part_of
        self._regulates = ontology.type_regulates
        self._neg_regulates = ontology.type_neg_regulates
        self._pos_regulates = ontology.type_pos_regulates
        self._is_a = ontology.type_is_a
        self._sub_property_of = ontology.type_sub_property_of
        # mRNA --> GO
        self._annotated = "annotated"
        # sRNA --> mRNA     
        self._targets = "targets"
    
    def get_strains(self) -> List[str]:
        return list(self.strains_data.keys())
    
    def get_graph(self) -> nx.Graph:
        if not self.graph_is_built:
            raise Exception("Graph is not built yet. Please call build_graph() first.")
        return self.G

    def build_graph(self):
        """
        GO node is represented as a dict item:
            <id_str> : {'type': <str>, 'lbl': <str>, 'meta': <dict>}
        mRNA/sRNA node is represented as a dict item:
            <id_str> : {'type': <str>, 'strain': <str>, 'locus_tag': <str>, 'name': <str>, 'synonyms': <List[str]>, 'start': <float>, 'end': <float>, 'strand': <str>, 'sequence': <str>}
        edge is represented as a dict item:
            (<id_str>, <id_str>) : {'type': <str>}
        """
        self.logger.info(f"building graph")
        # self._create_3D_visualization(self.ontology.BP)
        self._process_curated_annot()
        self._add_mrna_nodes_and_annotation_edges()
        self._add_srna_nodes_and_interaction_edges()
        self._log_graph_info()
        self.graph_is_built = True
        self.logger.info(f"graph is built")
    
    def _process_curated_annot(self):
        for strain, data in self.strains_data.items():
            prev_missing = self.curated_go_ids_missing_in_ontology.copy()
            if 'all_mrna_w_curated_annot' in data.keys():
                cu_annot = data['all_mrna_w_curated_annot']  
                cu_annot[['GO_BP', 'GO_MF', 'GO_CC']] = pd.DataFrame(list(map(lambda x: self._split(x), cu_annot["GO_Terms"])))
                new_missing = self.curated_go_ids_missing_in_ontology - prev_missing
                deprecated = [i for i in new_missing if i in self.ontology.get_deprecated_node_ids()]
                if len(new_missing) > 0:
                    self.logger.warning(f"{strain}: {len(new_missing)} curated GO ids are missing in ontology, {len(deprecated)} of them are deprecated")

                has_go = sum(pd.notnull(cu_annot["GO_Terms"]))
                has_bp = sum(cu_annot['GO_BP'].apply(lambda x: len(x) > 0))
                has_mf = sum(cu_annot['GO_MF'].apply(lambda x: len(x) > 0))
                has_cc = sum(cu_annot['GO_CC'].apply(lambda x: len(x) > 0))
                self.logger.info(f"{strain}: out of {has_go} currated annotations, BP = {has_bp} ({(has_bp/has_go)*100:.2f}%), MF = {has_mf} ({(has_mf/has_go)*100:.2f}%), CC = {has_cc} ({(has_cc/has_go)*100:.2f}%)")

    def _add_mrna_nodes_and_annotation_edges(self):
        for strain, data in self.strains_data.items():
            if 'all_mrna_w_curated_annot' in data.keys():
                self._add_all_mrna_and_curated_bp_annot(strain, data['all_mrna_w_curated_annot'])
                # Example mRNA = 'EG10001', GO BP = ['0006522', '0030632', '0071555', '0009252', '0008360']
            elif 'all_mrna_w_ips_annot' in data.keys():
                self._add_all_mrna_and_ips_bp_annot(strain, data['all_mrna_w_ips_annot'])
            # describe proprocessing in the latex paper
            
            self._assert_mrna_nodes_addition(strain)

            
    def _add_srna_nodes_and_interaction_edges(self):
        for strain, data in self.strains_data.items():
            self.logger.info(f"adding sRNA nodes and sRNA-mRNA interactions for {strain}")
            # 1 - add all sRNA nodes to graph
            for _, r in data['all_srna'].iterrows():
                srna_node_id = r[self.srna_acc_col]
                self._add_node_rna(id=srna_node_id, type=self._srna, strain=strain, locus_tag=r['sRNA_locus_tag'], 
								   name=r['sRNA_name'], synonyms=r['sRNA_name_synonyms'], start=r['sRNA_start'], end=r['sRNA_end'],
								   strand=r['sRNA_strand'], sequence=r['sRNA_sequence'])
            
            # TODO: Decide how to use interactions data (growth cond, hfq, only pos inter, count?)
            # 2 - add sRNA-mRNA interaction edges
            for _, r in data['unq_inter'].iterrows():
                # 2.1 - add the interaction edges between sRNA and mRNA nodes
                srna_node_id = r[self.srna_acc_col]
                mrna_node_id = r[self.mrna_acc_col]
                self._add_edge_srna_mrna_inter(srna_node_id, mrna_node_id)
            
            self._assert_srna_nodes_addition(strain)
            self._assert_srna_mrna_inter_addition(strain)
    
    def _log_graph_info(self):
        self.logger.info("Logging graph information...")
        strains = self.get_strains()
        
        for strain in strains:
            # --------------------------------
            #              mRNA
            # --------------------------------
            # Filter mRNA nodes for the current strain
            mrna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self._mrna and d['strain'] == strain]
            mrna_count = len(mrna_nodes)

            # Count mRNA nodes with interactions (sRNA-mRNA)
            # The edge ('0030632', 'EG10001') is not in the graph.
            # mRNA = 'EG10001'
            # sorted([(u, v) for u, v, d in self.G.edges(data=True) if u == 'EG10001' or v == 'EG10001'])
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter'][self.mrna_acc_col] == 'EG10001']
            # TODO: fix the issue with the edge not being in the graph
            mrna_with_interactions = [
                n for n in mrna_nodes if any(
                    self.G.edges[neighbor, n]['type'] == self._targets and
                    self.G.nodes[neighbor]['type'] == self._srna
                    for neighbor in self.G.neighbors(n)
                )
            ]
            mrna_with_interactions_count = len(mrna_with_interactions)

            # mRNA nodes with interactions (sRNA-mRNA) and BP GO annotations
            mrna_bp_annotations = {
                n: [
                    neighbor for neighbor in self.G.neighbors(n)
                    if self.G.edges[n, neighbor]['type'] == self._annot and 
                    self.G.nodes[neighbor]['type'] == self._bp  # node_types = [self._bp, self._mf, self._cc]
                ]
                for n in mrna_with_interactions
            }

            emb_type = self.ontology.emb_type_po2vec
            mrna_w_bp_annot = []
            mrna_w_bp_annot_and_at_least_1_emb = []
            for n, bp_nodes in mrna_bp_annotations.items():
                if len(bp_nodes) > 0:
                    mrna_w_bp_annot.append(n)
                    # Check if any of the BP nodes have embeddings
                    if any(emb_type in self.G.nodes[go_id] for go_id in bp_nodes):
                        mrna_w_bp_annot_and_at_least_1_emb.append(n)
            
            mrna_w_bp_annot_count = len(mrna_w_bp_annot)
            mrna_w_bp_annot_and_at_least_1_emb_count = len(mrna_w_bp_annot_and_at_least_1_emb)
            
            # --------------------------------
            #              sRNA
            # --------------------------------
            # Filter sRNA nodes for the current strain
            srna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self._srna and d['strain'] == strain]
            srna_count = len(srna_nodes)

            # Count sRNA nodes with interactions (sRNA-mRNA)
            srna_with_interactions = [
                n for n in srna_nodes if any(
                    self.G.edges[n, neighbor]['type'] == self._targets and
                    self.G.nodes[neighbor]['type'] == self._mrna
                    for neighbor in self.G.neighbors(n)
                )
            ]
            srna_with_interactions_count = len(srna_with_interactions)

            # Count sRNA nodes with interactions (sRNA-mRNA) and BP GO annotations
            srna_w_bp_annot = [
                n for n in srna_nodes if any(
                    neighbor in mrna_w_bp_annot for neighbor in self.G.neighbors(n)
                )
            ]
            srna_w_bp_annot_count = len(srna_w_bp_annot)

            # --------------------------------
            #        Log the information
            # --------------------------------
            self.logger.info(
                f"\n   Strain: {strain} "
                f"\n   _______________________________ "
                f"\n   mRNA count: {mrna_count} "
                f"\n   mRNA with interactions: {mrna_with_interactions_count} "
                f"\n   mRNA with interactions and BP annotations: {mrna_w_bp_annot_count} "
                f"\n   mRNA with interactions and BP annotations and at least 1 embedding: {mrna_w_bp_annot_and_at_least_1_emb_count} "
                f"\n   sRNA count: {srna_count} "
                f"\n   sRNA with interactions: {srna_with_interactions_count} "
                f"\n   sRNA with interactions and BP annotations: {srna_w_bp_annot_count} "
            )
            print()
    
    def _add_all_mrna_and_curated_bp_annot(self, strain, all_mrna_w_curated_annot):
        self.logger.info(f"adding mRNA nodes and curated mRNA-GO annotations for {strain}")
        assert sum(pd.isnull(all_mrna_w_curated_annot['mRNA_accession_id'])) == 0
        bp_count, missing_bp_ids = 0, []

        for _, r in all_mrna_w_curated_annot.iterrows():
            # 1 - add the mRNA node to graph
            mrna_node_id = r[self.mrna_acc_col]
            self._add_node_rna(id=mrna_node_id, type=self._mrna, strain=strain, locus_tag=r['mRNA_locus_tag'], 
                               name=r['mRNA_name'], synonyms=r['mRNA_name_synonyms'], start=r['mRNA_start'], end=r['mRNA_end'],
                               strand=r['mRNA_strand'], sequence=r['mRNA_sequence'])
            # 2 - add annotation edge between the mRNA node and its BP nodes
            go_bp_ids = r['GO_BP']
            if isinstance(go_bp_ids, list) and len(go_bp_ids) > 0:
                for bp_id in go_bp_ids:
                    bp_id_is_missing = self._add_edge_mrna_go_annot(mrna_node_id, bp_id)
                    if bp_id_is_missing:
                        missing_bp_ids.append(bp_id)
                    bp_count += 1
        dep = [m for m in set(missing_bp_ids) if m in self.ontology.deprecated_nodes]
        self.logger.info(f"{strain}: out of {bp_count} curated BP annotations, missing: {len(missing_bp_ids)} ({len(set(missing_bp_ids))} unique), deprecated: {len(dep)}")
                    
    def _add_all_mrna_and_ips_bp_annot(self, strain, all_mrna_w_ips_annot):
        self.logger.info(f"adding mRNA nodes and InterProScan mRNA-GO annotations for {strain}")
        assert sum(pd.isnull(all_mrna_w_ips_annot['mRNA_accession_id'])) == 0
        bp_count, missing_bp_dicts = 0, []

        for _, r in all_mrna_w_ips_annot.iterrows():
            # 1 - add the mRNA node to graph
            mrna_node_id = r[self.mrna_acc_col]
            self._add_node_rna(id=mrna_node_id, type=self._mrna, strain=strain, locus_tag=r['mRNA_locus_tag'], 
                               name=r['mRNA_name'], synonyms=r['mRNA_name_synonyms'], start=r['mRNA_start'], end=r['mRNA_end'],
                               strand=r['mRNA_strand'], sequence=r['mRNA_sequence'])
            # 2 - add annotation edges between the mRNA node and BP nodes
            bp_go_xrefs = r['BP_go_xrefs']
            if isinstance(bp_go_xrefs, list) and len(bp_go_xrefs) > 0:
                for bp_dict in bp_go_xrefs:
                    bp_id = bp_dict['id'].split(":")[1]
                    bp_id_is_missing = self._add_edge_mrna_go_annot(mrna_node_id, bp_id)
                    if bp_id_is_missing:
                        missing_bp_dicts.append(bp_dict)
                    bp_count += 1
        # log
        unq_missing = set([x['id'] for x in missing_bp_dicts])
        dep = [m for m in unq_missing if m in self.ontology.deprecated_nodes]
        self.logger.info(f"{strain}: out of {bp_count} InterProScan BP annotations, missing: {len(missing_bp_dicts)} ({len(unq_missing)} unique), deprecated: {len(dep)}")
    
    def _assert_mrna_nodes_addition(self, strain):
        raw_mrna = sorted(self.strains_data[strain]['all_mrna'][self.mrna_acc_col])
        graph_mrna = sorted([n for n, d in self.G.nodes(data=True) if d['type'] == self._mrna and d['strain'] == strain])
        assert raw_mrna == graph_mrna, f"mRNA nodes in the graph do not match the raw data for strain {strain}"
    
    def _assert_srna_nodes_addition(self, strain):
        raw_srna = sorted(self.strains_data[strain]['all_srna'][self.srna_acc_col])
        graph_srna = sorted([n for n, d in self.G.nodes(data=True) if d['type'] == self._srna and d['strain'] == strain])
        assert raw_srna == graph_srna, f"sRNA nodes in the graph do not match the raw data for strain {strain}"
    
    def _assert_srna_mrna_inter_addition(self, strain):
        raw_intr = sorted(zip(self.strains_data[strain]['unq_inter'][self.srna_acc_col], self.strains_data[strain]['unq_inter'][self.mrna_acc_col]))
        graph_intr = sorted([(u, v) for u, v, d in self.G.edges(data=True) if d['type'] == self._targets and 
                             self.G.nodes[u]['strain'] == strain
                             and self.G.nodes[v]['strain'] == strain])
        assert raw_intr == graph_intr, f"sRNA-mRNA interactions in the graph do not match the raw data for strain {strain}"
    
    def _add_node_rna(self, id, type, strain, locus_tag, name, synonyms, start, end, strand, sequence):
        if not self.G.has_node(id):
            self.G.add_node(id, type=type, 
							strain=strain, locus_tag=locus_tag, name=name, synonyms=synonyms, start=start, end=end, strand=strand, sequence=sequence)
        else:
            self.logger.warning(f"{type} node {id} already in graph")
    
    def _add_edge_mrna_go_annot(self, mrna_node_id, go_id) -> bool:
        """ Add "annotated" edge from the mRNA node to the GO node.
        Args:
            mrna_node_id (str): the mRNA node id (accession id)
            go_id (str): the GO id

        Returns:
            bool: whtether the go_id is missing in the ontology
        """
        assert self.G.has_node(mrna_node_id), f"mRNA id {mrna_node_id} is missing in the graph"
        if self.G.has_node(go_id):
            self.G.add_edge(mrna_node_id, go_id, type=self._annotated)
            return False
        return True
    
    def _add_edge_srna_mrna_inter(self, srna_node_id, mrna_node_id):
        """ Add "targets" edge from the sRNA node to the mRNA node

        Args:
            srna_node_id (str): the sRNA node id (accession id)
            mrna_node_id (str): the mRNA node id (accession id)
        """
        if self.G.has_node(srna_node_id) and self.G.has_node(mrna_node_id):
            self.G.add_edge(srna_node_id, mrna_node_id, type=self._targets)
        else:
            self.logger.warning("mRNA and/or sRNA nodes are missing in the graph")

    def _split(self, go_terms: set):
        BP_go_ids, MF_go_ids, CC_go_ids = [], [], []
        
        if pd.notnull(go_terms):   
            for t in go_terms:
                go_id = t.split(':')[1]
                if go_id in self.ontology.BP.nodes:
                    BP_go_ids.append(go_id)
                elif go_id in self.ontology.MF.nodes:
                    MF_go_ids.append(go_id)
                elif go_id in self.ontology.CC.nodes:
                    CC_go_ids.append(go_id)
                else:
                    self.curated_go_ids_missing_in_ontology.add(go_id)
            
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

