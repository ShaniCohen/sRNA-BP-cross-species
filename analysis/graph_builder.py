from typing import List, Dict
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

class GraphBuilder:
    def __init__(self, config, logger, data_loader, ontology):
        self.logger = logger
        self.logger.info(f"initializing GraphBuilder")
        self.config = config

        self.strains_data = data_loader.strains_data
        self.srna_acc_col = data_loader.srna_acc_col
        self.mrna_acc_col = data_loader.mrna_acc_col
        self.clustering_data = data_loader.clustering_data
        
        self.ontology = ontology
        self.curated_go_ids_missing_in_ontology = set()
        
        self.G = nx.MultiDiGraph()
        # add BP nodes and edges
        self.G.add_nodes_from(ontology.BP.nodes(data=True))
        self.G.add_edges_from(ontology.BP.edges(data=True))
        """
        # add MF nodes and edges
        self.G.add_nodes_from(ontology.MF.nodes(data=True))
        self.G.add_edges_from(ontology.MF.edges(data=True))
        
        # add CC nodes and edges
        self.G.add_nodes_from(ontology.CC.nodes(data=True))
        self.G.add_edges_from(ontology.CC.edges(data=True))
        """
        self.graph_is_built = False

        # node types
        # GO
        self._bp = ontology.type_bp
        self._mf = ontology.type_mf
        self._cc = ontology.type_cc
        self._go_types = [self._bp, self._mf, self._cc]
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
        # RNA <--> RNA
        self._paralog = "paralog"    # paralogs: same strain
        self._ortholog = "ortholog"  # orthologs: different strains

        # edge annot types (annot_type)
        self._curated = "curated"
        self._ips = "interproscan"
        self._eggnog = "eggnog"
        self._annot_types = [self._curated, self._ips, self._eggnog]

        # define annotation types to add (curated are always added)
        self.add_ips_annot = True
        self.add_eggnog_annot = True
    
    def get_strains(self) -> List[str]:
        return list(self.strains_data.keys())
    
    def get_graph(self) -> nx.Graph:
        if not self.graph_is_built:
            raise Exception("Graph is not built yet. Please call build_graph() first.")
        return self.G
    
    def get_ips_go_annotations(self) -> Dict[str, pd.DataFrame]:
        out = {}
        for strain, data in self.strains_data.items():
            if 'all_mrna_w_ips_annot' in data.keys():
                out[strain] = data['all_mrna_w_ips_annot']
        return out

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
        self._add_homology_edges()
        self._log_graph_info()
        # self._log_graph_info(dump=True)
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
            else:
                if self.add_ips_annot and 'all_mrna_w_ips_annot' in data.keys():
                    self._add_all_mrna_and_ips_bp_annot(strain, data['all_mrna_w_ips_annot'])
                if self.add_eggnog_annot and 'all_mrna_w_eggnog_annot' in data.keys():
                    self._add_all_mrna_and_eggnog_annot(strain, data['all_mrna_w_eggnog_annot'])
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
    
    def _add_homology_edges(self):
        self._add_rna_homology_edges(rna_type=self._srna)
        self._add_rna_homology_edges(rna_type=self._mrna)

    def _add_rna_homology_edges(self, rna_type: str):
        """Add homology edges between RNA nodes based on clustering data (bacteria pairs)."""
        self.logger.info(f"adding {rna_type} homology edges")
        for (b1, b2), clstr_data in self.clustering_data[rna_type].items():
            # self.logger.info(f"adding {rna_type} homology edges for {b1} - {b2}")
            assert set(clstr_data['strain_str']) <= set(self.strains_data.keys()), f"strain strings mismatch"
            for cluster_id, group in clstr_data.groupby('cluster_id'):
                if len(group) > 1:
                    nodes = list(zip(group['rna_accession_id'], group['strain_str']))
                    for n1, n2 in itertools.combinations(nodes, 2):
                        node_id_1, node_id_2 = n1[0], n2[0]
                        strain_1, strain_2 = n1[1], n2[1]
                        if strain_1 == strain_2:
                            # paralogs: same strain
                            if not self._are_paralogs(node_id_1, node_id_2, strain_1):
                                self._add_edges_rna_rna_paralogs(node_id_1, node_id_2)
                        else:
                            # orthologs: different strains
                            if not self._are_orthologs(node_id_1, node_id_2, strain_1, strain_2):
                                self._add_edges_rna_rna_orthologs(node_id_1, node_id_2)

    def _log_graph_info(self, dump=False):
        self.logger.info("Logging graph information...")
        strains = self.get_strains()
        
        for strain in strains:
            # --------------------------------
            #              mRNA
            # --------------------------------
            # Filter mRNA nodes for the current strain
            mrna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self._mrna and d['strain'] == strain]
            mrna_count = len(mrna_nodes)

            # Count mRNA nodes with interactions (sRNA --targets--> mRNA)
            # mRNA = 'EG10001'
            # sorted([(u, v) for u, v, d in self.G.edges(data=True) if u == 'EG10001' or v == 'EG10001'])
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter'][self.mrna_acc_col] == 'EG10001']
            
            mrna_with_interactions = [
                n for n in mrna_nodes if any(self._is_target(p, n, strain) for p in self.G.predecessors(n) if self.G.nodes[p]['type'] == self._srna)
            ]
            mrna_with_interactions_count = len(mrna_with_interactions)

            # mRNA nodes with interactions (sRNA --targets--> mRNA) and BP GO annotations (mRNA --annotation--> BP)
            mrna_bp_annotations = {
                n: [neighbor for neighbor in self.G.neighbors(n) if self.G.nodes[neighbor]['type'] == self._bp and self._is_annotated(n, neighbor, self._bp)]
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
            
            annotation_types = [self._curated] + ([self._ips] if self.add_ips_annot else []) + ([self._eggnog] if self.add_eggnog_annot else [])
            # --------------------------------
            #              sRNA
            # --------------------------------
            # sRNA = 'G0-16636'
            # sorted([(u, v) for u, v, d in self.G.edges(data=True) if u == 'G0-16636' or v == 'G0-16636'])
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter'][self.srna_acc_col] == 'G0-16636']
            # Filter sRNA nodes for the current strain
            srna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self._srna and d['strain'] == strain]
            srna_count = len(srna_nodes)

            # Count sRNA nodes with interactions (sRNA-mRNA)
            srna_with_interactions = [
                n for n in srna_nodes if any(self._is_target(n, neighbor, strain) for neighbor in self.G.neighbors(n) if self.G.nodes[neighbor]['type'] == self._mrna)
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
                f"\n   mRNA with interactions and BP annotations {annotation_types}: {mrna_w_bp_annot_count} "
                f"\n   mRNA with interactions and BP annotations and at least 1 embedding: {mrna_w_bp_annot_and_at_least_1_emb_count} "
                f"\n   sRNA count: {srna_count} "
                f"\n   sRNA with interactions: {srna_with_interactions_count} "
                f"\n   sRNA with interactions and BP annotations: {srna_w_bp_annot_count} "
            )
            if dump:
                self._dump_mrna_bp_annotations(mrna_with_interactions, strain)
    
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
    
    def _dump_mrna_bp_annotations(self, mrna_with_interactions, strain):
        """ Dump mRNA with interactions and their BP annotations to a CSV file """
        self.logger.info(f"Dumping mRNA with interactions and their BP annotations for {strain}")
        data = []
        for mrna in mrna_with_interactions:
            bp_nodes = sorted([n for n in self.G.neighbors(mrna) if self._is_annotated(mrna, n, self._bp)])
            bp_nodes_ips = sorted([n for n in self.G.neighbors(mrna) if self._is_annotated(mrna, n, self._bp, self._ips)])
            bp_nodes_eggnog = sorted([n for n in self.G.neighbors(mrna) if self._is_annotated(mrna, n, self._bp, self._eggnog)])

            BP_ips_only = set(bp_nodes_ips) - set(bp_nodes_eggnog)
            BP_eggnog_only = set(bp_nodes_eggnog) - set(bp_nodes_ips)
            BP_ips_and_ennog = set(bp_nodes) - set(BP_ips_only) - set(BP_eggnog_only)
            
            # --- new: get BP_eggnog_only_name ---
            BP_eggnog_only_name = ', '.join([self.G.nodes[bp]['lbl'] for bp in BP_eggnog_only]) if len(BP_eggnog_only) > 0 else np.nan

            if len(bp_nodes) > 0:
                data.append({
                    'mRNA_accession_id': mrna,
                    'mrna_name': self.G.nodes[mrna]['name'],
                    "BP_count": len(set(bp_nodes)),
                    "BP_ips_and_eggnog_count": len(BP_ips_and_ennog),
                    'BP_ips_only_count': len(BP_ips_only),
                    'BP_eggnog_only_count': len(BP_eggnog_only),
                    "BP_ips_and_eggnog_ratio": round(len(BP_ips_and_ennog)/len(set(bp_nodes)), 2),
                    'BP_ips_only_ratio': round(len(BP_ips_only)/len(set(bp_nodes)), 2),
                    'BP_eggnog_only_ratio': round(len(BP_eggnog_only)/len(set(bp_nodes)), 2),
                    'BP_ips_and_eggnog': ', '.join(BP_ips_and_ennog) if len(BP_ips_and_ennog) > 0 else np.nan,
                    'BP_ips_only': ', '.join(BP_ips_only) if len(BP_ips_only) > 0 else np.nan,
                    'BP_eggnog_only': ', '.join(BP_eggnog_only) if len(BP_eggnog_only) > 0 else np.nan,
                    'BP_eggnog_only_name': BP_eggnog_only_name
                })
        df = pd.DataFrame(data)
        output_path = os.path.join(self.config['builder_output_dir'], f"{strain}_mrna_bp_annotations.csv")
        write_df(df, output_path)
    
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
                    bp_id_is_missing = self._add_edge_mrna_go_annot(mrna_node_id, bp_id, annot_type=self._curated)
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
                    bp_id_is_missing = self._add_edge_mrna_go_annot(mrna_node_id, bp_id, annot_type=self._ips)
                    if bp_id_is_missing:
                        missing_bp_dicts.append(bp_dict)
                    bp_count += 1
        # log
        unq_missing = set([x['id'] for x in missing_bp_dicts])
        dep = [m for m in unq_missing if m in self.ontology.deprecated_nodes]
        self.logger.info(f"{strain}: out of {bp_count} InterProScan BP annotations, missing: {len(missing_bp_dicts)} ({len(unq_missing)} unique), deprecated: {len(dep)}")
    
    def _add_all_mrna_and_eggnog_annot(self, strain, all_mrna_w_eggnog_annot):
        self.logger.info(f"adding mRNA nodes and EggNog mRNA-GO annotations for {strain}")
        assert sum(pd.isnull(all_mrna_w_eggnog_annot['mRNA_accession_id'])) == 0
        go_count, missing_go_ids = 0, []

        for _, r in all_mrna_w_eggnog_annot.iterrows():
            # 1 - add the mRNA node to graph
            mrna_node_id = r[self.mrna_acc_col]
            self._add_node_rna(id=mrna_node_id, type=self._mrna, strain=strain, locus_tag=r['mRNA_locus_tag'], 
                               name=r['mRNA_name'], synonyms=r['mRNA_name_synonyms'], start=r['mRNA_start'], end=r['mRNA_end'],
                               strand=r['mRNA_strand'], sequence=r['mRNA_sequence'], log_warning=False)
            # 2 - add annotation edges between the mRNA node and GO nodes (BO, MF, CC if exist in the ontology)
            if pd.notnull(r['GOs']) and r['GOs'] != '-':
                go_ids = [g.replace("GO:", "") for g in r['GOs'].split(',')]
                for go_id in go_ids:
                    go_id_is_missing = self._add_edge_mrna_go_annot(mrna_node_id, go_id, annot_type=self._eggnog)
                    if go_id_is_missing:
                        missing_go_ids.append(go_id)
                    go_count += 1
        # log
        unq_missing = set(missing_go_ids)
        dep = [m for m in unq_missing if m in self.ontology.deprecated_nodes]
        self.logger.info(f"{strain}: out of {go_count} EggNog GO annotations, missing: {len(missing_go_ids)} ({len(unq_missing)} unique), deprecated: {len(dep)}")

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
    
    def _add_node_rna(self, id, type, strain, locus_tag, name, synonyms, start, end, strand, sequence, log_warning=True):
        if not self.G.has_node(id):
            self.G.add_node(id, type=type, 
							strain=strain, locus_tag=locus_tag, name=name, synonyms=synonyms, start=start, end=end, strand=strand, sequence=sequence)
        elif log_warning:
            self.logger.warning(f"{type} node {id} already in graph")

    def _add_edge_srna_mrna_inter(self, srna_node_id, mrna_node_id):
        """ Add "targets" edge from the sRNA node to the mRNA node

        Args:
            srna_node_id (str): the sRNA node id (accession id)
            mrna_node_id (str): the mRNA node id (accession id)
        """
        assert self.G.has_node(srna_node_id) and self.G.has_node(mrna_node_id)
        assert self.G.nodes[srna_node_id]['strain'] == self.G.nodes[mrna_node_id]['strain']
        assert self.G.nodes[srna_node_id]['type'] == self._srna
        assert self.G.nodes[mrna_node_id]['type'] == self._mrna
        self.G.add_edge(srna_node_id, mrna_node_id, type=self._targets)
        
    def _add_edge_mrna_go_annot(self, mrna_node_id, go_id, annot_type) -> bool:
        """ Add "annotated" edge from the mRNA node to the GO node.
        Args:
            mrna_node_id (str): the mRNA node id (accession id)
            go_id (str): the GO id

        Returns:
            bool: whtether the go_id is missing in the ontology
        """
        assert self.G.has_node(mrna_node_id), f"mRNA id {mrna_node_id} is missing in the graph"
        if self.G.has_node(go_id):
            assert self.G.nodes[mrna_node_id]['type'] == self._mrna
            assert self.G.nodes[go_id]['type'] in self._go_types
            assert annot_type in self._annot_types
            self.G.add_edge(mrna_node_id, go_id, type=self._annotated, annot_type=annot_type)
            return False
        return True
    
    def _add_edges_rna_rna_paralogs(self, rna_node_id_1, rna_node_id_2):
        """ Add "targets" edge from the sRNA node to the mRNA node

        Args:
            srna_node_id (str): the sRNA node id (accession id)
            mrna_node_id (str): the mRNA node id (accession id)
        """
        assert self.G.has_node(rna_node_id_1) and self.G.has_node(rna_node_id_2)
        assert self.G.nodes[rna_node_id_1]['strain'] == self.G.nodes[rna_node_id_2]['strain']
        both_srna = (self.G.nodes[rna_node_id_1]['type'] == self._srna) & (self.G.nodes[rna_node_id_2]['type'] == self._srna)
        both_mrna = (self.G.nodes[rna_node_id_2]['type'] == self._mrna) & (self.G.nodes[rna_node_id_2]['type'] == self._mrna)
        assert both_srna or both_mrna
        self.G.add_edge(rna_node_id_1, rna_node_id_2, type=self._paralog)
        self.G.add_edge(rna_node_id_2, rna_node_id_1, type=self._paralog)

    def _add_edges_rna_rna_orthologs(self, rna_node_id_1, rna_node_id_2):
        """ Add "ortholog" edge between two RNA nodes of different strains

        Args:
            rna_node_id_1 (str): the first RNA node id (accession id)
            rna_node_id_2 (str): the second RNA node id (accession id)
        """
        assert self.G.has_node(rna_node_id_1) and self.G.has_node(rna_node_id_2)
        assert self.G.nodes[rna_node_id_1]['strain'] != self.G.nodes[rna_node_id_2]['strain']
        both_srna = (self.G.nodes[rna_node_id_1]['type'] == self._srna) & (self.G.nodes[rna_node_id_2]['type'] == self._srna)
        both_mrna = (self.G.nodes[rna_node_id_2]['type'] == self._mrna) & (self.G.nodes[rna_node_id_2]['type'] == self._mrna)
        assert both_srna or both_mrna
        self.G.add_edge(rna_node_id_1, rna_node_id_2, type=self._ortholog)
        self.G.add_edge(rna_node_id_2, rna_node_id_1, type=self._ortholog)
    
    def _is_target(self, srna_node_id, mrna_node_id, strain):
        """ Check if there is an interaction edge between sRNA and mRNA nodes """
        if not self.G.nodes[srna_node_id]['strain'] == self.G.nodes[mrna_node_id]['strain'] == strain:
            print()
        assert self.G.nodes[srna_node_id]['strain'] == self.G.nodes[mrna_node_id]['strain'] == strain
        assert self.G.nodes[srna_node_id]['type'] == self._srna
        assert self.G.nodes[mrna_node_id]['type'] == self._mrna
        
        is_interaction = False
        for d in self.G[srna_node_id][mrna_node_id].values():
            if d['type'] == self._targets:
                is_interaction = True
                break 
        return is_interaction
    
    def _is_annotated(self, mrna_node_id, go_node_id, go_node_type, annot_type=None):
        """ Check if there is an annotation edge from mRNA to GO node."""
        assert self.G.nodes[mrna_node_id]['type'] == self._mrna
        assert self.G.nodes[go_node_id]['type'] == go_node_type
        
        is_annotated = False
        for d in self.G[mrna_node_id][go_node_id].values():
            if d['type'] == self._annotated and (annot_type is None or d['annot_type'] == annot_type):
                is_annotated = True
                break
        return is_annotated
    
    def _are_paralogs(self, rna_node_id_1, rna_node_id_2, strain):
        """ Check if there are paralog edges between two RNA nodes of the same strain """
        assert self.G.nodes[rna_node_id_1]['strain'] == self.G.nodes[rna_node_id_2]['strain'] == strain
        both_srna = (self.G.nodes[rna_node_id_1]['type'] == self._srna) & (self.G.nodes[rna_node_id_2]['type'] == self._srna)
        both_mrna = (self.G.nodes[rna_node_id_2]['type'] == self._mrna) & (self.G.nodes[rna_node_id_2]['type'] == self._mrna)
        assert both_srna or both_mrna
        
        is_paralog_1_2 = False
        if self.G.has_edge(rna_node_id_1, rna_node_id_2):
            for d in self.G[rna_node_id_1][rna_node_id_2].values():
                if d['type'] == self._paralog:
                    is_paralog_1_2 = True
                    break
        is_paralog_2_1 = False
        if self.G.has_edge(rna_node_id_2, rna_node_id_1):
            for d in self.G[rna_node_id_2][rna_node_id_1].values():
                if d['type'] == self._paralog:
                    is_paralog_2_1 = True
                    break
        assert is_paralog_1_2 == is_paralog_2_1, "Paralog edge should be symmetric"
        return is_paralog_1_2 and is_paralog_2_1
    
    def _are_orthologs(self, rna_node_id_1, rna_node_id_2, strain_1, strain_2):
        """ Check if there are ortholog edges between two RNA nodes of different strains """
        assert self.G.nodes[rna_node_id_1]['strain'] == strain_1
        assert self.G.nodes[rna_node_id_2]['strain'] == strain_2
        assert strain_1 != strain_2, "Strains should be different for orthologs"
        both_srna = (self.G.nodes[rna_node_id_1]['type'] == self._srna) & (self.G.nodes[rna_node_id_2]['type'] == self._srna)
        both_mrna = (self.G.nodes[rna_node_id_2]['type'] == self._mrna) & (self.G.nodes[rna_node_id_2]['type'] == self._mrna)
        assert both_srna or both_mrna
        
        is_ortholog_1_2 = False
        if self.G.has_edge(rna_node_id_1, rna_node_id_2):
            for d in self.G[rna_node_id_1][rna_node_id_2].values():
                if d['type'] == self._ortholog:
                    is_ortholog_1_2 = True
                    break
        is_ortholog_2_1 = False
        if self.G.has_edge(rna_node_id_2, rna_node_id_1):
            for d in self.G[rna_node_id_2][rna_node_id_1].values():
                if d['type'] == self._ortholog:
                    is_ortholog_2_1 = True
                    break
        assert is_ortholog_1_2 == is_ortholog_2_1, "Ortholog edge should be symmetric"
        return is_ortholog_1_2 and is_ortholog_2_1
        
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

