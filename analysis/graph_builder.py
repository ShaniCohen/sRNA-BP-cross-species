from typing import List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from utils.general import write_df, create_dir_if_not_exists
import networkx as nx
from pyvis.network import Network
from os.path import join
import json
import sys
import os

ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

class GraphBuilder:
    def __init__(self, config, logger, data_loader, ontology, graph_utils):
        self.logger = logger
        self.logger.info(f"initializing GraphBuilder")
        self.config = config

        self.ecoli_k12_nm = data_loader.ecoli_k12_nm
        # self.vibrio_nm = data_loader.vibrio_nm
        # self.pseudomonas_nm = data_loader.pseudomonas_nm

        self.strains_data = data_loader.strains_data
        self.srna_acc_col = data_loader.srna_acc_col
        self.mrna_acc_col = data_loader.mrna_acc_col
        self.clustering_data = data_loader.clustering_data
        
        self.ontology = ontology
        self.curated_go_ids_missing_in_ontology = set()
        self.U = graph_utils
        
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

        # ---------  RUNTIME FLAGS  ---------  
        # #TODO: adjust that (merge with version/ add to conf)
        self.add_ips_annot = True
        self.add_eggnog_annot = False
        
        # ---------  CONFIGURATIONS  ---------
        self.version = self.config['version']  # "k12_curated_ips", "k12_curated", "k12_ips"
        
        conf_str = f"{self.version}"

        # output paths
        parent_dir = join(self.config['builder_output_dir'], conf_str)
        self.out_path_mrna_bp_annot = create_dir_if_not_exists(join(parent_dir, "mRNA_BP_annot"))
        self.out_path_homology_pairs_stats = create_dir_if_not_exists(join(parent_dir, "Homology_pairs_statistics"))

        # file names suffix
        self.out_file_suffix = f"v_{conf_str}"
    
    def get_version(self) -> str:
        return self.config.get('version')
    
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
        # self._log_graph_info()
        self._log_graph_info(dump=True)
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
                self.logger.info(f"{strain}: out of {has_go} mRNAs with curated annotations, has BP = {has_bp} ({(has_bp/has_go)*100:.2f}%), has MF = {has_mf} ({(has_mf/has_go)*100:.2f}%), has CC = {has_cc} ({(has_cc/has_go)*100:.2f}%)")

    def _add_mrna_nodes_and_annotation_edges(self):
        for strain, data in self.strains_data.items():  # describe proprocessing in the latex paper
            # E.coli K12
            if strain == self.ecoli_k12_nm:
                if self.version in ["k12_curated", "k12_curated_ips"]:
                    self._add_all_mrna_and_curated_bp_annot(strain, data['all_mrna_w_curated_annot'])
                    # Example mRNA = 'EG10001', GO BP = ['0006522', '0030632', '0071555', '0009252', '0008360']
                if self.version in ["k12_ips", "k12_curated_ips"]:
                    self._add_all_mrna_and_ips_bp_annot(strain, data['all_mrna_w_ips_annot'])
            # Other strains
            else:
                if self.add_ips_annot and 'all_mrna_w_ips_annot' in data.keys():
                    self._add_all_mrna_and_ips_bp_annot(strain, data['all_mrna_w_ips_annot'])
                if self.add_eggnog_annot and 'all_mrna_w_eggnog_annot' in data.keys():
                    self._add_all_mrna_and_eggnog_annot(strain, data['all_mrna_w_eggnog_annot'])
            
            self._assert_mrna_nodes_addition(strain)
            
    def _add_srna_nodes_and_interaction_edges(self):
        for strain, data in self.strains_data.items():
            self.logger.info(f"adding sRNA nodes and sRNA-mRNA interactions for {strain}")
            # 1 - add all sRNA nodes to graph
            for _, r in data['all_srna'].iterrows():
                srna_node_id = r[self.srna_acc_col]
                self.G = self.U.add_node_rna(self.G, id=srna_node_id, type=self.U.srna, strain=strain, locus_tag=r['sRNA_locus_tag'], 
                                             name=r['sRNA_name'], synonyms=r['sRNA_name_synonyms'], start=r['sRNA_start'], end=r['sRNA_end'],
                                             strand=r['sRNA_strand'], rna_seq=r['sRNA_sequence'])
            
            # TODO: Decide how to use interactions data (growth cond, hfq, only pos inter, count?)
            # 2 - add sRNA-mRNA interaction edges
            for _, r in data['unq_inter'].iterrows():
                # 2.1 - add the interaction edges between sRNA and mRNA nodes
                srna_node_id = r[self.srna_acc_col]
                mrna_node_id = r[self.mrna_acc_col]
                self.G = self.U.add_edge_srna_mrna_inter(self.G, srna_node_id, mrna_node_id)
            
            self._assert_srna_nodes_addition(strain)
            self._assert_srna_mrna_inter_addition(strain)
    
    def _add_homology_edges(self):
        # 1 - clustering-based homology edges (for all strains)
        self._add_rna_homology_edges_clustering_based(rna_type=self.U.srna) 
        self._add_rna_homology_edges_clustering_based(rna_type=self.U.mrna)
        # 2 - named-based homology edges (for all strains)
        self._add_rna_homology_edges_name_based(rna_type=self.U.srna)
        self._add_rna_homology_edges_name_based(rna_type=self.U.mrna)
        # 3 - calc & dump statistics
        self._dump_homology_edges_stats(rna_type=self.U.srna)
        self._dump_homology_edges_stats(rna_type=self.U.mrna)

    def _add_rna_homology_edges_clustering_based(self, rna_type: str):
        """Add homology edges between RNA nodes based on clustering data (bacteria pairs)."""
        self.logger.info(f"adding {rna_type} homology edges - clustering based")
        for (b1, b2), clstr_data in self.clustering_data[rna_type].items():
            # self.logger.info(f"adding {rna_type} homology edges for {b1} - {b2}")
            assert set(clstr_data['strain_str']) <= set(self.U.strains), f"invalid strain strings"
            for cluster_id, group in clstr_data.groupby('cluster_id'):
                if len(group) > 1:
                    nodes = list(zip(group['rna_accession_id'], group['strain_str']))
                    for n1, n2 in itertools.combinations(nodes, 2):
                        node_id_1, node_id_2 = n1[0], n2[0]
                        strain_1, strain_2 = n1[1], n2[1]
                        if strain_1 == strain_2:
                            # paralogs: same strain
                            if not self.U.are_paralogs(self.G, node_id_1, node_id_2, strain_1):
                                self.G = self.U.add_edges_rna_rna_paralogs(self.G, node_id_1, node_id_2)
                        else:
                            # orthologs: different strains
                            if not self.U.are_orthologs_by_seq(self.G, node_id_1, node_id_2, strain_1, strain_2):
                                self.G = self.U.add_edges_rna_rna_orthologs_by_seq(self.G, node_id_1, node_id_2)
    
    def _dump_homology_edges_stats(self, rna_type: str):
        """Calc and dump statistics of homology edge types between RNA nodes of different strains"""
        self.logger.info(f"Calc and dump statistics - {rna_type} homology edges")
        records = []
        for curr_strain in self.U.strains:
            curr_node_ids = [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == curr_strain]
            rec = {"Strain": f"{curr_strain} \n {rna_type} nodes = {len(curr_node_ids)}"}
            # compare with all other strains
            for other_strain in self.U.strains:
                if other_strain != curr_strain:
                    other_node_ids = [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == other_strain]
                    _col_nm = f"{other_strain} \n {rna_type} nodes = {len(other_node_ids)}"
                    # calculate the number of ortholog edges (seq and name) between RNAs of curr & other strain
                    # TODO: ----------------
                    num_ortholog_by_seq = self._get_num_of_ortholog_edges(curr_node_ids, other_node_ids, [self.U.ortholog_by_seq])
                    num_ortholog_by_name = self._get_num_of_ortholog_edges(curr_node_ids, other_node_ids, [self.U.ortholog_by_name])
                    num_ortholog_by_seq_or_name = self._get_num_of_ortholog_edges(curr_node_ids, other_node_ids, [self.U.ortholog_by_seq, self.U.ortholog_by_name])

                    _col_val = 0
                    # TODO: ----------------
                    rec = {_col_nm: _col_val}
        df = pd.DataFrame(records)
        write_df(df, join(self.out_path_homology_pairs_stats, f"{rna_type}_homology_edges__{self.out_file_suffix}.csv"))
    
    def _get_num_of_ortholog_edges(node_ids_1: List[str], node_ids_2: List[str], ortholog_edge_types: List[str]):
        return

    def _add_rna_homology_edges_name_based(self, rna_type: str):
        """Add homology edges between their RNA nodes and RNA nodes of all other strain 
           based on RNA names (bacteria pairs)."""
        self.logger.info(f"adding {rna_type} homology edges - name based")
        nm_col = 'sRNA_name' if rna_type==self.U.srna else 'mRNA_name'
        id_col = self.srna_acc_col if rna_type==self.U.srna else self.mrna_acc_col

        for curr_strain in self.U.strains:
            all_rna_curr = self.strains_data[curr_strain][f'all_{rna_type}'].copy()[[id_col, nm_col]]
            all_rna_curr[nm_col] = all_rna_curr[nm_col].apply(lambda x: x.lower())
            all_rna_curr = all_rna_curr.rename(columns={col: f"{col}_curr" for col in all_rna_curr.columns})
            # compare with all other strains
            for other_strain, other_data in self.strains_data.items():
                if other_strain != curr_strain:
                    all_rna_other = other_data[f'all_{rna_type}'].copy()[[id_col, nm_col]]
                    all_rna_other[nm_col] = all_rna_other[nm_col].apply(lambda x: x.lower())
                    all_rna_other = all_rna_other.rename(columns={col: f"{col}_other" for col in all_rna_other.columns})
                    matches = pd.merge(all_rna_curr, all_rna_other, left_on=f"{nm_col}_curr", right_on=f"{nm_col}_other", how='inner')
                    for _, match in matches.iterrows():
                        curr_node_id = match[f"{id_col}_curr"]
                        other_node_id = match[f"{id_col}_other"]
                        if not self.U.are_orthologs_by_name(self.G, curr_node_id, other_node_id, curr_strain, other_strain):
                            # TODO: cosider to add a 'name_based' attribute to the edge data
                            self.G = self.U.add_edges_rna_rna_orthologs_by_name(self.G, curr_node_id, other_node_id)

    def _log_graph_info(self, dump=False):
        self.logger.info("Logging graph information...")
        for strain in self.U.strains:
            # --------------------------------
            #              mRNA
            # --------------------------------
            # Filter mRNA nodes for the current strain
            mrna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.mrna and d['strain'] == strain]
            mrna_count = len(mrna_nodes)

            # Count mRNA nodes with interactions (sRNA --targets--> mRNA)
            # mRNA = 'EG10001'
            # sorted([(u, v) for u, v, d in self.G.edges(data=True) if u == 'EG10001' or v == 'EG10001'])
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter'][self.mrna_acc_col] == 'EG10001']
            
            mrna_with_interactions = [
                mrna for mrna in mrna_nodes if any(self.U.is_target(self.G, p, mrna) for p in self.G.predecessors(mrna) if self.G.nodes[p]['type'] == self.U.srna)
            ]
            mrna_with_interactions_count = len(mrna_with_interactions)

            # mRNA nodes with interactions (sRNA --targets--> mRNA) and BP GO annotations (mRNA --annotation--> BP)
            mrna_bp_annotations = {
                mrna: [n for n in self.G.neighbors(mrna) if self.U.is_annotated(self.G, mrna, n, self.U.bp)]
                for mrna in mrna_with_interactions
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
            
            annotation_types = [self.U.curated] + ([self.U.ips] if self.add_ips_annot else []) + ([self.U.eggnog] if self.add_eggnog_annot else [])
            # --------------------------------
            #              sRNA
            # --------------------------------
            # sRNA = 'G0-16636'
            # sorted([(u, v) for u, v, d in self.G.edges(data=True) if u == 'G0-16636' or v == 'G0-16636'])
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter'][self.srna_acc_col] == 'G0-16636']
            # Filter sRNA nodes for the current strain
            srna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.srna and d['strain'] == strain]
            srna_count = len(srna_nodes)

            # Count sRNA nodes with interactions (sRNA-mRNA)
            srna_with_interactions = [
                srna for srna in srna_nodes if any(self.U.is_target(self.G, srna, neighbor) for neighbor in self.G.neighbors(srna) if self.G.nodes[neighbor]['type'] == self.U.mrna)
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
            bp_nodes = sorted([n for n in self.G.neighbors(mrna) if self.U.is_annotated(self.G, mrna, n, self.U.bp)])
            bp_nodes_ips = sorted([n for n in self.G.neighbors(mrna) if self.U.is_annotated(self.G, mrna, n, self.U.bp, self.U.ips)])
            bp_nodes_eggnog = sorted([n for n in self.G.neighbors(mrna) if self.U.is_annotated(self.G, mrna, n, self.U.bp, self.U.eggnog)])

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
        write_df(df, join(self.out_path_mrna_bp_annot, f"{strain}_mrna_bp_annot_{self.out_file_suffix}.csv"))
    
    def _add_all_mrna_and_curated_bp_annot(self, strain, all_mrna_w_curated_annot):
        self.logger.info(f"adding mRNA nodes and curated mRNA-GO annotations for {strain}")
        assert sum(pd.isnull(all_mrna_w_curated_annot['mRNA_accession_id'])) == 0
        bp_count, missing_bp_ids = 0, []

        for _, r in all_mrna_w_curated_annot.iterrows():
            # 1 - add the mRNA node to graph
            mrna_node_id = r[self.mrna_acc_col]
            self.G = self.U.add_node_rna(self.G, id=mrna_node_id, type=self.U.mrna, strain=strain, locus_tag=r['mRNA_locus_tag'], 
                               name=r['mRNA_name'], synonyms=r['mRNA_name_synonyms'], start=r['mRNA_start'], end=r['mRNA_end'],
                               strand=r['mRNA_strand'], rna_seq=r['mRNA_sequence'], protein_seq=r['protein_seq'])
            # 2 - add annotation edge between the mRNA node and its BP nodes
            go_bp_ids = r['GO_BP']
            if isinstance(go_bp_ids, list) and len(go_bp_ids) > 0:
                for bp_id in go_bp_ids:
                    self.G, bp_id_is_missing = self.U.add_edge_mrna_go_annot(self.G, mrna_node_id, bp_id, annot_type=self.U.curated)
                    if bp_id_is_missing:
                        missing_bp_ids.append(bp_id)
                    bp_count += 1
        dep = [m for m in set(missing_bp_ids) if m in self.ontology.deprecated_nodes]
        self.logger.info(f"{strain}: out of {bp_count} curated BP annotations, missing: {len(missing_bp_ids)} ({len(set(missing_bp_ids))} unique), deprecated: {len(dep)}")
                    
    def _add_all_mrna_and_ips_bp_annot(self, strain, all_mrna_w_ips_annot):
        self.logger.info(f"adding mRNA nodes and InterProScan mRNA-GO annotations for {strain}")
        assert sum(pd.isnull(all_mrna_w_ips_annot['mRNA_accession_id'])) == 0
        log_warning = False if strain == self.ecoli_k12_nm and self.version == 'k12_curated_ips' else True
        bp_count, missing_bp_dicts = 0, []

        for _, r in all_mrna_w_ips_annot.iterrows():
            # 1 - add the mRNA node to graph
            mrna_node_id = r[self.mrna_acc_col]
            self.G = self.U.add_node_rna(self.G, id=mrna_node_id, type=self.U.mrna, strain=strain, locus_tag=r['mRNA_locus_tag'], 
                               name=r['mRNA_name'], synonyms=r['mRNA_name_synonyms'], start=r['mRNA_start'], end=r['mRNA_end'],
                               strand=r['mRNA_strand'], rna_seq=r['mRNA_sequence'], protein_seq=r['protein_seq'], log_warning=log_warning)
            # 2 - add annotation edges between the mRNA node and BP nodes
            bp_go_xrefs = r['BP_go_xrefs']
            if isinstance(bp_go_xrefs, list) and len(bp_go_xrefs) > 0:
                for bp_dict in bp_go_xrefs:
                    bp_id = bp_dict['id'].split(":")[1]
                    self.G, bp_id_is_missing = self.U.add_edge_mrna_go_annot(self.G, mrna_node_id, bp_id, annot_type=self.U.ips)
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
            self.G = self.U.add_node_rna(self.G, id=mrna_node_id, type=self.U.mrna, strain=strain, locus_tag=r['mRNA_locus_tag'], 
                                         name=r['mRNA_name'], synonyms=r['mRNA_name_synonyms'], start=r['mRNA_start'], end=r['mRNA_end'],
                                         strand=r['mRNA_strand'], rna_seq=r['mRNA_sequence'], protein_seq=r['protein_seq'], log_warning=False)
            # 2 - add annotation edges between the mRNA node and GO nodes (BO, MF, CC if exist in the ontology)
            if pd.notnull(r['GOs']) and r['GOs'] != '-':
                go_ids = [g.replace("GO:", "") for g in r['GOs'].split(',')]
                for go_id in go_ids:
                    self.G, go_id_is_missing = self.U.add_edge_mrna_go_annot(self.G, mrna_node_id, go_id, annot_type=self.U.eggnog)
                    if go_id_is_missing:
                        missing_go_ids.append(go_id)
                    go_count += 1
        # log
        unq_missing = set(missing_go_ids)
        dep = [m for m in unq_missing if m in self.ontology.deprecated_nodes]
        self.logger.info(f"{strain}: out of {go_count} EggNog GO annotations, missing: {len(missing_go_ids)} ({len(unq_missing)} unique), deprecated: {len(dep)}")

    def _assert_mrna_nodes_addition(self, strain):
        raw_mrna = sorted(self.strains_data[strain]['all_mrna'][self.mrna_acc_col])
        graph_mrna = sorted([n for n, d in self.G.nodes(data=True) if d['type'] == self.U.mrna and d['strain'] == strain])
        assert raw_mrna == graph_mrna, f"mRNA nodes in the graph do not match the raw data for strain {strain}"
    
    def _assert_srna_nodes_addition(self, strain):
        raw_srna = sorted(self.strains_data[strain]['all_srna'][self.srna_acc_col])
        graph_srna = sorted([n for n, d in self.G.nodes(data=True) if d['type'] == self.U.srna and d['strain'] == strain])
        assert raw_srna == graph_srna, f"sRNA nodes in the graph do not match the raw data for strain {strain}"
    
    def _assert_srna_mrna_inter_addition(self, strain):
        raw_intr = sorted(zip(self.strains_data[strain]['unq_inter'][self.srna_acc_col], self.strains_data[strain]['unq_inter'][self.mrna_acc_col]))
        graph_intr = sorted([(u, v) for u, v, d in self.G.edges(data=True) if d['type'] == self.U.targets and 
                             self.G.nodes[u]['strain'] == strain
                             and self.G.nodes[v]['strain'] == strain])
        assert raw_intr == graph_intr, f"sRNA-mRNA interactions in the graph do not match the raw data for strain {strain}"
    