from typing import Tuple, Set, List, Dict
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import pickle
from os.path import join
import itertools
from pathlib import Path
from utils.general import read_df, write_df, create_dir_if_not_exists
from preprocessing.general_pr import convert_count_to_val
import networkx as nx
from pyvis.network import Network
from scipy.stats import hypergeom, false_discovery_control
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
from itertools import combinations
import ast
import sys
import os

ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

class Analyzer:
    def __init__(self, config, logger, graph_builder, graph_utils):
        """
        GO node is represented as a dict item:
            <id_str> : {'type': <str>, 'lbl': <str>, 'meta': <dict>}
        mRNA/sRNA node is represented as a dict item:
            <id_str> : {'type': <str>, 'strain': <str>, 'locus_tag': <str>, 'name': <str>, 'synonyms': <List[str]>, 'start': <float>, 'end': <float>, 'strand': <str>, 'sequence': <str>}
        edge is represented as a dict item:
            (<id_str>, <id_str>) : {'type': <str>}
        """
        self.logger = logger
        self.logger.info(f"initializing Analyzer")
        self.config = config
        self.bp_clustering_config = self.config['bp_clustering_config']
        
        # graph and utils
        self.graph_version = graph_builder.get_version()
        self.G = graph_builder.get_graph()
        self.U = graph_utils

        # ---------  RUNTIME FLAGS  ---------
        self.run_clustering_of_rna_homologs = False        # Chapter 4.3.3: Clustering of RNA Homologs Across Multiple Strains
        self.run_clustering_of_rna_paralogs_only = False   # Chapter 4.3.3: Clustering of RNA Homologs Across Multiple Strains

        self.run_homolog_clusters_stats = True            # Chapter 4.3.3: Clustering of RNA Homologs Across Multiple Strains
        
        # ---------  CONFIGURATIONS  ---------
        self.run_enrichment = self.config['run_enrichment']
        self.run_multiple_testing_correction = self.config['run_multiple_testing_correction']
        self.enrichment_pv_threshold = self.config['enrichment_pv_threshold']
        
        conf_str = f"{self.graph_version}{'_w_Enrichment' if self.run_enrichment else ''}{'_MTC' if self.run_multiple_testing_correction else ''}"

        # output paths
        parent_dir = join(self.config['analysis_output_dir'], conf_str)
        self.out_path_clustering_homologs = create_dir_if_not_exists(join(parent_dir, "Clustering_homologs"))
        self.out_path_clustering_paralogs_only = create_dir_if_not_exists(join(parent_dir, "Clustering_paralogs_only"))
        self.out_path_rna_homologs_multi_strains = create_dir_if_not_exists(join(parent_dir, "RNA_homologs_multi_strains"))
        self.out_path_enrichment = create_dir_if_not_exists(join(parent_dir, "Enrichment"))
        self.out_path_summary_tables = create_dir_if_not_exists(join(parent_dir, "Summary_tables"))
        self.out_path_analysis_tool_1 = create_dir_if_not_exists(join(parent_dir, "Analysis_tool_1"))
        self.out_path_analysis_tool_2 = create_dir_if_not_exists(join(parent_dir, "Analysis_tool_2"))

        # file names suffix
        self.out_file_suffix = f"v_{conf_str}"
        
        # analysis 1 params
        self.srna_cluster_id_col = 'srna_homologs_cluster_id'
        self.srna_subgroup_id_col = 'subgroup_id'

        # TODO: remove after testing
        self.strains_data = graph_builder.strains_data
        self.ips_go_annotations = graph_builder.get_ips_go_annotations()

    def run_analysis(self):
        """
        GO node is represented as a dict item:
            <id_str> : {'type': <str>, 'lbl': <str>, 'meta': <dict>}
        mRNA/sRNA node is represented as a dict item:
            <id_str> : {'type': <str>, 'strain': <str>, 'locus_tag': <str>, 'name': <str>, 'synonyms': <List[str]>, 'start': <float>, 'end': <float>, 'strand': <str>, 'sequence': <str>}
        edge is represented as a dict item:
            (<id_str>, <id_str>) : {'type': <str>}
        """
        self.logger.info(f"Running Analysis...")
        self.logger.info(f"--------------   Preprocess   --------------")
        # 1 - Dump BPs of annotated mRNAs
        self._dump_bps_of_annotated_mrnas()

        # 2 - Cluster RNA homologs
        if self.run_clustering_of_rna_homologs:   
            self._cluster_rna_homologs()
        if self.run_clustering_of_rna_paralogs_only:   
            self._cluster_rna_paralogs_only()
        self.srna_homologs = read_df(join(self.out_path_clustering_homologs, f"sRNA_homologs__{self.out_file_suffix}.csv"))
        self.mrna_homologs = read_df(join(self.out_path_clustering_homologs, f"mRNA_homologs__{self.out_file_suffix}.csv"))

        # 3 - Calculate and dump statistics of homolog clusters
        if self.run_homolog_clusters_stats:
            self._dump_stats_rna_orthologs_n_paralogs_per_strain(val_type = 'ratio')
            self._dump_stats_rna_homolog_clusters_size(val_type = 'ratio')
            self._dump_stats_rna_homolog_clusters_strains_composition(val_type = 'ratio', min_val_limit = 0.020)

        # 4 - Map sRNAs to biological processes (BPs)
        self.logger.info("----- Before enrichment:")
        # 4.1 - extract subgraphs
        srna_bp_mapping = self._generate_srna_bp_mapping()
        bp_rna_mapping = self._generate_bp_rna_mapping(srna_bp_mapping)
        # 4.2 - log
        self._log_srna_bp_mapping(srna_bp_mapping)

        # 5 - run Wang similarity on all BPs
        bp_to_cluster = self._run_wang_similarity_between_bps(bp_rna_mapping)
        self._log_bp_clustering_stats(bp_to_cluster)
        
        # 6 - Enrichment (per strain): per sRNA, find and keep only significant biological processes (BPs) that its targets are invovlved in.
        if self.run_enrichment:
            self.logger.info("----- After enrichment:")
            # 6.1 - extract subgraphs
            srna_bp_mapping_en, meta_en = self._apply_enrichment(srna_bp_mapping)
            bp_rna_mapping_en = self._generate_bp_rna_mapping(srna_bp_mapping_en)
            # 6.2 - log and dump
            self._log_srna_bp_mapping(srna_bp_mapping_en)
            self._dump_metadata(meta_en)

        self.logger.info(f"--------------   Analysis Tools   --------------")
        # ------   Analysis 1 - Cross-Species Conservation of sRNAs' Functionality
        self._analysis_1_srna_homologs_to_commom_bps(srna_bp_mapping, bp_to_cluster)
        # self._analysis_1_srna_homologs_to_commom_bps(srna_bp_mapping_en, self.U.exact_bp)  # TODO: add indication when dumping results

        # ------   Analysis 2 - sRNA Regulation of Biological Processes (BPs)
        self._analysis_2_bp_rna_mapping(bp_rna_mapping)
        # self._analysis_2_bp_rna_mapping(bp_rna_mapping_en)  # TODO: add indication when dumping results
    
    def _run_ad_hoc_outputs_analysis_1(self, srnas_cluster: Tuple[str], bp_str: str):
        self.logger.info(f"Running Ad-hoc Analysis...")
        self.srna_homologs = read_df(join(self.out_path_clustering_homologs, f"sRNA_homologs__{self.out_file_suffix}.csv"))
        self.mrna_homologs = read_df(join(self.out_path_clustering_homologs, f"mRNA_homologs__{self.out_file_suffix}.csv"))
        df = read_df(join(self.out_path_analysis_tool_1, f"Tool_1__sRNA_homologs_to_common_BPs__{self.out_file_suffix}.csv"))
        
        # 1 - get tree of sRNA homologs cluster
        srnas_to_targets_to_bps = ast.literal_eval(df[df['cluster'] == str(srnas_cluster)]['srnas_to_targets_to_BPs'].values[0])
        
        # 2 - pruned tree: keep only paths to the given BP
        pruned = {}
        targets_strs = set()
        for srna, targets_to_bps in srnas_to_targets_to_bps.items():
            for target, bp_strs in targets_to_bps.items():
                if bp_str in bp_strs:
                    if not srna in pruned.keys():
                        pruned[srna] = {}
                    pruned[srna][target] = bp_str
                    targets_strs.add(target)
        targets_ids = sorted(set([s.split("__")[0] for s in targets_strs]))
        
        # 3 - homolog clusters of targets in pruned
        complete_ortholog_clusters_of_targets = self._get_orthologs_clusters(targets_ids, 'mRNA')
        filtered_ortholog_clusters_of_targets = [tuple(set(tpl).intersection(targets_ids)) for tpl in complete_ortholog_clusters_of_targets if len(set(tpl).intersection(targets_ids)) > 1]        
        
        # 4 - add original info
        complete_ortholog_clusters_of_targets = [tuple([f"{id}__{self.G.nodes[id]['name']}" for id in tpl]) for tpl in complete_ortholog_clusters_of_targets]
        filtered_ortholog_clusters_of_targets = [tuple([f"{id}__{self.G.nodes[id]['name']}" for id in tpl]) for tpl in filtered_ortholog_clusters_of_targets]
        # 4 - dump
        with open(join(self.out_path_analysis_tool_1, f"Tool_1__Ad-hoc_{srnas_cluster[0]}_{bp_str}_{self.graph_version}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Cluster: \n{srnas_cluster}\n\n")
            f.write(f"BP: \n{bp_str}\n\n")
            f.write(f"PRUNED_srnas_to_targets_to_BPs: \n{pruned}\n\n")
            f.write(f"PRUNED_targets: \n{sorted(targets_strs)}\n\n")
            f.write(f"PRUNED_filtered_ortholog_clusters_of_targets: \n{filtered_ortholog_clusters_of_targets}\n\n")
            f.write(f"PRUNED_complete_ortholog_clusters_of_targets: \n{complete_ortholog_clusters_of_targets}\n\n")

        return pruned

    def _cluster_rna_homologs(self):
        self._cluster_homologs('sRNA', self.U.srna)
        self._cluster_homologs('mRNA', self.U.mrna)
    
    def _cluster_rna_paralogs_only(self):
        self._cluster_paralogs_only('sRNA', self.U.srna)
        self._cluster_paralogs_only('mRNA', self.U.mrna)

    def _generate_srna_bp_mapping(self) -> dict:
        """
        Generate a mapping of sRNA to biological processes (BPs) based on the edges in the graph.
        
        Returns:
            dict: A dictionary in the following format:
            {
                <strain_id>: {
                        <sRNA_id>: {
                            <mRNA_target_id>: [<bp_id1>, <bp_id2>, ...],
                            ...
                        },
                        ...
                },
                ...
            }               
        """
        bp_mapping = {}  
        for strain in self.U.strains:
            unq_targets_without_bp = set()
            # Filter sRNA nodes for the current strain
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter']['sRNA_accession_id'] == 'G0-16636']
            srna_bp_mapping = {}
            srna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.srna and d['strain'] == strain]
            for srna in srna_nodes:
                targets_to_bp = {}
                srna_targets = [n for n in self.G.neighbors(srna) if self.G.nodes[n]['type'] == self.U.mrna and self.U.is_target(self.G, srna, n)]
                for target in srna_targets:
                    # Find the biological processes associated with the target
                    bp_nodes = [n for n in self.G.neighbors(target) if self.U.is_annotated(self.G, target, n, self.U.bp)]
                    if bp_nodes:
                        targets_to_bp[target] = bp_nodes
                    else:
                        unq_targets_without_bp.add(target)
                if targets_to_bp:
                    srna_bp_mapping[srna] = targets_to_bp
            
            bp_mapping[strain] = srna_bp_mapping
            # self._log_mrna_without_bp(strain, unq_targets_without_bp)  # check if they have IPS MF/CC annotations

        return bp_mapping
    
    def _generate_bp_rna_mapping(self, srna_bp_mapping: dict):
        """Generate a mapping of BP to mRNAs and sRNAs based on the srna_bp_mapping.

        Args:
            srna_bp_mapping (dict): A dictionary in the following format:
            {
                <strain_id>: {
                        <sRNA_id>: {
                            <mRNA_target_id>: [<bp_id1>, <bp_id2>, ...],
                            ...
                        },
                        ...
                },
                ...
            }      

        Returns:
            dict: A dictionary in the following format:
            {
                <bp_id>: {
                        <strain_id>: {
                            <mRNA_target_id>: [<sRNA_id1>, <sRNA_id2>, ...],   # if the mRNA target has sRNA interactions
                            <mRNA_target_id>: []                               # if the mRNA target has no sRNA interactions but is annotated with the BP
                            ...
                        },
                        ...
                },
                ...
            }   
        """
        # 1 - get all unique BP ids
        unq_bps = set()
        for strain_dict in srna_bp_mapping.values():
            for srna_targets in strain_dict.values():
                for mrna_bps in srna_targets.values():
                    unq_bps.update(mrna_bps)
        unq_bps = sorted(unq_bps)

        # 2 - build mapping
        bp_rna_mapping = {}
        for bp in unq_bps:
            bp_rna_mapping[bp] = {}
            for strain, srna_dict in srna_bp_mapping.items():
                # per strain: 
                # (2.1) find all interacting mRNAs that relate to curr BP
                strain_mrna_to_srnas = {}
                for srna_id, mrna_to_bps in srna_dict.items():
                    for mrna_id, bps in mrna_to_bps.items():
                        if bp in bps:
                            if mrna_id not in strain_mrna_to_srnas:
                                strain_mrna_to_srnas[mrna_id] = []
                            strain_mrna_to_srnas[mrna_id].append(srna_id)
                if strain_mrna_to_srnas:
                    bp_rna_mapping[bp][strain] = strain_mrna_to_srnas
                # (2.2) add also strain's annotated mRNAs that do not interact with sRNAs
                strain_annotated_mrnas = [
                    n for n, d in self.G.nodes(data=True) 
                    if d['type'] == self.U.mrna and d['strain'] == strain 
                    and self.G.has_edge(n, bp) and any(edge_data.get('type') == self.U.annotated for edge_data in self.G[n][bp].values())
                ]
                for mrna_id in strain_annotated_mrnas:
                    if strain not in bp_rna_mapping[bp].keys():
                        bp_rna_mapping[bp][strain] = {}
                    if mrna_id not in bp_rna_mapping[bp][strain].keys():
                        bp_rna_mapping[bp][strain][mrna_id] = []

        return bp_rna_mapping
    
    def _get_bp_clustering_params_dir_n_nm(self) -> tuple:
        linkage_method = self.bp_clustering_config['linkage_method']    # 'single', 'complete', 'average'
        threshold_dist_prec = self.bp_clustering_config['threshold_distance_percentile']  # 10, 20 ...
        _dir = create_dir_if_not_exists(join(self.config['bp_clustering_dir'], linkage_method))
        f_name = f"bp_clustering_{linkage_method}_threshold_percentile_{threshold_dist_prec}"
        return _dir, f_name, linkage_method, threshold_dist_prec
    
    def _run_wang_similarity_between_bps(self, bp_rna_mapping: dict) -> Dict[str, int]:
        """
        Compute Wang semantic similarity between GO terms (BPs)
        # https://github.com/tanghaibao/goatools/blob/main/notebooks/semantic_similarity_wang.ipynb

        Args:
            bp_rna_mapping (dict): A dictionary in the following format:
            {
                <bp_id>: {
                        <strain_id>: {
                            <mRNA_target_id>: [<sRNA_id1>, <sRNA_id2>, ...],
                            ...
                        },
                        ...
                },
                ...
            }
        Returns:
            dict: A dictionary mapping BP IDs to their cluster labels.   
        """
        # config params
        _dir, f_name, linkage_method, threshold_dist_prec = self._get_bp_clustering_params_dir_n_nm()
        self.logger.info(f"clustering params - linkage_method: {linkage_method}, threshold_dist_prec: {threshold_dist_prec}")
        
        # load and verify go DAG for wang
        self.logger.debug("loading GO DAG for Wang similarity...")
        godag = get_godag("go-basic.obo", optional_attrs={'relationship'})
        for bp in bp_rna_mapping.keys():
            assert self.G.nodes[bp]['lbl'] == godag[f"GO:{bp}"].name, f"GO term name mismatch for {bp}: {self.G.nodes[bp]['lbl']} != {godag[f'GO:{bp}'].name}"
        # Optional relationships. (Relationship, is_a, is required and always used)
        relationships = {'part_of'}

        goids_sorted = sorted({f'GO:{bp}'for bp in bp_rna_mapping.keys()})
        sim_matrix = pd.DataFrame(None, index=goids_sorted, columns=goids_sorted, dtype=float)
        # set diagonal (self similarity)
        for g in goids_sorted:
            sim_matrix.at[g, g] = 1.0

        self.logger.debug("computing pairwise Wang similarities...")
        wang_r1 = SsWang(goids_sorted, godag, relationships)

        records = []
        for go1, go2 in combinations(goids_sorted, 2):
            sim = wang_r1.get_sim(go1, go2)
            records.append({'go1': go1, 'go2': go2, 'similarity': sim, 'distance': 1 - sim, 'go1_name': godag[go1].name, 'go2_name': godag[go2].name})
            sim_matrix.at[go1, go2] = sim
            sim_matrix.at[go2, go1] = sim
        
        dis_meta = pd.DataFrame(records).sort_values(by='distance').reset_index(drop=True)

        dis_matrix = 1 - sim_matrix
        distances = squareform(dis_matrix.values)
        assert sum([0 <= d <= 1 for d in distances]) == len(distances), "Distances not in [0, 1] range"

        distance_threshold = np.percentile(a=distances, q=threshold_dist_prec)
        self.logger.info(f"distance threshold (at {threshold_dist_prec} percentile): {distance_threshold:.4f}")
        
        Z = linkage(y=distances, method=linkage_method)
        cluster_labels = fcluster(Z, distance_threshold, criterion='distance')

        # map GO IDs to their cluster labels
        bp_clusters = pd.DataFrame({'bp_id': goids_sorted, 'cluster': cluster_labels, 'bp_name': [godag[g].name for g in goids_sorted]}).sort_values(by='cluster').reset_index(drop=True)
        write_df(bp_clusters, join(_dir, f"{f_name}__bp_clusters___{self.out_file_suffix}.csv"))
        write_df(dis_meta, join(_dir, f"{f_name}__dis_meta___{self.out_file_suffix}.csv"))
        bp_to_cluster = dict(zip(bp_clusters['bp_id'].apply(lambda x: x.replace('GO:', '')), bp_clusters['cluster']))

        return bp_to_cluster
    
    def _log_bp_clustering_stats(self, bp_to_cluster: dict):
        self.logger.info(f"##############   BP Clustering Statistics   ##############")
        self.logger.info(f"Number of BPs in sub-graph: {len(bp_to_cluster)}")
        cluster_sizes = pd.Series(list(bp_to_cluster.values())).value_counts().sort_index().reset_index(drop=False).rename(
            columns={'index': 'cluster_id', 'count': 'cluster_size'})
        
        # self.logger.info("------------------------- INCLUDING singletones")
        # # 1 - number of clusters
        # num_clusters = len(cluster_sizes)
        # self.logger.info(f"Number of BP clusters: {num_clusters}")
        # # 2 - cluster size distribution
        # unq, counts = np.unique(cluster_sizes['cluster_size'], return_counts=True)
        # sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
        # cluster_size_dist = "\n   ".join([f"{counts} with size = {unq} ({int(round(counts/num_clusters, 3)*100)} %)" for unq, counts in sorted_dict.items()])
        # self.logger.info(
        #     f"-------   Cluster Size Distribution: \n"
        #     f"   {cluster_size_dist}"
        # )
        
        self.logger.info("------------------------- NO singletones")
        cluster_sizes_no_singletons = cluster_sizes[cluster_sizes['cluster_size'] > 1]
        # 1 - number of clusters
        num_clusters = len(cluster_sizes_no_singletons)
        self.logger.info(f"Number of BP clusters: {num_clusters}")
        # 2 - cluster size distribution
        unq, counts = np.unique(cluster_sizes_no_singletons['cluster_size'], return_counts=True)
        sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
        cluster_size_dist = "\n   ".join([f"{counts} with size = {unq} ({round(counts/num_clusters, 3)*100} %)" for unq, counts in sorted_dict.items()])
        self.logger.info(
            f"-------   Cluster Size Distribution: \n"
            f"   {cluster_size_dist}"
        )

    def _analysis_1_srna_homologs_to_commom_bps(self, srna_bp_mapping: dict, bp_to_cluster: dict, add_pairs_info: bool = False):
        self.logger.info(f"##############   Analsis 1 - Cross-Species Conservation of sRNAs' Functionality   ##############")
        # 1 - load clusters of sRNA homologs
        df = self.srna_homologs.copy()

        dfs = []
        for index, row in df.iterrows():
            cluster_id = index + 1
            cluster = ast.literal_eval(row['cluster'])
            # 2 - dump a csv of the cluster subgraph (tree)
            cluster_srnas_to_targets_to_bps = {}
            cluster_srnas_to_bps = {}
            for srna_id in cluster:
                strain = self.G.nodes[srna_id]['strain']
                srna_targets = srna_bp_mapping[strain].get(srna_id)
                if srna_targets:
                    srna_complete = f"{strain}__{srna_id}__{self.G.nodes[srna_id]['name']}" 
                    # sRNAs to targets to BPs with clusters
                    targets_to_bps = {f"{target_id}__{self.G.nodes[target_id]['name']}": sorted([(bp_to_cluster[bp_id], bp_id, self.G.nodes[bp_id]['lbl']) for bp_id in bps]) for target_id, bps in srna_targets.items()}
                    cluster_srnas_to_targets_to_bps[srna_complete] = targets_to_bps
                    # sRNAs to all its BPs with clusters
                    all_unq_bps = sorted({(bp_to_cluster[bp_id], bp_id, self.G.nodes[bp_id]['lbl']) for bps in srna_targets.values() for bp_id in bps})
                    cluster_srnas_to_bps[srna_complete] = all_unq_bps

            #TODO: dump the tree as a csv (srnas_to_targets_to_bps)

            # 3 - generate cluster df
            # 3.1 - generate the cluster record
            rec = {
                 self.srna_cluster_id_col: cluster_id,
                'cluster': cluster,
                'cluster_size': row['cluster_size'],
                'strains': ast.literal_eval(row['strains']),
                'num_strains': row['num_strains']
            }
            if add_pairs_info:
                rec.update({
                    'ortholog_pairs': ast.literal_eval(row['ortholog_pairs']),
                    'num_ortholog_pairs': row['num_ortholog_pairs'],
                    'paralog_pairs': ast.literal_eval(row['paralog_pairs']),
                    'num_paralog_pairs': row['num_paralog_pairs']
                })
            rec.update({self.srna_subgroup_id_col: 0})
            # 3.2 - add subgroups records
            cluster_df = self._get_cluster_df(rec, cluster_srnas_to_bps, srna_bp_mapping)
            dfs.append(cluster_df)

        out_df = pd.concat(dfs, ignore_index=True)
        # TODO: add sort and reset cluster id ?




        ######### prev
        # 2 - analyze common BPs of sRNA homologs
        records = []
        for cluster in df['cluster'].apply(ast.literal_eval):
            rec = self._get_common_bps_of_srna_orthologs(cluster, srna_bp_mapping, bp_to_cluster)
            records.append(rec)
        df[list(records[0].keys())] = pd.DataFrame(records)
        
        # 3 - score
        max_of_max_filtered_homolog_cluster_size = df['max_filtered_homolog_cluster_size'].max() if df['max_filtered_homolog_cluster_size'].max() > 0 else 1
        df['score'] = 100 * df['max_strains_with_common_BPs'] + 10 * (df['max_filtered_homolog_cluster_size'] / max_of_max_filtered_homolog_cluster_size)
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

        # 4 - dump results
        # 4.1 - csv
        write_df(df, join(self.out_path_analysis_tool_1, f"Tool_1__sRNA_homologs_to_common_BPs_w_clusters__{self.out_file_suffix}.csv"))
        # 4.2 - manual txt
        clusters_for_txt = [('E2348C_ncR46', 'G0-8867', 'GcvB', 'gcvB', 'ncRNA0016'), ('E2348C_ncR22', 'G0-8863', 'gene-SGH10_RS11370', 'ncRNA0059'), ('E2348C_ncR58', 'G0-8871', 'arcZ', 'ncRNA0002'), ('E2348C_ncR33', 'G0-8878', 'cyaR', 'ncRNA0009'), ('E2348C_ncR60', 'G0-8872', 'RyhB', 'ncRNA0030', 'ncRNA0069', 'ryhB-1', 'ryhB-2'), ('E2348C_ncR66', 'EG30098', 'Spot42', 'ncRNA0077', 'spf'), ('E2348CN_0008', 'G0-10671', 'mgrR', 'ncRNA0046'), ('E2348CN_0007', 'G0-10677', 'fnrS', 'ncRNA0015'), ('E2348CN_0013', 'G0-16649', 'cpxQ', 'ncRNA0206'), ('E2348C_ncR48', 'G0-8882', 'ncRNA0052', 'omrB')]
        dump_manual_txt = True
        if dump_manual_txt:
            with open(join(self.out_path_analysis_tool_1, f"Tool_1__Manual_{self.graph_version}.txt"), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    cluster = row['cluster'] if type(row['cluster']) == tuple else ast.literal_eval(row['cluster'])
                    if cluster in clusters_for_txt:
                        f.write(f"{row['cluster']}\n")
                        # f.write(f"{row['srnas_to_targets_to_BPs']}\n\n")
                        f.write(f"{row['srnas_to_targets_to_BPs_w_clusters']}\n\n")

        # 5 - log statistics
        num_clusters = len(df)
        self.logger.info(
            f"----------------   sRNA homologs \n"
            f"Number of sRNA homolog clusters: {len(df)}"
        )
        # 5.1 - max common BPs distribution
        for bp_definition, max_common_col in [(self.U.bp_id, 'max_common_BPs')]:

            unq, counts = np.unique(df[max_common_col], return_counts=True)
            sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
            max_common_dist = "\n   ".join([f"{counts} with {max_common_col} = {unq} ({int(round(counts/num_clusters, 2)*100)} %)" for unq, counts in sorted_dict.items()])
            
            self.logger.info(
                f"-------   BP definition = {bp_definition} \n"
                f"{max_common_col} distribution: \n"
                f"   {max_common_dist}"
            )
    
    def _PREV_analysis_1_srna_homologs_to_commom_bps(self, srna_bp_mapping: dict, bp_to_cluster: dict):
        self.logger.info(f"##############   Analsis 1 - Cross-Species Conservation of sRNAs' Functionality   ##############")
        # 1 - load clusters of sRNA homologs
        df = self.srna_homologs.copy()

        # 2 - analyze common BPs of sRNA homologs
        records = []
        for cluster in df['cluster'].apply(ast.literal_eval):
            rec = self._get_common_bps_of_srna_orthologs(cluster, srna_bp_mapping, bp_to_cluster)
            records.append(rec)
        df[list(records[0].keys())] = pd.DataFrame(records)
        
        # 3 - score
        max_of_max_filtered_homolog_cluster_size = df['max_filtered_homolog_cluster_size'].max() if df['max_filtered_homolog_cluster_size'].max() > 0 else 1
        df['score'] = 100 * df['max_strains_with_common_BPs'] + 10 * (df['max_filtered_homolog_cluster_size'] / max_of_max_filtered_homolog_cluster_size)
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

        # 4 - dump results
        # 4.1 - csv
        write_df(df, join(self.out_path_analysis_tool_1, f"Tool_1__sRNA_homologs_to_common_BPs_w_clusters__{self.out_file_suffix}.csv"))
        # 4.2 - manual txt
        clusters_for_txt = [('E2348C_ncR46', 'G0-8867', 'GcvB', 'gcvB', 'ncRNA0016'), ('E2348C_ncR22', 'G0-8863', 'gene-SGH10_RS11370', 'ncRNA0059'), ('E2348C_ncR58', 'G0-8871', 'arcZ', 'ncRNA0002'), ('E2348C_ncR33', 'G0-8878', 'cyaR', 'ncRNA0009'), ('E2348C_ncR60', 'G0-8872', 'RyhB', 'ncRNA0030', 'ncRNA0069', 'ryhB-1', 'ryhB-2'), ('E2348C_ncR66', 'EG30098', 'Spot42', 'ncRNA0077', 'spf'), ('E2348CN_0008', 'G0-10671', 'mgrR', 'ncRNA0046'), ('E2348CN_0007', 'G0-10677', 'fnrS', 'ncRNA0015'), ('E2348CN_0013', 'G0-16649', 'cpxQ', 'ncRNA0206'), ('E2348C_ncR48', 'G0-8882', 'ncRNA0052', 'omrB')]
        dump_manual_txt = True
        if dump_manual_txt:
            with open(join(self.out_path_analysis_tool_1, f"Tool_1__Manual_{self.graph_version}.txt"), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    cluster = row['cluster'] if type(row['cluster']) == tuple else ast.literal_eval(row['cluster'])
                    if cluster in clusters_for_txt:
                        f.write(f"{row['cluster']}\n")
                        # f.write(f"{row['srnas_to_targets_to_BPs']}\n\n")
                        f.write(f"{row['srnas_to_targets_to_BPs_w_clusters']}\n\n")

        # 5 - log statistics
        num_clusters = len(df)
        self.logger.info(
            f"----------------   sRNA homologs \n"
            f"Number of sRNA homolog clusters: {len(df)}"
        )
        # 5.1 - max common BPs distribution
        for bp_definition, max_common_col in [(self.U.bp_id, 'max_common_BPs')]:

            unq, counts = np.unique(df[max_common_col], return_counts=True)
            sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
            max_common_dist = "\n   ".join([f"{counts} with {max_common_col} = {unq} ({int(round(counts/num_clusters, 2)*100)} %)" for unq, counts in sorted_dict.items()])
            
            self.logger.info(
                f"-------   BP definition = {bp_definition} \n"
                f"{max_common_col} distribution: \n"
                f"   {max_common_dist}"
            )

    def _get_cluster_df(self, rec: dict, cluster_srnas_to_bps: dict, srna_bp_mapping: dict) -> pd.DataFrame:
        sub_recs = []

        #cluster_id_col: str, subgroup_id_col: str, cluster_id: int, orthologs_cluster: Tuple[str]
        cluster_id = rec[self.srna_cluster_id_col]
        orthologs_cluster = rec['cluster']
        
        # _all_common_bps, _num_common_bps = {}, {}

        
        # _all_common_bp_clusters, _num_common_bp_clusters = {}, {}


        # _new_common_bps_added_by_clustering = {}
        # _common_bps_extended_by_clustering = {}


        out_df = pd.DataFrame([rec])

        for size in range(2, len(cluster_srnas_to_bps.keys()) + 1):
            for srna_subgroup in itertools.combinations(cluster_srnas_to_bps.keys(), size):
                bps_of_srnas = list(map(cluster_srnas_to_bps.get, srna_subgroup))
                
                # 1 - common BP clusters of sRNAs
                common_bp_clusters: List[Tuple[str, str, str]] = bps_of_srnas[0]
                for l in bps_of_srnas[1:]:
                    common_bp_clusters = self._find_common_bps_by_cluster(common_bp_clusters, l)

                if common_bp_clusters:
                    # 2 - add subgroup to out df
                    subgroup_size = len(srna_subgroup)
                    subgroup_strains = sorted(set([rna.split('__')[0] for rna in srna_subgroup]))
                    subgroup_num_strains = len(subgroup_strains)

                    num_common_bp_clusters = len(set([clus for clus, bp, nm in common_bp_clusters]))

                    # 3 - common BP ids of sRNAs
                    common_bp_ids: List[Tuple[str, str, str]] = bps_of_srnas[0]
                    for l in bps_of_srnas[1:]:
                        common_bp_ids = self._find_common_bp_ids(common_bp_ids, l)
                    
                    # 4 - get expanded and emergent BP clusters
                    expanded_bp_clusters, emergent_bp_clusters = self._get_expanded_and_emergent_bp_clusters(common_bp_clusters, common_bp_ids)
                    num_emergent_bp_clusters = len(set([clus for clus, bp, nm in emergent_bp_clusters]))



                    #  - srnas_to_targets_to_common_BP_clusters
                    #TODO
                    # srnas_to_targets_to_common_bp_clusters  - dump as json file (beautifier?)

                    #  - homolog clusters of targets
                    #TODO
                    # homolog_clusters_of_targets


                    subbgroup_rec = {
                        'srna_subgroup': srna_subgroup,
                        'subgroup_size': subgroup_size,
                        'subgroup_strains': subgroup_strains,
                        'subgroup_num_strains': subgroup_num_strains,
                        'common_bp_clusters': common_bp_clusters,
                        'num_common_bp_clusters': num_common_bp_clusters,
                        'common_bp_ids': common_bp_ids,
                        'expanded_bp_clusters': expanded_bp_clusters,
                        'emergent_bp_clusters': emergent_bp_clusters,
                        'num_emergent_bp_clusters': num_emergent_bp_clusters,
                        'homolog_clusters_of_targets': None,  # TODO
                    }

        # TODO: - add subgroup id after sort by subgroup size


        ###############################
        ###############################
        # out_rec.update(rec)

        # # 4 - homolog clusters of targets (cross-strains)
        # # complete clusters
        # complete_homolog_clusters_of_targets = self._find_homologs(strain_to_mrna_list, 'mRNA')
        # # filtered clusters - only those that contain at least two targets of sRNAs in the cluster
        # relevant_mrnas = set()
        # for rna_list in strain_to_mrna_list.values():
        #     relevant_mrnas.update(rna_list)
        
        # filtered_homolog_clusters_of_targets = set()
        # for cluster in complete_homolog_clusters_of_targets:
        #     rna_homologs = tuple(sorted(set(cluster).intersection(relevant_mrnas)))
        #     if len(rna_homologs) >= 2:
        #         filtered_homolog_clusters_of_targets.add(rna_homologs)
        # # add info + sort clusters by size (length) from largest to smallest
        # complete_homolog_clusters_of_targets = sorted([tuple(sorted([f"{self.G.nodes[rna]['strain']}__{rna}__{self.G.nodes[rna]['name']}" for rna in cluster])) for cluster in complete_homolog_clusters_of_targets], key=lambda x: len(x), reverse=True)
        # filtered_homolog_clusters_of_targets = sorted([tuple(sorted([f"{self.G.nodes[rna]['strain']}__{rna}__{self.G.nodes[rna]['name']}" for rna in cluster])) for cluster in filtered_homolog_clusters_of_targets], key=lambda x: len(x), reverse=True)

        # # 5 - max filtered homolog cluster size
        # max_filtered_homolog_cluster_size = max([len(c) for c in filtered_homolog_clusters_of_targets], default=0)

        # # 6 - output record
        # out_rec.update({
        #     # 'all_BPs': all_bps,  # REMOVED
        #     'all_BPs_w_clusters': all_bps_w_clusters,  # NEW
        #     'num_all_BPs': num_all_bps,
        #     'num_all_BP_clusters': num_all_bp_clusters, # NEW
        #     'complete_homolog_clusters_of_targets': complete_homolog_clusters_of_targets,
        #     'filtered_homolog_clusters_of_targets': filtered_homolog_clusters_of_targets,
        #     'max_filtered_homolog_cluster_size': max_filtered_homolog_cluster_size,
        #     # 'srnas_to_targets_to_BPs': srnas_to_targets_to_bps,  # REMOVED
        #     'srnas_to_targets_to_BPs_w_clusters': srnas_to_targets_to_bps_w_clusters,  # NEW
        # })

        # recs.append(out_rec)
        return out_df
    
    def _find_common_bps_by_cluster(self, bps1: List[Tuple[str, str, str]], bps2: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        common_clus = np.intersect1d([x[0] for x in bps1], [x[0] for x in bps2])
        common_bps_by_clus = [t for t in set(bps1).union(set(bps2)) if t[0] in common_clus]
        return common_bps_by_clus

    def _find_common_bp_ids(self, bps1: List[Tuple[str, str, str]], bps2: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        common_bps = sorted(set(bps1).intersection(set(bps2)))
        return common_bps
    
    def _get_expanded_and_emergent_bp_clusters(self, common_bp_clusters: List[Tuple[str, str, str]], common_bp_ids: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """_summary_

        Args:
            common_bp_clusters (List[Tuple[str, str, str]]): list of tuples (cluster_id, bp_id)
            common_bp_ids (List[Tuple[str, str, str]]): list of tuples (cluster_id, bp_id)

        Returns:
            expanded_bp_clusters  (List[Tuple[str, str, str]])
            emergent_bp_clusters  (List[Tuple[str, str, str]])
        """
        expanded_bp_clusters, emergent_bp_clusters = [], []
        if len(common_bp_clusters) > len(common_bp_ids):
            new_bps = sorted(set(common_bp_clusters) - set(common_bp_ids))
            new_clusters = sorted(set([clus for clus, bp, nm in common_bp_clusters]) - set([clus for clus, bp, nm in common_bp_ids])) 
            expanded_bp_clusters = [(clus, bp, nm) for clus, bp, nm in new_bps if clus not in new_clusters]
            emergent_bp_clusters = [(clus, bp, nm) for clus, bp, nm in new_bps if clus in new_clusters]
        
        return expanded_bp_clusters, emergent_bp_clusters
    
    def PREV_get_common_bps_of_srna_orthologs(self, orthologs_cluster: Tuple[str], srna_bp_mapping: dict, bp_to_cluster: dict) -> dict:
        # 1 - all BPs
        all_bps, num_all_bps = {}, {}
        all_bp_clusters, num_all_bp_clusters = {}, {}
        # 2 - sRNAs to targets to BPs w clusters (complete)
        srnas_to_targets_to_bps_w_clusters = {}
        # for orthologs clusters of targets
        strain_to_mrna_list = {}

        for srna_id in orthologs_cluster:
            strain = self.G.nodes[srna_id]['strain']
            srna_targets = srna_bp_mapping[strain].get(srna_id)
            if srna_targets:
                unq_bps = sorted({bp for bps in srna_targets.values() for bp in bps})
                all_bps[f'{strain}__{srna_id}'] = unq_bps
                num_all_bps[f'{strain}__{srna_id}'] = len(unq_bps)
                
                unq_bp_clusters = sorted({bp_to_cluster[bp_id] for bp_id in unq_bps})
                all_bp_clusters[f'{strain}__{srna_id}'] = unq_bp_clusters
                num_all_bp_clusters[f'{strain}__{srna_id}'] = len(unq_bp_clusters)

                # sRNAs to targets to BPs (complete)
                srna_complete = f"{strain}__{srna_id}__{self.G.nodes[srna_id]['name']}" 
                targets_to_bps_complete = {f"{target_id}__{self.G.nodes[target_id]['name']}": sorted([(bp_to_cluster[bp_id], bp_id, self.G.nodes[bp_id]['lbl']) for bp_id in bps]) for target_id, bps in srna_targets.items()}
                srnas_to_targets_to_bps_w_clusters[srna_complete] = targets_to_bps_complete

                # for orthologs clusters of targets
                if strain not in strain_to_mrna_list.keys():
                    strain_to_mrna_list[strain] = []
                strain_to_mrna_list[strain].extend(list(srna_targets.keys()))

        num_srna_w_bps = len(all_bps)
        srnas_to_targets_to_bps_w_clusters = dict(sorted(srnas_to_targets_to_bps_w_clusters.items()))

        out_rec = {'num_srna_w_bps': num_srna_w_bps}
        
        # 3 - common BPs
        # all_common_bps, num_common_bps, max_strains_w_common_bps, all_common_bps_of_max_strains, max_common_bps = self._calc_common_bps(all_bps)
        all_bps_w_clusters = {}
        for srna, bps in all_bps.items():
            all_bps_w_clusters[srna] = [(bp_to_cluster[bp], bp) for bp in bps]
        rec = self._calc_common_bps_n_clusters(all_bps_w_clusters)
        out_rec.update(rec)

        # 4 - homolog clusters of targets (cross-strains)
        # complete clusters
        complete_homolog_clusters_of_targets = self._find_homologs(strain_to_mrna_list, 'mRNA')
        # filtered clusters - only those that contain at least two targets of sRNAs in the cluster
        relevant_mrnas = set()
        for rna_list in strain_to_mrna_list.values():
            relevant_mrnas.update(rna_list)
        
        filtered_homolog_clusters_of_targets = set()
        for cluster in complete_homolog_clusters_of_targets:
            rna_homologs = tuple(sorted(set(cluster).intersection(relevant_mrnas)))
            if len(rna_homologs) >= 2:
                filtered_homolog_clusters_of_targets.add(rna_homologs)
        # add info + sort clusters by size (length) from largest to smallest
        complete_homolog_clusters_of_targets = sorted([tuple(sorted([f"{self.G.nodes[rna]['strain']}__{rna}__{self.G.nodes[rna]['name']}" for rna in cluster])) for cluster in complete_homolog_clusters_of_targets], key=lambda x: len(x), reverse=True)
        filtered_homolog_clusters_of_targets = sorted([tuple(sorted([f"{self.G.nodes[rna]['strain']}__{rna}__{self.G.nodes[rna]['name']}" for rna in cluster])) for cluster in filtered_homolog_clusters_of_targets], key=lambda x: len(x), reverse=True)

        # 5 - max filtered homolog cluster size
        max_filtered_homolog_cluster_size = max([len(c) for c in filtered_homolog_clusters_of_targets], default=0)

        # 6 - output record
        out_rec.update({
            # 'all_BPs': all_bps,  # REMOVED
            'all_BPs_w_clusters': all_bps_w_clusters,  # NEW
            'num_all_BPs': num_all_bps,
            'num_all_BP_clusters': num_all_bp_clusters, # NEW
            'complete_homolog_clusters_of_targets': complete_homolog_clusters_of_targets,
            'filtered_homolog_clusters_of_targets': filtered_homolog_clusters_of_targets,
            'max_filtered_homolog_cluster_size': max_filtered_homolog_cluster_size,
            # 'srnas_to_targets_to_BPs': srnas_to_targets_to_bps,  # REMOVED
            'srnas_to_targets_to_BPs_w_clusters': srnas_to_targets_to_bps_w_clusters,  # NEW
        })

        return out_rec

    def PREV_calc_common_bps_n_clusters(self, all_bps_w_clusters: Dict[str, List[Tuple[str, str]]]) -> dict:
        """_summary_

        Args:
            all_bps_w_clusters (Dict[str, List[Tuple[str, str]]]): 
                mapping of sRNA ('{strain}__{srna_id}') to a list of all its unique BPs w clusters, i.e., list of tuples (cluster_id, bp_id)

        Returns:
        """

        _all_common_bps, _num_common_bps = {}, {}
        _max_common_bps = 0
        _max_strains_w_common_bps = 0
        
        _all_common_bp_clusters, _num_common_bp_clusters = {}, {}
        _max_common_bp_clusters = 0
        _max_strains_w_common_bp_clusters = 0

        _new_common_bps_added_by_clustering = {}
        _common_bps_extended_by_clustering = {}

        for size in range(2, len(all_bps_w_clusters.keys()) + 1):
            for rna_comb in itertools.combinations(all_bps_w_clusters.keys(), size):
                # RNAs info
                rnas_strains = set([rna.split('__')[0] for rna in rna_comb])
                rnas_bps_w_cluster = list(map(all_bps_w_clusters.get, rna_comb))
                
                # 1 - common BPs of RNAs
                _common_bps: List[Tuple[str, str]] = rnas_bps_w_cluster[0]
                for l in rnas_bps_w_cluster[1:]:
                    _common_bps = self.U.find_common_bps(_common_bps, l)
                
                if _common_bps:
                    _all_common_bps[rna_comb] = [(clus, bp, self.G.nodes[bp]['lbl']) for clus, bp in _common_bps]
                    NUM_COMMON_BPS = len(_common_bps)
                    _num_common_bps[rna_comb] = NUM_COMMON_BPS
                    _max_strains_w_common_bps = max(_max_strains_w_common_bps, len(rnas_strains))
                    _max_common_bps = max(_max_common_bps, NUM_COMMON_BPS)

                # 2 - common BPs of RNAs by cluster
                _common_bp_clusters: List[Tuple[str, str]] = rnas_bps_w_cluster[0]
                for l in rnas_bps_w_cluster[1:]:
                    _common_bp_clusters = self.U.find_common_bps_by_cluster(_common_bp_clusters, l)
                
                if _common_bp_clusters:
                    _all_common_bp_clusters[rna_comb] = [(clus, bp, self.G.nodes[bp]['lbl']) for clus, bp in _common_bp_clusters]
                    NUM_COMMON_BP_CLUSTERS = len(set([clus for clus, bp in _common_bp_clusters]))
                    _num_common_bp_clusters[rna_comb] = NUM_COMMON_BP_CLUSTERS
                    _max_strains_w_common_bp_clusters = max(_max_strains_w_common_bp_clusters, len(rnas_strains))
                    _max_common_bp_clusters = max(_max_common_bp_clusters, NUM_COMMON_BP_CLUSTERS)

                # check which new common BPs were added by clustering
                new_clusters_added, new_bps_of_clusters = self._get_clusters_of_common_bps_added(_common_bp_clusters, _common_bps)
                
                if new_clusters_added:
                    common_bps_added_per_srna = {}
                    for srna in rna_comb:
                        common_bps_added = sorted([(clus, bp, self.G.nodes[bp]['lbl']) for clus, bp in all_bps_w_clusters[srna] if clus in new_clusters_added])
                        if common_bps_added:
                            common_bps_added_per_srna[srna] = common_bps_added
                    _new_common_bps_added_by_clustering[rna_comb] = common_bps_added_per_srna
                
                if new_bps_of_clusters:
                    common_bps_extended_per_srna = {}
                    for srna in rna_comb:
                        common_bps_extended = sorted([(clus, bp, self.G.nodes[bp]['lbl']) for clus, bp in all_bps_w_clusters[srna] if (clus, bp) in new_bps_of_clusters])
                        if common_bps_extended:
                            common_bps_extended_per_srna[srna] = common_bps_extended
                    _common_bps_extended_by_clustering[rna_comb] = common_bps_extended_per_srna

                
        _all_common_bps_of_max_strains = {}
        for rna_comb, bp_lst in _all_common_bps.items():
            # get all common BPs of max strains
            rnas_strains = set([rna.split('__')[0] for rna in rna_comb])
            if len(rnas_strains) == _max_strains_w_common_bps:
                _all_common_bps_of_max_strains[rna_comb] = _all_common_bps[rna_comb]
        
        _all_common_bp_clusters_of_max_strains = {}
        for rna_comb, bp_lst in _all_common_bp_clusters.items():
            # get all common BP clusters of max strains
            rnas_strains = set([rna.split('__')[0] for rna in rna_comb])
            if len(rnas_strains) == _max_strains_w_common_bp_clusters:
                _all_common_bp_clusters_of_max_strains[rna_comb] = _all_common_bp_clusters[rna_comb]
        
        rec = {
            'all_common_BPs': _all_common_bps,
            'num_common_BPs': _num_common_bps,
            'max_strains_with_common_BPs': _max_strains_w_common_bps,
            'all_common_BPs_of_max_strains': _all_common_bps_of_max_strains,
            'max_common_BPs': _max_common_bps,

            'new_common_BPs_added_by_clustering': _new_common_bps_added_by_clustering,
            'common_BPs_extended_by_clustering': _common_bps_extended_by_clustering,

            'all_common_BP_clusters': _all_common_bp_clusters,
            'num_common_BP_clusters': _num_common_bp_clusters,
            'max_strains_with_common_BP_clusters': _max_strains_w_common_bp_clusters,
            'all_common_BP_clusters_of_max_strains': _all_common_bp_clusters_of_max_strains,
            'max_common_BP_clusters': _max_common_bp_clusters
        }
        return rec
    
    def _calc_common_bps(self, all_bps: Dict[str, list]) -> Tuple[Dict[tuple, list], Dict[tuple, int], int, int]:
        """_summary_

        Args:
            all_bps (Dict[str, list]): mapping of sRNA ('{strain}__{srna_id}') to all its unique BPs
        Returns:
            Dict[tuple, list]: 
            Dict[tuple, int]: 
            int:
            int: 
        """
        all_common_bps, num_common_bps = {}, {}
        max_common_bps = 0
        max_strains_w_common_bps = 0

        for size in range(2, len(all_bps.keys()) + 1):
            for rna_comb in itertools.combinations(all_bps.keys(), size):
                rnas_bps = list(map(all_bps.get, rna_comb))
                # 1 - common BPs of rnas
                common_bps: List[str] = rnas_bps[0]
                for l in rnas_bps[1:]:
                    common_bps: List[str] = self.U.get_common_bps(common_bps, l) 
                if common_bps:
                    all_common_bps[rna_comb] = common_bps
                    # 2 - num common BPs
                    num_common_bps[rna_comb] = len(common_bps)
                    # 3 - max strains with common BPs
                    rnas_strains = set([rna.split('__')[0] for rna in rna_comb])
                    max_strains_w_common_bps = max(max_strains_w_common_bps, len(rnas_strains))
                    # 4 - max common BPs
                    max_common_bps = max(max_common_bps, len(common_bps))
        
        all_common_bps_of_max_strains = {}
        for rna_comb, bp_lst in all_common_bps.items():
            # complete BP info in all common bps 
            all_common_bps[rna_comb] = [f"{bp}__{self.G.nodes[bp]['lbl'].replace(" ", "_")}" for bp in sorted(bp_lst)]
            # get all common BPs of max strains
            rnas_strains = set([rna.split('__')[0] for rna in rna_comb])
            if len(rnas_strains) == max_strains_w_common_bps:
                all_common_bps_of_max_strains[rna_comb] = all_common_bps[rna_comb]
        
        return all_common_bps, num_common_bps, max_strains_w_common_bps, all_common_bps_of_max_strains, max_common_bps
    
    def _cluster_homologs(self, rna_str: str, rna_type: str) -> pd.DataFrame:
        self.logger.info(f"Cluster and analyze {rna_str} homologs")
        # 1 - get all homologs clusters
        all_homologs_clusters = set()
        for strain in self.U.strains:
            homologs_clusters = self._get_homologs_clusters_of_strain(rna_type, strain)
            all_homologs_clusters = all_homologs_clusters.union(homologs_clusters)
        # 2 - validate and dump
        all_homologs_df = self._validate_homologs_clusters(rna_type, all_homologs_clusters)
        write_df(all_homologs_df, join(self.out_path_clustering_homologs, f"{rna_str}_homologs__{self.out_file_suffix}.csv"))
    
    def _cluster_paralogs_only(self, rna_str: str, rna_type: str):
        self.logger.info(f"Cluster and analyze {rna_str} paralogs only")
        
        for strain in self.U.strains:
            # 1 - get paralogs only clusters per strain
            paralogs_clusters = self._get_paralog_only_clusters_of_strain(rna_type, strain)
            if not paralogs_clusters:
                self.logger.debug(f"No {rna_type} paralog clusters for {strain}")
                continue
            # 2 - validate and dump 
            paralogs_df = self._validate_paralogs_only_clusters(rna_type, paralogs_clusters, strain)
            write_df(paralogs_df, join(self.out_path_clustering_paralogs_only, f"{strain}_{rna_type}_paralogs__{self.out_file_suffix}.csv"))
    
    def _get_homologs_clusters_of_strain(self, rna_type: str, strain: str) -> Set[Tuple[str, str]]:
        rna_nodes =  [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == strain]
        # 1 - all homologs clusters
        all_homologs_clusters = set()
        clusters_items = []
        for rna in rna_nodes:
            if rna not in clusters_items:
                cluster = set()
                # the recursive function bellow may return a cluster with both orthologs and paralogs.
                cluster = self.U.get_orthologs_cluster(self.G, rna, cluster)  
                if cluster:
                    all_homologs_clusters.add(tuple(sorted(cluster)))
                    clusters_items = clusters_items + sorted(cluster)
        assert set(rna_nodes) <= set(clusters_items), "some RNA nodes are missing in the clusters"
        
        # 2 - homologs clusters with size > 1
        homologs_clusters = {c for c in all_homologs_clusters if len(c) > 1}
        if not homologs_clusters:
            self.logger.warning(f"#### no {rna_type} orthologs clusters for {strain}")
        
        return homologs_clusters
    
    def _get_paralog_only_clusters_of_strain(self, rna_type: str, strain: str) -> Set[Tuple[str, str]]:
        rna_nodes =  [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == strain]
        # 1 - all paralogs clusters
        all_clusters = set()
        clusters_items = []
        for rna in rna_nodes:
            if rna not in clusters_items:
                cluster = set()
                # recursive function returns a cluster with paralogs only.
                cluster = self.U.get_paralogs_only_cluster(self.G, strain, rna, cluster)  
                if cluster:
                    all_clusters.add(tuple(sorted(cluster)))
                    clusters_items = clusters_items + sorted(cluster)
        assert set(rna_nodes) <= set(clusters_items), "some RNA nodes are missing in the clusters"
        
        # 2 - paralogs clusters with size > 1
        paralogs_clusters = {c for c in all_clusters if len(c) > 1}
        if not paralogs_clusters:
            self.logger.warning(f"#### no {rna_type} paralogs clusters for {strain}")
        
        return paralogs_clusters
    
    def _validate_homologs_clusters(self, rna_type: str, homologs_clusters: Set[Tuple[str]]) -> pd.DataFrame:
        # 1 - general validation
        nodes = [item for tpl in homologs_clusters for item in tpl]
        assert len(set(nodes)) == len(nodes), f"some {rna_type} nodes are duplicated in the orthologs clusters"
        # 2 - per cluster validation
        records = []
        for cluster in homologs_clusters:
            # 2.1 - get strains, paralog pairs and ortholog pairs
            strains = set()
            paralog_pairs, paralog_pairs_w_meta = list(), list()
            ortholog_pairs, ortholog_pairs_w_meta = list(), list()

            for n1, n2 in itertools.combinations(list(cluster), 2):
                s1, s2 = self.G.nodes[n1]['strain'], self.G.nodes[n2]['strain']
                strains = strains.union({s1, s2})
                if s1 == s2 and self.U.are_paralogs(self.G, n1, n2, s1):
                    paralog_pairs.append((n1, n2))
                    paralog_pairs_w_meta.append((f"{s1}__{n1}__{self.G.nodes[n1]['name']}", f"{s2}__{n2}__{self.G.nodes[n2]['name']}"))
                elif (self.U.are_orthologs_by_seq(self.G, n1, n2, s1, s2) or self.U.are_orthologs_by_name(self.G, n1, n2, s1, s2)):
                    ortholog_pairs.append((n1, n2))
                    ortholog_pairs_w_meta.append((f"{s1}__{n1}__{self.G.nodes[n1]['name']}", f"{s2}__{n2}__{self.G.nodes[n2]['name']}"))
                else:
                    continue
            # 2.2 - validate
            assert set(cluster) == set([n for tpl in ortholog_pairs for n in tpl] + [n for tpl in paralog_pairs for n in tpl]), f"misalignment between {rna_type} homologs cluster VS orthologs/paralogs pairs"
            # 2.3 - add record
            records.append({
                'cluster': cluster,
                'cluster_size': len(cluster),
                'strains': tuple(sorted(strains)),
                'num_strains': len(strains),
                'ortholog_pairs': sorted(ortholog_pairs_w_meta),
                'num_ortholog_pairs': len(ortholog_pairs_w_meta),
                'paralog_pairs': sorted(paralog_pairs_w_meta),
                'num_paralog_pairs': len(paralog_pairs_w_meta)
            })
        homologs_df = pd.DataFrame(records)
        return homologs_df

    def _validate_paralogs_only_clusters(self, rna_type: str, paralogs_clusters: Set[Tuple[str]], strain: str) -> pd.DataFrame:
        # 1 - general validation
        nodes = [item for tpl in paralogs_clusters for item in tpl]
        assert len(set(nodes)) == len(nodes), f"some {rna_type} nodes are duplicated in the paralogs clusters"
        # 2 - per cluster validation
        records = []
        for cluster in paralogs_clusters:
            # 2.1 - validate strains
            n_to_strain = dict(zip(list(cluster), [self.G.nodes[n]['strain'] for n in cluster]))
            assert (len(set(n_to_strain.values())) == 1) and (list(n_to_strain.values())[0] == strain), f"wrong cluster of {strain}"
            cluster_w_meta = tuple([f"{strain}__{n}__{self.G.nodes[n]['name']}" for n in cluster])
            # 2.2 - add record
            records.append({
                'cluster': cluster,
                'cluster_w_meta': cluster_w_meta,
                'cluster_size': len(cluster),
            })
        paralogs_only_df = pd.DataFrame(records)
        return paralogs_only_df
    
    def _convert_strain_names(self, tuples_of_names: List[tuple]) -> List[tuple]:
        new_tuples_of_names = [tuple([self.U.get_short_strain_nm(nm) for nm in ast.literal_eval(tpl)]) for tpl in tuples_of_names]
        return new_tuples_of_names

    def _dump_stats_rna_homolog_clusters_size(self, val_type: str, min_val_limit: float = None):
        """
        Args:
            val_type (str): 'ratio' or 'percentage'
            min_val_limit (float)
        """
        records = []
        for (rna_type, all_homologs_df) in [(self.U.srna, self.srna_homologs.copy()), (self.U.mrna, self.mrna_homologs.copy())]:
            # 1 - num_clusters
            num_clusters = len(all_homologs_df)

            # 2 - size
            # 2.1 - get distributoin values per size
            unq, counts = np.unique(all_homologs_df['cluster_size'], return_counts=True)
            vals = [convert_count_to_val(count, num_clusters, 'ratio') for count in counts]
            
            # 2.2 - list of tuples (cluster_size , val) **sorted by cluster_size**
            size_to_val = dict(zip(unq, vals))
            cluster_sizes_vals = [(size, size_to_val.get(size, 0)) for size in range(2, max(unq) + 1)]
            self.logger.debug(f"{rna_type} No. Clusters = {num_clusters}, cluster_sizes_vals ({len(cluster_sizes_vals)}): {cluster_sizes_vals}")
            # 2.3 - limit vals
            if min_val_limit:
                cluster_sizes_vals = [(x, val) for (x, val) in cluster_sizes_vals if val >= min_val_limit]
                self.logger.debug(f"{rna_type} cluster_sizes_vals AFTER limit (val >= {min_val_limit}) ({len(cluster_sizes_vals)}): {cluster_sizes_vals}")
            cluster_sizes_vals_str = str(cluster_sizes_vals).replace(", ", ",").replace("),", ") ").replace("[", "{").replace("]", "}")
            # 2.4 - get sizes
            cluster_sizes = [x[0] for x in cluster_sizes_vals]
            cluster_sizes_str = str(cluster_sizes).replace(", ", ",").replace("[", "{").replace("]", "}")
            # 2.5 - y max (maximal val)
            y_max_val = max(vals)

            rec = {
                "rna_type": rna_type,
                "num_clusters": num_clusters,
                "symbolic_x_coords_cluster_sizes": cluster_sizes_str,
                f"coordinates_cluster_sizes_{val_type}s": cluster_sizes_vals_str,
                "num_coordinates": len(cluster_sizes),
                f"y_max_{val_type}": y_max_val
            }
            records.append(rec)

        df = pd.DataFrame(records)
        write_df(df, join(self.out_path_rna_homologs_multi_strains, f"cluster_sizes_{val_type}_{min_val_limit}__{self.out_file_suffix}.csv"))
    
    def _dump_stats_rna_homolog_clusters_strains_composition(self, val_type: str, min_val_limit: float = None, add_textit: bool = False):
        """
        Args:
            val_type (str): 'ratio' or 'percentage'
            min_val_limit (float)
        """
        records = []
        for (rna_type, all_homologs_df) in [(self.U.srna, self.srna_homologs.copy()), (self.U.mrna, self.mrna_homologs.copy())]:
            # 1 - num_clusters
            num_clusters = len(all_homologs_df)

            # 2 - strains composition
            # 2.1 - get distributoin values per strains composition
            unq, counts = np.unique(all_homologs_df['strains'], return_counts=True)
            unq = self._convert_strain_names(list(unq))
            vals = [convert_count_to_val(count, num_clusters, val_type, ndigits=3) for count in counts]

            # 2.2 - list of tuples (strains_composition , val) **sorted by val** in descending order  
            cluster_compositions_vals = sorted(list(zip(unq, vals)), key=lambda x: x[1], reverse=True)
            self.logger.debug(f"{rna_type} cluster_compositions_vals ({len(cluster_compositions_vals)})")
            # 2.3 - limit vals
            if min_val_limit:
                cluster_compositions_vals = [(x, val) for (x, val) in cluster_compositions_vals if val > min_val_limit]
                self.logger.debug(f"{rna_type} cluster_compositions_vals AFTER limit (val >= {min_val_limit}) ({len(cluster_compositions_vals)})")
            cluster_compositions_vals_str = str(cluster_compositions_vals).replace("('", "{(").replace("'), ", ")},").replace("', '", ", ").replace("), (", ") (").replace("[", "{").replace("]", "}")
            # 2.4 - get compositions
            cluster_compositions = [x[0] for x in cluster_compositions_vals]
            cluster_compositions_str = str(cluster_compositions).replace("('", "{(").replace("'), ", ")},").replace("', '", ", ").replace("[", "{").replace("')]", ")}}")
            # 2.5 - add LateX "\textit" command
            if add_textit:
                for n in list(self.U.strain_nm_to_short.values()):
                    cluster_compositions_vals_str = cluster_compositions_vals_str.replace(f"{n}", "\textit{" f"{n}" + "}")
                    cluster_compositions_str = cluster_compositions_str.replace(f"{n}", "\textit{" f"{n}" + "}")
            # 2.6 - y max (maximal val)
            y_max_val = max(vals)
            
            rec = {
                "rna_type": rna_type,
                "num_clusters": num_clusters,
                "symbolic_x_coords_cluster_strains_compositions": cluster_compositions_str,
                f"coordinates_cluster_strains_compositions_{val_type}s": cluster_compositions_vals_str,
                "num_coordinates": len(cluster_compositions),
                f"y_max_{val_type}": y_max_val
            }
            records.append(rec)
        
        df = pd.DataFrame(records)
        write_df(df, join(self.out_path_rna_homologs_multi_strains, f"cluster_compositions_{val_type}_{min_val_limit}__{self.out_file_suffix}.csv"))
    
    def _dump_stats_rna_orthologs_n_paralogs_per_strain(self, val_type: str):
        """
        Args:
            val_type (str): 'ratio' or 'percentage'
        """
        # -----------------   sRNA, mRNA rows   -----------------
        _map = {}
        records = []
        for (rna_type, all_homologs_df) in [(self.U.srna, self.srna_homologs.copy()), (self.U.mrna, self.mrna_homologs.copy())]:
            _map[rna_type] = {}

            # 1 - num RNAs per strain
            strain_2_num_rnas = {}
            for strain in self.U.strains:
                num_rnas = len([n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == strain])
                strain_2_num_rnas[strain] = num_rnas
            _map[rna_type]['strain_2_num_rnas'] = strain_2_num_rnas

            # 2 - orthologs val per strain
            clusters = [ast.literal_eval(c) for c in all_homologs_df['cluster'] ]
            unq, counts = np.unique([self.G.nodes[rna]['strain'] for cls in clusters for rna in cls], return_counts=True)
            strain_2_orthologs_val = {}
            for (strain, o_count) in list(zip(unq, counts)):
                num_rnas = strain_2_num_rnas[strain]
                val = convert_count_to_val(o_count, num_rnas, val_type)
                strain_2_orthologs_val[strain] = val
            _map[rna_type]['strain_2_orthologs_val'] = strain_2_orthologs_val

            # 3 - paralogs val per strain
            strain_2_paralogs_val = {}
            for strain in self.U.strains:
                file_path = join(self.out_path_clustering_paralogs_only, f"{strain}_{rna_type}_paralogs__{self.out_file_suffix}.csv")
                if not os.path.exists(file_path):
                    self.logger.info(f"#### no {rna_type} paralogs clusters for {strain}")
                    continue
                # paralogs count
                paralogs_df = read_df(file_path)
                clusters = [ast.literal_eval(c) for c in paralogs_df['cluster']]
                p_count = sum([len(cls) for cls in clusters])
                # paralogs val
                num_rnas = strain_2_num_rnas[strain]
                val = convert_count_to_val(p_count, num_rnas, val_type)
                strain_2_paralogs_val[strain] = val
            _map[rna_type]['strain_2_paralogs_val'] = strain_2_paralogs_val

            # 4 - adjust output record
            rec = {"key": rna_type}
            max_val = 0
            for strain_nm, short in self.U.strain_nm_to_short.items():
                orthologs_val = strain_2_orthologs_val.get(strain_nm, 0)
                paralogs_val = strain_2_paralogs_val.get(strain_nm, 0)
                rec[short] = "{(O," + f"{orthologs_val}" + ") (P," + f"{paralogs_val}" +")}"
                max_val = max(max_val, orthologs_val, paralogs_val)
            rec['max_val'] = max_val
            records.append(rec)
        df1 = pd.DataFrame(records)
        
        # -----------------   Orthologs, Paralogs rows   -----------------
        records = []
        for k in ["orthologs", "paralogs"]:
            rec = {"key": k}
            max_val = 0
            for strain_nm, short in self.U.strain_nm_to_short.items():
                srna_val = _map[self.U.srna][f'strain_2_{k}_val'].get(strain_nm, 0)
                mrna_val = _map[self.U.mrna][f'strain_2_{k}_val'].get(strain_nm, 0)
                rec[short] = "{(s," + f"{srna_val}" + ") (m," + f"{mrna_val}" +")}"
                max_val = max(max_val, srna_val, mrna_val)
            rec['max_val'] = max_val
            records.append(rec)
        df2 = pd.DataFrame(records)

        df = pd.concat([df1, df2], ignore_index=True)
        write_df(df, join(self.out_path_rna_homologs_multi_strains, f"rna_homologs_per_strain_{val_type}__{self.out_file_suffix}.csv"))

    def _log_mrna_without_bp(self, strain: str, unq_targets_without_bp: Set[str]):
            ips_annot = self.ips_go_annotations.get(strain, None)
            if ips_annot is not None:
                mrna_w_mf = set(ips_annot[pd.notnull(ips_annot['MF_go_xrefs'])]['mRNA_accession_id'])
                mrna_w_cc = set(ips_annot[pd.notnull(ips_annot['CC_go_xrefs'])]['mRNA_accession_id'])
                mf_count = np.intersect1d(unq_targets_without_bp, mrna_w_mf).size
                cc_count = np.intersect1d(unq_targets_without_bp, mrna_w_cc).size
            self.logger.info(f"Strain: {strain}, Number of unique mRNAs targets without BPs: {len(unq_targets_without_bp)} (where {mf_count} have ips MF annotation, {cc_count} have ips CC annotation)")

    def _log_srna_bp_mapping(self, mapping: dict):
        self.logger.info(f"--------- sRNA to BP mapping")
        for strain, srna_bp_mapping in mapping.items():
            srna_count = len(srna_bp_mapping)
            mrna_targets = [mrna for srna_targets in srna_bp_mapping.values() for mrna in srna_targets.keys()]
            unique_mrna_targets = set(mrna_targets)
            bp_list = [bp for srna_targets in srna_bp_mapping.values() for bps in srna_targets.values() for bp in bps]
            unique_bps = set(bp_list)

            self.logger.info(
                f"Strain: {strain} \n"
                f"  Number of sRNA keys: {srna_count} \n"
                f"  Number of unique mRNA targets with BPs: {len(unique_mrna_targets)} \n"
                f"  Number of mRNA-BP annotations (edges): {len(bp_list)} \n"
                f"  Number of unique BPs: {len(unique_bps)} \n"
                f"  & {srna_count} & {len(unique_mrna_targets)} & {len(bp_list)} & {len(unique_bps)}"
            )
    
    def _find_homologs(self, strain_to_rna_list: Dict[str, List[str]], rna_str: str) -> List[Tuple[str]]:
        """
        For each cluster in homologs_df['cluster'], find which RNAs from strain_to_rna_list are homologs, i.e., belong to the same cluster.

        Args:
            strain_to_rna_list (dict): A dictionary in the following format:
            {
                <strain_id>: [<RNA_id1>, <RNA_id2>, ...],
                ...
            }
        """
        homologs_df = self.srna_homologs if rna_str == 'sRNA' else self.mrna_homologs
        # Flatten all RNAs from strain_to_rna_list
        all_rnas = set()
        for rna_list in strain_to_rna_list.values():
            all_rnas.update(rna_list)
        # Iterate over clusters
        all_rna_homologs = []
        for cluster in homologs_df['cluster']:
            cluster = cluster if type(cluster) == tuple else ast.literal_eval(cluster)
            # Find intersection with all_rnas
            rna_orthologs = tuple(sorted(set(cluster).intersection(all_rnas)))
            if len(rna_orthologs) > 1:
                all_rna_homologs.append(rna_orthologs)
        return sorted(set(all_rna_homologs))
    
    def _get_orthologs_clusters(self, rna_list: List[str], rna_str: str) -> List[Tuple[str]]:
        """
        for each RNA in rna_list, find its orthologs cluster from homologs_df['cluster'].

        Args:
            rna_list (list): [<RNA_id1>, <RNA_id2>, ...]
        """

        homologs_df = self.srna_homologs if rna_str == 'sRNA' else self.mrna_homologs

        orthologs_clusters = []
        for rna in rna_list:
            # find the cluster that contains this RNA
            for cluster in homologs_df['cluster']:
                cluster = cluster if type(cluster) == tuple else ast.literal_eval(cluster)
                if rna in cluster:
                    orthologs_clusters.append(tuple(sorted(cluster)))
                    break

        return sorted(set(orthologs_clusters))
    
    def _find_rnas_with_max_orthologs(self, rna_str: str, max_orthologs: int = 0) -> List[str]:
        """
        For each cluster in orthologs_df['cluster'], find which RNAs from strain_to_rna_list are othologs, i.e., belong to the same cluster.
        Return a set of tuples, each tuple contains the RNAs from strain_to_rna_list that belong to the same cluster.

        Args:
            strain_to_rna_list (dict): A dictionary in the following format:
            {
                <strain_id>: [<RNA_id1>, <RNA_id2>, ...],
                ...
            }
        """
        if rna_str == 'sRNA':
            rna_type = self.U.srna
            orthologs_df = self.srna_homologs
        else:
            rna_type = self.U.mrna
            orthologs_df = self.mrna_homologs
        
        all_rnas = set([n for n, d in self.G.nodes(data=True) if d.get('type') == rna_type])
        # RNAs that have orthologs (above max_orthologs threshold)
        mask_invalid_rnas = orthologs_df['num_strains'] > 1 + max_orthologs
        invalid_rnas = orthologs_df[mask_invalid_rnas]['cluster'].apply(ast.literal_eval).tolist()
        invalid_rnas = set([rna for tpl in invalid_rnas for rna in tpl])

        rnas_with_max_orthologs = sorted(all_rnas - invalid_rnas)
        self.logger.info(f"out of {len(all_rnas)} {rna_str}s, {len(rnas_with_max_orthologs)} have num orthologs <= {max_orthologs}")

        return rnas_with_max_orthologs
    
    def _get_targets_of_focus_sRNAs(self, related_mrnas_and_srnas: Dict[str, Dict[str, List[str]]], focus_srnas: List[str]) -> List[str]:
        targets_of_focus_srnas = set()
        for mrna_to_srnas in related_mrnas_and_srnas.values():
            for mrna, srnas in mrna_to_srnas.items():
                if set(srnas).intersection(focus_srnas):
                    targets_of_focus_srnas.add(mrna)
        return sorted(targets_of_focus_srnas)
    
    def _get_strain_to_mrna_to_focus_srnas(self, strain_dict_series, all_focus_srnas: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], List[str], List[str]]:
        lst_strain_to_mrna_to_focus_srnas = []
        lst_focus_srnas_flat = []
        lst_targets_of_focus_srnas_flat = []
        for strain_dict in strain_dict_series:
            strain_to_mrna_to_focus_srnas = {}
            focus_srnas_flat = []
            targets_of_focus_srnas_flat = []
            
            for strain, mrna_to_srnas in strain_dict.items():
                strain_to_mrna_to_focus_srnas[strain] = {}
                for mrna, srnas in mrna_to_srnas.items():
                    focus_srnas = sorted(set(srnas).intersection(all_focus_srnas))
                    if len(focus_srnas) > 0:
                        focus_srnas = [f"{srna}__{self.G.nodes[srna]['name']}" for srna in focus_srnas]
                        strain_to_mrna_to_focus_srnas[strain][f"{mrna}__{self.G.nodes[mrna]['name']}"] = focus_srnas
                        focus_srnas_flat.extend(focus_srnas)
                        targets_of_focus_srnas_flat.append(mrna)  # use only mrna id
            
            strain_to_mrna_to_focus_srnas = {strain: v for strain, v in strain_to_mrna_to_focus_srnas.items() if v}  # remove strains with empty dict
            focus_srnas_flat = sorted(set(focus_srnas_flat))
            targets_of_focus_srnas_flat = sorted(set(targets_of_focus_srnas_flat))

            lst_strain_to_mrna_to_focus_srnas.append(strain_to_mrna_to_focus_srnas)
            lst_focus_srnas_flat.append(focus_srnas_flat)
            lst_targets_of_focus_srnas_flat.append(targets_of_focus_srnas_flat)

        return lst_strain_to_mrna_to_focus_srnas, lst_focus_srnas_flat, lst_targets_of_focus_srnas_flat

    def _analysis_2_bp_rna_mapping(self, bp_rna_mapping: dict):
        """
        Generate a DataFrame with information about BPs and their related mRNAs and sRNAs

        Args:
            bp_rna_mapping (dict): A dictionary in the following format:
            {
                <bp_id>: {
                        <strain_id>: {
                            <mRNA_target_id>: [<sRNA_id1>, <sRNA_id2>, ...],
                            ...
                        },
                        ...
                },
                ...
            }   
        """
        self.logger.info(f"##############   Analsis 2 - Cross-Species Conservation of Biological Processes   ##############")
        # 1 - generate df
        records = []
        for bp_id, strain_dict in bp_rna_mapping.items():
            bp_lbl = self.G.nodes[bp_id]['lbl']
            bp_definition = self.G.nodes[bp_id]['meta']['definition']['val']
            
            strains = sorted(strain_dict.keys())
            num_strains = len(strains)
            related_mRNAs_and_sRNAs_complete = {}
            related_mRNAs = {}
            num_related_mRNAs = {}
            related_sRNAs = {}
            num_related_sRNAs = {}
            for strain, mrna_to_srnas in strain_dict.items():
                # related_mRNAs_and_sRNAs
                if not strain in related_mRNAs_and_sRNAs_complete:
                    related_mRNAs_and_sRNAs_complete[strain] = {}
                for mrna, srnas in mrna_to_srnas.items():
                    related_mRNAs_and_sRNAs_complete[strain][f"{mrna}__{self.G.nodes[mrna]['name']}"] = [f"{srna}__{self.G.nodes[srna]['name']}" for srna in srnas]
                # related mRNAs
                related_mRNAs[strain] = sorted(mrna_to_srnas.keys())
                num_related_mRNAs[strain] = len(mrna_to_srnas)
                #   flatten sRNA lists for all mRNAs in this strain
                srna_set = set()
                for srna_list in mrna_to_srnas.values():
                    srna_set.update(srna_list)
                # related mRNAs
                related_sRNAs[strain] = sorted(srna_set)
                num_related_sRNAs[strain] = len(srna_set)
            records.append({
                'bp_id': bp_id,
                'bp_lbl': bp_lbl,
                'bp_definition': bp_definition,
                'strains': strains,
                'num_strains': num_strains,
                'strain_dict': strain_dict,
                'related_mRNAs_and_sRNAs': related_mRNAs_and_sRNAs_complete,
                'related_mRNAs': related_mRNAs,
                'num_related_mRNAs': num_related_mRNAs,
                'related_sRNAs': related_sRNAs,
                'num_related_sRNAs': num_related_sRNAs
            })
        df = pd.DataFrame(records)

        # 2 - identify orthologs
        for rna_str in ['sRNA', 'mRNA']:
            df[f'related_{rna_str}_orthologs'] = list(map(self._find_homologs, df[f'related_{rna_str}s'], np.repeat(rna_str, len(df))))

        # 3 - rank and add info
        # 3.1 - all Focus sRNAs: find all sRNAs in G that have num orthologs <= 0
        srna_max_orthologs = 0
        all_focus_srnas = self._find_rnas_with_max_orthologs('sRNA', max_orthologs=srna_max_orthologs)
        # 3.2 - BP tree for focus sRNAs
        strain_to_mRNA_to_focus_srnas, focus_srnas, targets_of_focus_srnas = self._get_strain_to_mrna_to_focus_srnas(df['strain_dict'], all_focus_srnas)
        df['strain_to_mRNA_to_focus_sRNAs'] = strain_to_mRNA_to_focus_srnas  # map of strains to their mRNAs to focus sRNAs
        # 3.3 - Focus sRNAs
        df[f'focus_sRNAs_{srna_max_orthologs}_orthologs'] = focus_srnas
        df['num_focus_sRNAs'] = df[f'focus_sRNAs_{srna_max_orthologs}_orthologs'].apply(lambda x: len(x))  #   --- second criterion for ranking
        df['num_strains_w_focus_sRNAs'] = df['strain_to_mRNA_to_focus_sRNAs'].apply(lambda x: len(x))      #   --- first criterion for ranking
        # 3.4 - mRNA targets of focus sRNAs
        df['targets_of_focus_sRNAs'] = targets_of_focus_srnas
        df['complete_ortholog_clusters_of_targets'] = list(map(self._get_orthologs_clusters, df['targets_of_focus_sRNAs'], np.repeat('mRNA', len(df))))
        df['filtered_ortholog_clusters_of_targets'] = list(map(lambda clusters_lst, targets_lst: [tuple(set(tpl).intersection(targets_lst)) for tpl in clusters_lst if len(set(tpl).intersection(targets_lst)) > 1], df['complete_ortholog_clusters_of_targets'], df['targets_of_focus_sRNAs']))
        df['num_filtered_ortholog_clusters'] = df['filtered_ortholog_clusters_of_targets'].apply(lambda x: len(x))
        df['strains_of_filtered_ortholog_clusters'] = df['filtered_ortholog_clusters_of_targets'].apply(lambda x: sorted(set([self.G.nodes[rna]['strain'] for tpl in x for rna in tpl])))
        df['num_strains_of_filtered_ortholog_clusters'] = df['strains_of_filtered_ortholog_clusters'].apply(lambda x: len(x))
        # 3.4 - score
        max_num_focus_srnas = df['num_focus_sRNAs'].max() if df['num_focus_sRNAs'].max() > 0 else 1
        df['score'] = 100 * df['num_strains_w_focus_sRNAs'] + 10 * (df['num_focus_sRNAs'] / max_num_focus_srnas)
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

        # 4 - complete info
        for rna_str in ['sRNA', 'mRNA']:
            df[f'related_{rna_str}s'] = df[f'related_{rna_str}s'].apply(lambda x: {strain: [f"{rna}__{self.G.nodes[rna]['name']}" for rna in rnas] for strain, rnas in x.items()})
            df[f'related_{rna_str}_orthologs'] = df[f'related_{rna_str}_orthologs'].apply(lambda x: [tuple(f"{rna}__{self.G.nodes[rna]['name']}" for rna in rnas) for rnas in x])
        df['targets_of_focus_sRNAs'] = df['targets_of_focus_sRNAs'].apply(lambda x: [f"{mrna}__{self.G.nodes[mrna]['name']}" for mrna in x])
        for col in ['complete_ortholog_clusters_of_targets', 'filtered_ortholog_clusters_of_targets']:
            df[col] = df[col].apply(lambda x: [tuple(f"{rna}__{self.G.nodes[rna]['name']}" for rna in tpl) for tpl in x])

        # 5 - remove temp columns
        df = df.drop(columns=['strain_dict'])

        # 6 - dump results
        # 6.1 - csv
        write_df(df, join(self.out_path_analysis_tool_2, f"Tool_2__BP_of_focus_sRNAs_{srna_max_orthologs}_orthologs__{self.out_file_suffix}.csv"))
        # 6.2 - manual txt
        bps_for_txt = [6355, 55085]
        dump_manual_txt = True
        if dump_manual_txt:
            for _, row in df.iterrows():
                bp_id = int(row['bp_id'])
                if bp_id in bps_for_txt:
                    with open(join(self.out_path_analysis_tool_2, f"Tool_2__Manual_BP_{bp_id}__{self.graph_version}.txt"), 'w', encoding='utf-8') as f:
                        for col in df.columns:
                            f.write(f"{col}\n{row[col]}\n\n")         

        # 7 - log statistics
        self.logger.info(f"--------- BP to RNAs mapping\n{df.head()}")
        # self.logger.info(
        #     f"Strain: {strain} \n"
        #     f"  Number of sRNA keys: {srna_count} \n"
        #     f"  Number of unique mRNA targets with BPs: {len(unique_mrna_targets)} \n"
        #     f"  Number of BP annotations: {len(bp_list)} \n"
        #     f"  Number of unique BPs: {len(unique_bps)}"
        #     )
        # Optionally, dump to file if needed
        # out_path = self.config['analysis_output_dir']
        # df.to_csv(os.path.join(out_path, "bp_to_rnas_mapping.csv"), index=False)
    
    def _dump_bps_of_annotated_mrnas(self):
        # 1 - Find all BP nodes that are annotated to at least one mRNA
        bp_nodes_with_annotation = [
            node for node, data in self.G.nodes(data=True)
            if data.get('type') == self.U.bp and any(
                self.G.has_edge(mrna, node) and
                any(
                    edge_data.get('type') == self.U.annotated
                    for edge_data in self.G[mrna][node].values()
                )
                for mrna in self.G.predecessors(node)
            )
        ]
        # 2 - Create a DataFrame with the BP IDs, labels, and definitions
        records = []
        for bp in sorted(bp_nodes_with_annotation):
            records.append({'bp_id': bp, 'lbl': self.G.nodes[bp]['lbl'], 'definition': self.G.nodes[bp]['meta']['definition']['val']})
        bps_of_annotated_mrnas = pd.DataFrame(records)
        # 3 - Dump the DataFrame to a CSV file
        write_df(bps_of_annotated_mrnas, join(self.out_path_summary_tables, f"BPs_of_annotated_mrnas__{self.out_file_suffix}.csv"))
        return
    
    def _dump_metadata(self, metadata: dict):
        """
        Dump metadata to a CSV file per strain.

        Args:
            metadata (dict): A dictionary in the following format:
            {
                <strain_id>: {
                        <sRNA_id>: {
                            <BP_id>: {                            
                                'BP_id': bp,
                                'BP_lbl': bp_lbl,
                                'M': M,
                                'n': n,
                                'N': N,
                                'k': k,
                                'k_targets_associated_w_BP': targets,
                                'p_value': p_value,
                                'adj_p_value': adj_p_value
                            },
                            ...
                        },
                        ...
                },
                ...
            }  
        """
        self.logger.info(f"Dumping enrichment metadata...")
        for strain, srna_data in metadata.items():
            rows = []
            for srna_id, bp_data in srna_data.items():
                srna_name = self.G.nodes[srna_id].get('name', 'N/A')
                for bp_id, meta in bp_data.items():
                    rows.append({
                        'strain': strain,
                        'sRNA_id': srna_id,
                        'sRNA_name': srna_name,
                        'BP_id': meta['BP_id'],
                        'BP_lbl': meta['BP_lbl'],
                        'M': meta['M'],
                        'n': meta['n'],
                        'N': meta['N'],
                        'k': meta['k'],
                        'k_targets_associated_w_BP': ";".join(meta['k_targets_associated_w_BP']),
                        'p_value': meta['p_value'],
                        'adj_p_value': meta['adj_p_value']
                    })

            df = pd.DataFrame(rows)
            df.to_csv(join(self.out_path_enrichment, f"metadata_per_srna_{strain}.csv"), index=False)
    
    def compare_bp_to_accociated_mrnas(self, bp_to_accociated_mrnas, bp_to_accociated_mrnas_g):
        # Check if keys are the same
        keys1 = set(bp_to_accociated_mrnas.keys())
        keys2 = set(bp_to_accociated_mrnas_g.keys())
        
        if keys1 != keys2:
            missing_in_g = keys1 - keys2
            missing_in_original = keys2 - keys1
            self.logger.info(f"Keys missing in bp_to_accociated_mrnas_g: {missing_in_g}")
            self.logger.info(f"Keys missing in bp_to_accociated_mrnas: {missing_in_original}")
        
        # Compare values for each key
        for key in keys1.intersection(keys2):
            value1 = set(bp_to_accociated_mrnas[key])
            value2 = set(bp_to_accociated_mrnas_g[key])
            
            if value1 != value2:
                self.logger.info(f"Difference for key {key}:")
                self.logger.info(f"  In bp_to_accociated_mrnas: {value1 - value2}")
                self.logger.info(f"  In bp_to_accociated_mrnas_g: {value2 - value1}")
    
    def _apply_enrichment(self, mapping: dict) -> Tuple[dict, dict]:
        """
        Enrichment (per strain): 
            per sRNA, find and keep only significant biological processes (BPs) that its targets invovlved in.
            significant BPs are found using a hypergeometric test.

        Args:
            mapping (dict): A dictionary in the following format:
            {
                <strain_id>: {
                        <sRNA_id>: {
                            <mRNA_target_id>: [<bp_id1>, <bp_id2>, ...],
                            ...
                        },
                        ...
                },
                ...
            }   
        
        Returns:
            filtered_mapping (dict): A dictionary in the same format post filtering of insignificant BPs.
            metadata (dict): A dictionary in the following format:
            {
                <strain_id>: {
                        <sRNA_id>: {
                            <BP_id>: {                            
                                'BP_id': bp,
                                'BP_lbl': self.G.nodes[bp]['lbl'],
                                'M': M,
                                'n': n,
                                'N': N,
                                'k': k,
                                'k_targets_associated_w_BP': targets,
                                'p_value': p_value,
                                'adj_p_value': adj_p_value
                            },
                            ...
                        },
                        ...
                },
                ...
            }

            where:
            parameters of the hypergeometric test are:
                M (int): Total number of elements in the population (i.e., all targets).
                n (int): Number of marked elements in the population (i.e. genes of the population associated to this biological process).
                N (int): Size of the selection (i.e., selection = all targets of a specific sRNA).
                k (int): Number of marked elements in the selection (i.e. genes of the selection that are associated to this biological process).
            k_targets_associated_w_BP (list): the target genes of the selection that are associated to this biological process
            p_value (float):  the p-value of the hypergeometric test
            adj_p_value (float): the adjusted p-value of the hypergeometric test (after multiple testing correction)
        """
        self.logger.info(f"applying enrichment {'+ MTC' if self.run_multiple_testing_correction else ''} (finding significant BPs)")
        filtered_mapping, metadata = {}, {}
        for strain, d in mapping.items():   
            ############  Population
            #   population (mrna_targets) = all mRNA targets (of all sRNAs) in the strain
            mrna_targets = sorted(set([target for target_to_bp in d.values() for target in target_to_bp.keys()]))
            #   population size (M)
            M = len(mrna_targets)

            # 1 - for each BP in the population, find the number of mRNA TARGETS associated to it (n per BP)
            strain_bps = sorted(set([bp for target_to_bp in d.values() for bp_lst in target_to_bp.values() for bp in bp_lst]))
            bp_to_accociated_mrna_targets = {bp: set() for bp in strain_bps}
            for target_to_bp in d.values():
                for target, bp_lst in target_to_bp.items():
                    for bp in bp_lst:
                        bp_to_accociated_mrna_targets[bp].add(target)

            ############  Selection
            #  metadata:  per sRNA, its BPS, BP name, M,N,n,k, pv before correction, pv after correction
            #  filtered_mapping:  for each sRNA, keep only significant BPs
            d_filtered, d_meta = {}, {}
            for srna, target_to_bp_of_srna in d.items():
                # 1 - Define selection
                #   selection (mrna_targets_of_srna) = all mRNA targets of a specific sRNA in the strain
                mrna_targets_of_srna = sorted(set(target_to_bp_of_srna.keys()))
                #   selection size (N)
                N = len(mrna_targets_of_srna)

                # 2 - for each BP in the selection, find the number of mRNA TARGETS associated to it (k per BP) + update metadata
                srna_bps = sorted(set([bp for bp_lst in target_to_bp_of_srna.values() for bp in bp_lst]))
                bp_to_accociated_mrna_targets_of_srna = {bp: set() for bp in srna_bps}
                for target, bp_lst in target_to_bp_of_srna.items():
                    for bp in bp_lst:
                        bp_to_accociated_mrna_targets_of_srna[bp].add(target)

                # 3 - per BP, assert that the mRNA targets in the selection (k) is a subset of the mRNA targets in the population (n)
                for bp, targets in bp_to_accociated_mrna_targets_of_srna.items():
                    assert targets.issubset(bp_to_accociated_mrna_targets[bp])
                
                # 4 - compute p-value for each BP in the selection
                bp_to_meta = {}
                for bp, targets in bp_to_accociated_mrna_targets_of_srna.items():
                    # 4.1 - Define marked elements
                    #   number of marked elements in the population (n) = number of mRNA targets in the population associated to this BP
                    n = len(bp_to_accociated_mrna_targets[bp])
                    #   number of marked elements in the selection (k) = number of mRNA targets in the selection (sRNA) associated to this BP
                    k = len(targets)

                    # 4.2 - Run cumulative hypergeometric test
                    p_value = self._run_cumulative_hypergeometric_test(M=M, n=n, N=N, k=k)
                    
                    # 4.3 - Update metadata
                    bp_to_meta[bp] = {
                        'BP_id': bp,
                        'BP_lbl': self.G.nodes[bp]['lbl'],
                        'M': M,
                        'n': n,
                        'N': N,
                        'k': k,
                        'k_targets_associated_w_BP': targets,
                        'p_value': p_value
                    }

                # 5 - apply multiple testing correction (Benjamini-Hochberg)
                bps = list(bp_to_meta.keys())
                p_values = [bp_to_meta[bp]['p_value'] for bp in bps]
                # 5.1 - adjusted p-values (FDR with Benjamini-Hochberg)
                adj_p_values = false_discovery_control(p_values, method='bh')
                # 5.2 - update metadata with adjusted p-values
                for i, bp in enumerate(bps):
                    bp_to_meta[bp]['adj_p_value'] = adj_p_values[i]

                ########################################################################################
                #     Decide how to filter the BPs (pre or post correction, which threshold, etc.)
                ########################################################################################
                pv_col = 'adj_p_value' if self.run_multiple_testing_correction else 'p_value'
                
                # 6 - find significant BPs for the sRNA - use adjusted p-values
                significant_srna_bps = []
                for bp, meta in bp_to_meta.items():
                    bp_p_value = meta[pv_col]
                    if bp_p_value <= self.enrichment_pv_threshold:
                        significant_srna_bps.append(bp)
                
                # 7 - Keep only significant BPs for each target of the sRNA
                filtered_target_to_bp_of_srna = {}
                for target, bp_lst in target_to_bp_of_srna.items():
                    # Keep only significant BPs for the target
                    filtered_bps = [bp for bp in bp_lst if bp in significant_srna_bps]
                    if filtered_bps:
                        filtered_target_to_bp_of_srna[target] = filtered_bps
                
                # 8 - if the sRNA has at least one significant BP, keep it in the filtered mapping
                if filtered_target_to_bp_of_srna:
                    d_filtered[srna] = filtered_target_to_bp_of_srna
                
                # 9 - update metadata for the sRNA
                d_meta[srna] = bp_to_meta
                ########################################################################################
                
            filtered_mapping[strain] = d_filtered
            metadata[strain] = d_meta
        
        return filtered_mapping, metadata
    
    def _run_cumulative_hypergeometric_test(self, M: int, n: int, N: int, k: int) -> float:
        """
        Hypergeometric test to find the cumulative probability of observing k or more marked elements in a selection of N elements 
        from a population of M elements, where n is the number of marked elements in the population.
        fornmally:  
                    P(X >= k) = sum_{i=k}^{min(n, N)} (n choose i) * (M-n choose N-i) / (M choose N)

        Args:
            M (int): Total number of elements in the population (i.e., all targets).
            n (int): Number of marked elements in the population (i.e. genes of the population associated to this biological process).
            N (int): Size of the selection (i.e., selection = all targets of a specific sRNA).
            k (int): Number of marked elements in the selection (i.e. genes of the selection that are associated to this biological process).

        Returns:
            float: P(X >= k)
        """
        test_pv = hypergeom(M=M, n=n, N=N).sf(k-1)

        return test_pv
    
############################################
#######   Functions for assistance  ########
############################################

    def _get_node_neighbors_and_edges(self, node_id: str) -> dict:
        """
        Get all neighbors of a node and the edge types to those neighbors.

        Args:
            node_id (str): The ID of the node.

        Returns:
            dict: A dictionary in the following format:
            {
                <neighbor_id>: <edge_type>,
                ...
            }
        """
        if node_id not in self.G:
            self.logger.error(f"Node {node_id} does not exist in the graph.")
            return {}

        neighbors_and_edges = {}
        for neighbor in self.G.neighbors(node_id):
            edge_data = self.G[node_id][neighbor]
            edge_type = edge_data.get('type', 'unknown')
            neighbors_and_edges[neighbor] = edge_type

        return neighbors_and_edges