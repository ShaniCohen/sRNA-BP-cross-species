from typing import Tuple, Set, List, Dict
import pandas as pd
import numpy as np
from os.path import join
import itertools
from pathlib import Path
from utils.general import read_df, write_df, create_dir_if_not_exists
import networkx as nx
from pyvis.network import Network
from scipy.stats import hypergeom, false_discovery_control
import json
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
        
        # graph and utils
        self.graph_version = graph_builder.get_version()
        self.G = graph_builder.get_graph()
        self.U = graph_utils

        # ---------  RUNTIME FLAGS  ---------
        self.run_clustering_of_rna_homologs = False
        self.run_homolog_clusters_stats = True   # Chapter 4.3.3: Clustering of RNA Homologs Across Multiple Strains
        
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
        self.srna_homologs = read_df(join(self.out_path_clustering_homologs, f"sRNA_homologs__{self.out_file_suffix}.csv"))
        self.mrna_homologs = read_df(join(self.out_path_clustering_homologs, f"mRNA_homologs__{self.out_file_suffix}.csv"))

        # 3 - Calculate and dump statistics of homolog clusters
        if self.run_homolog_clusters_stats:  
            self._dump_stats_rna_homolog_clusters_size(val_type = 'ratio')
            self._dump_stats_rna_homolog_clusters_strains_composition(val_type = 'ratio', min_val_limit = 0.02)

        # 4 - Map sRNAs to biological processes (BPs)
        self.logger.info("----- Before enrichment:")
        srna_bp_mapping = self._generate_srna_bp_mapping()
        self._log_srna_bp_mapping(srna_bp_mapping)

        # 5 - Enrichment (per strain): per sRNA, find and keep only significant biological processes (BPs) that its targets are invovlved in.
        if self.run_enrichment:
            self.logger.info("----- After enrichment:")
            srna_bp_mapping, meta = self._apply_enrichment(srna_bp_mapping)
            self._log_srna_bp_mapping(srna_bp_mapping)
            self._dump_metadata(meta)

        self.logger.info(f"--------------   Analysis Tools   --------------")
        # ------   Analysis 1 - Cross-Species Conservation of sRNAs' Functionality
        self._analysis_1_srna_homologs_to_commom_bps(srna_bp_mapping, self.U.exact_bp)

        # ------   Analysis 2 - sRNA Regulation of Biological Processes (BPs)
        # generate mapping of BP to mRNAs and sRNAs
        bp_rna_mapping = self._generate_bp_rna_mapping(srna_bp_mapping)
        self._analysis_2_bp_rna_mapping(bp_rna_mapping)
    
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
        # 1 - paralogs only
        self._cluster_paralogs_only('sRNA', self.U.srna)
        self._cluster_paralogs_only('mRNA', self.U.mrna)
        # 2 - homologs
        self._cluster_homologs('sRNA', self.U.srna)
        self._cluster_homologs('mRNA', self.U.mrna)

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

    def _analysis_1_srna_homologs_to_commom_bps(self, srna_bp_mapping: dict, bp_similarity_method: str):
        self.logger.info(f"##############   Analsis 1 - Cross-Species Conservation of sRNAs' Functionality   ##############")
        # 1 - load clusters of sRNA homologs
        df = self.srna_homologs.copy()
        
        # 2 - analyze common BPs of sRNA homologs
        records = []
        for cluster in df['cluster'].apply(ast.literal_eval):
            rec = self._get_common_bps_of_srna_orthologs(cluster, srna_bp_mapping, bp_similarity_method)
            records.append(rec)
        df[list(records[0].keys())] = pd.DataFrame(records)
        
        # 3 - score
        max_of_max_filtered_homolog_cluster_size = df['max_filtered_homolog_cluster_size'].max() if df['max_filtered_homolog_cluster_size'].max() > 0 else 1
        df['score'] = 100 * df['max_strains_with_common_BPs'] + 10 * (df['max_filtered_homolog_cluster_size'] / max_of_max_filtered_homolog_cluster_size)
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

        # 4 - dump results
        # 4.1 - csv
        write_df(df, join(self.out_path_analysis_tool_1, f"Tool_1__sRNA_homologs_to_common_BPs__{self.out_file_suffix}.csv"))
        # 4.2 - manual txt
        clusters_for_txt = [('E2348C_ncR46', 'G0-8867', 'GcvB', 'gcvB', 'ncRNA0016'), ('E2348C_ncR22', 'G0-8863', 'gene-SGH10_RS11370', 'ncRNA0059'), ('E2348C_ncR58', 'G0-8871', 'arcZ', 'ncRNA0002'), ('E2348C_ncR33', 'G0-8878', 'cyaR', 'ncRNA0009'), ('E2348C_ncR60', 'G0-8872', 'RyhB', 'ncRNA0030', 'ncRNA0069', 'ryhB-1', 'ryhB-2'), ('E2348C_ncR66', 'EG30098', 'Spot42', 'ncRNA0077', 'spf'), ('E2348CN_0008', 'G0-10671', 'mgrR', 'ncRNA0046'), ('E2348CN_0007', 'G0-10677', 'fnrS', 'ncRNA0015'), ('E2348CN_0013', 'G0-16649', 'cpxQ', 'ncRNA0206'), ('E2348C_ncR48', 'G0-8882', 'ncRNA0052', 'omrB')]
        dump_manual_txt = True
        if dump_manual_txt:
            with open(join(self.out_path_analysis_tool_1, f"Tool_1__Manual_{self.graph_version}.txt"), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    cluster = row['cluster'] if type(row['cluster']) == tuple else ast.literal_eval(row['cluster'])
                    if cluster in clusters_for_txt:
                        f.write(f"{row['cluster']}\n")
                        f.write(f"{row['srnas_to_targets_to_BPs']}\n\n")

        # 5 - log statistics
        num_clusters = len(df)
        # 5.1 - max common BPs distribution
        unq, counts = np.unique(df['max_common_BPs'], return_counts=True)
        sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
        max_common_bps_dist = "\n   ".join([f"{counts} with max common BPs = {unq} ({int(round(counts/num_clusters, 2)*100)} %)" for unq, counts in sorted_dict.items()])
        
        self.logger.info(
            f"----------------   sRNA homologs \n"
            f"-------   BP similarity method = {bp_similarity_method} \n"
            f"Number of clusters: {len(df)} \n"
            f"max common BPs distribution: \n"
            f"   {max_common_bps_dist}"
        )

    def _get_common_bps_of_srna_orthologs(self, orthologs_cluster: Tuple[str], srna_bp_mapping: dict, bp_similarity_method: str) -> dict:
        # 1 - all BPs
        all_bps, num_all_bps = {}, {}
        # 2 - sRNAs to targets to BPs (complete)
        srnas_to_targets_to_bps = {}  
         # for orthologs clusters of targets
        strain_to_mrna_list = {}

        for srna_id in orthologs_cluster:
            strain = self.G.nodes[srna_id]['strain']
            srna_targets = srna_bp_mapping[strain].get(srna_id)
            if srna_targets:
                unq_bps = sorted({bp for bps in srna_targets.values() for bp in bps})
                all_bps[f'{strain}__{srna_id}'] = unq_bps
                num_all_bps[f'{strain}__{srna_id}'] = len(unq_bps)
                
                # sRNAs to targets to BPs (complete)
                srna_complete = f"{strain}__{srna_id}__{self.G.nodes[srna_id]['name']}" 
                targets_to_bps_complete = {f"{target_id}__{self.G.nodes[target_id]['name']}": [f"{bp_id}__{self.G.nodes[bp_id]['lbl']}" for bp_id in bps] for target_id, bps in srna_targets.items()}
                srnas_to_targets_to_bps[srna_complete] = targets_to_bps_complete
                
                # for orthologs clusters of targets
                if strain not in strain_to_mrna_list.keys():
                    strain_to_mrna_list[strain] = []
                strain_to_mrna_list[strain].extend(list(srna_targets.keys()))

        num_srna_w_bps = len(all_bps)
        srnas_to_targets_to_bps = dict(sorted(srnas_to_targets_to_bps.items()))
        
        # 3 - common BPs
        all_common_bps, num_common_bps, max_strains_w_common_bps, all_common_bps_of_max_strains, max_common_bps = self._calc_common_bps(all_bps, bp_similarity_method)

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
        
        # 5 - 
        max_filtered_homolog_cluster_size = max([len(c) for c in filtered_homolog_clusters_of_targets], default=0)

        # 6 - output record
        rec = {
                'num_srna_w_bps': num_srna_w_bps, 
                'all_common_BPs': all_common_bps, 
                'num_common_BPs': num_common_bps, 
                'max_strains_with_common_BPs': max_strains_w_common_bps,
                'all_common_BPs_of_max_strains': all_common_bps_of_max_strains,
                'max_common_BPs': max_common_bps, 
                'all_BPs': all_bps, 
                'num_all_BPs': num_all_bps,
                'complete_homolog_clusters_of_targets': complete_homolog_clusters_of_targets,
                'filtered_homolog_clusters_of_targets': filtered_homolog_clusters_of_targets,
                'max_filtered_homolog_cluster_size': max_filtered_homolog_cluster_size,
                'srnas_to_targets_to_BPs': srnas_to_targets_to_bps, 
        }

        return rec
    
    def _calc_common_bps(self, all_bps: Dict[str, list], bp_similarity_method: str) -> Tuple[Dict[tuple, list], Dict[tuple, int], int, int]:
        all_common_bps, num_common_bps = {}, {}
        max_common_bps = 0
        max_strains_w_common_bps = 0
        for size in range(2, len(all_bps.keys()) + 1):
            for rna_comb in itertools.combinations(all_bps.keys(), size):
                rnas_bps = list(map(all_bps.get, rna_comb))
                # 1 - common BPs of rnas
                common_bps: List[str] = rnas_bps[0]
                for l in rnas_bps[1:]:
                    common_bps: List[str] = self.U.get_common_bps(common_bps, l, bp_similarity_method)
                
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

    def _cluster_paralogs_only(self, rna_str: str, rna_type: str):
        self.logger.info(f"Clustering and analyzing {rna_str} paralogs only")
        for strain in self.U.strains:
            self.logger.info(f"Strain: {strain}")
            # 1 - all RNAs
            rna_nodes =  [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == strain]
            # 2 - RNAs that are paralogs
            rna_paralogs_clusters = []
            for rna in rna_nodes:
                rna_paralogs = [n for n in self.G.neighbors(rna) if self.G.nodes[n]['type'] == rna_type and self.U.are_paralogs(self.G, rna, n, strain)]
                if rna_paralogs:
                    cluster = set([rna] + rna_paralogs)
                    if cluster not in rna_paralogs_clusters:
                        rna_paralogs_clusters.append(cluster)
            # 3 - log
            self.logger.info(
                f"  ----------------   \n"
                f"  Number of {rna_str}: {len(rna_nodes)} \n"
                f"  Number of {rna_str} with paralogs: {sum(len(c) for c in rna_paralogs_clusters)} \n"
                f"  Number of {rna_str} paralogs clusters: {len(rna_paralogs_clusters)}"
            )
            # 4 - dump
            with open(join(self.out_path_clustering_paralogs_only, f"{strain}_{rna_type}_paralogs_clusters.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Strain: {strain}\n")
                f.write(f"{rna_type} paralogs:\n")
                for cluster in rna_paralogs_clusters:
                    f.write("\n")
                    for rna_node_id in cluster:
                        node_info = self.G.nodes[rna_node_id]
                        f.write(f"  {rna_node_id}: {json.dumps(node_info, ensure_ascii=False)}\n")
    
    def _cluster_homologs(self, rna_str: str, rna_type: str) -> pd.DataFrame:
        self.logger.info(f"Analyzing {rna_str} homologs")
        # 1 - get all homologs clusters
        all_homologs_clusters = set()
        for strain in self.U.strains:
            homologs_clusters = self._get_homologs_clusters_of_strain(rna_type, strain)
            all_homologs_clusters = all_homologs_clusters.union(homologs_clusters)
        # 2 - validate and dump
        all_homologs_df = self._validate_homologs_clusters(rna_type, all_homologs_clusters)
        write_df(all_homologs_df, join(self.out_path_clustering_homologs, f"{rna_str}_homologs__{self.out_file_suffix}.csv"))
    
    def _get_homologs_clusters_of_strain(self, rna_type: str, strain: str):
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
                elif (self.U.are_ortholog_by_seq(self.G, n1, n2, s1, s2) or self.U.are_ortholog_by_name(self.G, n1, n2, s1, s2)):
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
    
    def _convert_strain_names(self, tuples_of_names: List[tuple]) -> List[tuple]:
        new_tuples_of_names = [tuple([self.U.get_short_strain_nm(nm) for nm in ast.literal_eval(tpl)]) for tpl in tuples_of_names]
        return new_tuples_of_names
    
    def _convert_counts_to_vals(self, counts: np.array, denominator: int, val_type: str) -> list:
        """
        Args:
            counts (np.array): array of counts (int)
            denominator (int): denominator
            val_type (str): 'ratio' or 'percentage'
        Returns:
            list: vals according to val_type
        """
        if val_type == 'ratio':
            vals = [float(round(count/denominator, 2)) for count in counts]
        elif val_type == 'percentage':
            vals = [int(round(count/denominator, 2)*100) for count in counts]
        else:
            raise ValueError(f"val_type {val_type} is not supported")
        return vals
        
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
            vals = self._convert_counts_to_vals(counts, num_clusters, 'ratio')
            
            # 2.2 - list of tuples (cluster_size , val) **sorted by cluster_size**
            size_to_val = dict(zip(unq, vals))
            cluster_sizes_vals = [(size, size_to_val.get(size, 0)) for size in range(2, max(unq) + 1)]
            self.logger.debug(f"{rna_type} cluster_sizes_vals ({len(cluster_sizes_vals)}): {cluster_sizes_vals}")
            # 2.3 - limit vals
            if min_val_limit:
                cluster_sizes_vals = [(x, val) for (x, val) in cluster_sizes_vals if val >= min_val_limit]
                self.logger.debug(f"{rna_type} cluster_sizes_vals AFTER limit (val >= {min_val_limit}) ({len(cluster_sizes_vals)}): {cluster_sizes_vals}")
            cluster_sizes_vals_str = str(cluster_sizes_vals).replace(", ", ",").replace("),", ") ")
            # 2.4 - get sizes
            cluster_sizes = [x[0] for x in cluster_sizes_vals]
            cluster_sizes_str = str(cluster_sizes).replace(", ", ",")
            # 2.5 - y max (maximal val)
            y_max_val = max(vals)

            rec = {
                "rna_type": rna_type,
                "num_clusters": num_clusters,
                f"cluster_sizes_{val_type}s": cluster_sizes_vals_str,
                "cluster_sizes": cluster_sizes_str,
                "num_coordinates": len(cluster_sizes),
                f"y_max_{val_type}": y_max_val
            }
            records.append(rec)

        df = pd.DataFrame(records)
        write_df(df, join(self.out_path_rna_homologs_multi_strains, f"{rna_type}_cluster_sizes_{val_type}_{min_val_limit}__{self.out_file_suffix}.csv"))
    
    def _dump_stats_rna_homolog_clusters_strains_composition(self, val_type: str, min_val_limit: float = None):
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
            vals = self._convert_counts_to_vals(counts, num_clusters, val_type)

            # 2.2 - list of tuples (strains_composition , val) **sorted by val** in descending order  
            cluster_compositions_vals = sorted(list(zip(unq, vals)), key=lambda x: x[1], reverse=True)
            self.logger.debug(f"{rna_type} cluster_compositions_vals ({len(cluster_compositions_vals)})")
            # 2.3 - limit vals
            if min_val_limit:
                cluster_compositions_vals = [(x, val) for (x, val) in cluster_compositions_vals if val >= min_val_limit]
                self.logger.debug(f"{rna_type} cluster_compositions_vals AFTER limit (val >= {min_val_limit}) ({len(cluster_compositions_vals)})")
            cluster_compositions_vals_str = str(cluster_compositions_vals).replace("('", "{(").replace("'), ", ")},").replace("', '", ", ").replace("), (", ") (")
            # 2.4 - get compositions
            cluster_compositions = [x[0] for x in cluster_compositions_vals]
            cluster_compositions_str = str(cluster_compositions).replace("('", "{(").replace("'), ", ")},").replace("', '", ", ")
            # 2.5 - add LateX "\textit" command
            for n in list(self.U.strain_nm_to_short.values()):
                cluster_compositions_vals_str = cluster_compositions_vals_str.replace(f"{n}", "\textit{" f"{n}" + "}")
                cluster_compositions_str = cluster_compositions_str.replace(f"{n}", "\textit{" f"{n}" + "}")
            # 2.6 - y max (maximal val)
            y_max_val = max(vals)
            
            rec = {
                "rna_type": rna_type,
                "num_clusters": num_clusters,
                f"cluster_strains_compositions_{val_type}s": cluster_compositions_vals_str,
                "cluster_strains_compositions": cluster_compositions_str,
                "num_coordinates": len(cluster_compositions),
                f"y_max_{val_type}": y_max_val
            }
            records.append(rec)
        
        df = pd.DataFrame(records)
        write_df(df, join(self.out_path_rna_homologs_multi_strains, f"{rna_type}_cluster_compositions_{val_type}_{min_val_limit}__{self.out_file_suffix}.csv"))

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
                f"  Number of BP annotations: {len(bp_list)} \n"
                f"  Number of unique BPs: {len(unique_bps)}"
            )
    
    def _find_homologs(self, strain_to_rna_list: Dict[str, List[str]], rna_str: str) -> List[Tuple[str]]:
        """
        For each cluster in homologs_df['cluster'], find which RNAs from strain_to_rna_list are homologs, i.e., belong to the same cluster.
        Return a set of tuples, each tuple contains the RNAs from strain_to_rna_list that belong to the same cluster.

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