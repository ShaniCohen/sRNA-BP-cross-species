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
        self.G = graph_builder.get_graph()
        self.U = graph_utils

        # clustering_analysis
        self.run_clustering_analysis = False
        self.srna_orthologs = read_df(join(self.config['analysis_output_dir'], "orthologs", f"sRNA_orthologs.csv"))
        self.mrna_orthologs = read_df(join(self.config['analysis_output_dir'], "orthologs", f"mRNA_orthologs.csv"))

        # TODO: remove after testing
        self.strains_data = graph_builder.strains_data
        self.ips_go_annotations = graph_builder.get_ips_go_annotations()

        # TODO: set enrichment p-value threshold
        self.enrichment_pv_threshold = 0.05

    def run_analysis(self, dump_meta: bool = True):
        """
        GO node is represented as a dict item:
            <id_str> : {'type': <str>, 'lbl': <str>, 'meta': <dict>}
        mRNA/sRNA node is represented as a dict item:
            <id_str> : {'type': <str>, 'strain': <str>, 'locus_tag': <str>, 'name': <str>, 'synonyms': <List[str]>, 'start': <float>, 'end': <float>, 'strand': <str>, 'sequence': <str>}
        edge is represented as a dict item:
            (<id_str>, <id_str>) : {'type': <str>}
        """
        self.logger.info(f"running analysis")

        # --------------   run analysis   --------------
         # 1 - Analyze RNA clustering (orthologs and paralogs)
        if self.run_clustering_analysis:   
            self._analyze_rna_clustering()

        # 2 - Generate a mapping of sRNA to biological processes (BPs)
        self.logger.info("----- Before enrichment:")
        srna_bp_mapping = self._generate_srna_bp_mapping()
        self._log_srna_bp_mapping(srna_bp_mapping)
        
        # 3 - generate mapping of BP to mRNAs and sRNAs
        bp_rna_mapping = self._generate_bp_rna_mapping(srna_bp_mapping)
        #TODO: log BP to RNAs mapping
        self._log_bp_rna_mapping(bp_rna_mapping)

        # 4 - Enrichment (per strain): per sRNA, find and keep only significant biological processes (BPs) that its targets invovlved in.
        # self.logger.info("----- After enrichment:")
        # srna_bp_mapping_post_en, meta = self._apply_enrichment(srna_bp_mapping)
        # self._log_mapping(srna_bp_mapping_post_en)
        # # 4.1 - dump metadata
        # if dump_meta:
        #     self._dump_metadata(meta)

        # 5 - BPs of sRNA orthologs
        # 5.1 - BP similarity = exact
        self._analyze_bps_of_srna_orthologs(srna_bp_mapping, self.U.exact_bp)
        print()
    
    def _analyze_rna_clustering(self):
        # 1 - paralogs
        self._analyze_paralogs('sRNA', self.U.srna)
        self._analyze_paralogs('mRNA', self.U.mrna)
        # 2 - orthologs
        # 2.1 - analyze
        srna_orthologs = self._analyze_orthologs('sRNA', self.U.srna)
        mrna_orthologs = self._analyze_orthologs('mRNA', self.U.mrna)
        # 2.2 - save
        self.srna_orthologs = srna_orthologs
        self.mrna_orthologs = mrna_orthologs

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
                            <mRNA_target_id>: [<sRNA_id1>, <sRNA_id2>, ...],
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
                # per strain, find all interacting mRNAs that relate to curr BP
                strain_mrna_to_srnas = {}
                for srna_id, mrna_to_bps in srna_dict.items():
                    for mrna_id, bps in mrna_to_bps.items():
                        if bp in bps:
                            if mrna_id not in strain_mrna_to_srnas:
                                strain_mrna_to_srnas[mrna_id] = []
                            strain_mrna_to_srnas[mrna_id].append(srna_id)
                if strain_mrna_to_srnas:
                    bp_rna_mapping[bp][strain] = strain_mrna_to_srnas
        return bp_rna_mapping

    def _analyze_bps_of_srna_orthologs(self, srna_bp_mapping: dict, bp_similarity_method: str):
        self.logger.info(f"Analyzing BPs of sRNA orthologs")
        # 1 - load sRNA orthologs
        _path = create_dir_if_not_exists(join(self.config['analysis_output_dir'], "orthologs"))
        all_orthologs_df = read_df(join(_path, f"sRNA_orthologs.csv"))
        # 2 - analyze common BPs of sRNA orthologs
        records = []
        for cluster in all_orthologs_df['cluster'].apply(ast.literal_eval):
            all_common_bps, num_common_bps, max_common_bps, all_bps, num_all_bps = self._get_common_bps_of_srna_orthologs(cluster, srna_bp_mapping, bp_similarity_method)
            records.append({'all_common_BPs': all_common_bps, 'num_common_BPs': num_common_bps, 'max_common_BPs': max_common_bps, 'all_BPs': all_bps, 'num_all_BPs': num_all_bps})
        all_orthologs_df[list(records[0].keys())] = pd.DataFrame(records)
        # 3 - log and dump
        num_clusters = len(all_orthologs_df)
        # 3.1 - max common BPs distribution
        unq, counts = np.unique(all_orthologs_df['max_common_BPs'], return_counts=True)
        sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
        max_common_bps_dist = "\n   ".join([f"{counts} with max common BPs = {unq} ({int(round(counts/num_clusters, 2)*100)} %)" for unq, counts in sorted_dict.items()])
        
        self.logger.info(
            f"----------------   sRNA orthologs \n"
            f"-------   BP similarity method = {bp_similarity_method} \n"
            f"Number of clusters: {len(all_orthologs_df)} \n"
            f"max common BPs distribution: \n"
            f"   {max_common_bps_dist}"
        )
        write_df(all_orthologs_df, join(_path, f"sRNA_orthologs_w_BPs.csv"))

    def _get_common_bps_of_srna_orthologs(self, orthologs_cluster: Tuple[str], srna_bp_mapping: dict, bp_similarity_method: str) -> Tuple[Dict[tuple, list], Dict[tuple, int], int, Dict[str, list], Dict[str, int]]:
        # 1 - all BPs
        all_bps, num_bps = {}, {}
        for srna in orthologs_cluster:
            strain = self.G.nodes[srna]['strain']
            srna_targets = srna_bp_mapping[strain].get(srna)
            if srna_targets:
                unq_bps = sorted({bp for bps in srna_targets.values() for bp in bps})
                all_bps[f'{strain}__{srna}'] = unq_bps
                num_bps[f'{strain}__{srna}'] = len(unq_bps)
        # 2 - common BPs
        all_common_bps, num_common_bps, max_common_bps = self._calc_common_bps(all_bps, bp_similarity_method)

        return all_common_bps, num_common_bps, max_common_bps, all_bps, num_bps
    
    def _calc_common_bps(self, all_bps: Dict[str, list], bp_similarity_method: str) -> Tuple[Dict[tuple, list], Dict[tuple, int], int]:
        all_common_bps, num_common_bps = {}, {}
        max_common_bps = 0
        for size in range(2, len(all_bps.keys()) + 1):
            for rnas in itertools.combinations(all_bps.keys(), size):
                rnas_bps = list(map(all_bps.get, rnas))
                # 1 - common BPs of rnas
                common_bps: List[str] = rnas_bps[0]
                for l in rnas_bps[1:]:
                    common_bps: List[str] = self.U.get_common_bps(common_bps, l, bp_similarity_method)
                all_common_bps[rnas] = common_bps
                # 2 - num common BPs
                num_common_bps[rnas] = len(common_bps)
                # 3 - max common BPs
                max_common_bps = max(max_common_bps, len(common_bps))
        return all_common_bps, num_common_bps, max_common_bps

    def _analyze_paralogs(self, rna_str: str, rna_type: str):
        self.logger.info(f"Analyzing {rna_str} paralogs")
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
            # 3 - log and dump
            self.logger.info(
                f"  ----------------   \n"
                f"  Number of {rna_str}: {len(rna_nodes)} \n"
                f"  Number of {rna_str} with paralogs: {sum(len(c) for c in rna_paralogs_clusters)} \n"
                f"  Number of {rna_str} paralogs clusters: {len(rna_paralogs_clusters)}"
            )
            self._dump_paralogs(strain, rna_str, rna_paralogs_clusters)
    
    def _analyze_orthologs(self, rna_str: str, rna_type: str) -> pd.DataFrame:
        self.logger.info(f"Analyzing {rna_str} orthologs")
        # 1 - get all orthologs clusters
        all_orthologs_clusters = set()
        for strain in self.U.strains:
            orthologs_clusters = self._get_orthologs_clusters_of_strain(rna_type, strain)
            all_orthologs_clusters = all_orthologs_clusters.union(orthologs_clusters)
        # 2 - validate
        all_orthologs_df = self._validate_orthologs_clusters(rna_type, all_orthologs_clusters)
        # 3 - log and dump
        self._log_n_dump_orthologs(rna_str, rna_type, all_orthologs_df)

        return all_orthologs_df
    
    def _get_orthologs_clusters_of_strain(self, rna_type: str, strain: str):
        rna_nodes =  [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == strain]
        # 1 - all clusters
        all_clusters = set()
        clusters_items = []
        for rna in rna_nodes:
            if rna not in clusters_items:
                cluster = set()
                cluster = self.U.get_orthologs_cluster(self.G, rna, cluster)
                if cluster:
                    all_clusters.add(tuple(sorted(cluster)))
                    clusters_items = clusters_items + sorted(cluster)
        assert set(rna_nodes) <= set(clusters_items), "some RNA nodes are missing in the clusters"
        # 2 - orthologs clusters
        orthologs_clusters = {c for c in all_clusters if len(c) > 1}
        if not orthologs_clusters:
            self.logger.warning(f"#### no {rna_type} orthologs clusters for {strain}")
        
        return orthologs_clusters
    
    def _validate_orthologs_clusters(self, rna_type: str, orthologs_clusters: Set[Tuple[str]]) -> pd.DataFrame:
        # 1 - general validation
        nodes = [item for tpl in orthologs_clusters for item in tpl]
        assert len(set(nodes)) == len(nodes), f"some {rna_type} nodes are duplicated in the orthologs clusters"
        
        # 2 - per cluster validation
        records = []
        for cluster in orthologs_clusters:
            # 2.1 - get ortholog pairs and strains
            ortholog_pairs, strains = list(), set()
            ortholog_pairs_w_meta = list()
            for n1, n2 in itertools.combinations(list(cluster), 2):
                s1, s2 = self.G.nodes[n1]['strain'], self.G.nodes[n2]['strain']
                if self.U.are_orthologs(self.G, n1, n2, s1, s2):
                    ortholog_pairs.append((n1, n2))
                    strains = strains.union({s1, s2})
                    ortholog_pairs_w_meta.append((f"{s1}__{n1}__{self.G.nodes[n1]['name']}", f"{s2}__{n2}__{self.G.nodes[n2]['name']}"))
            # 2.2 - validate
            assert set(cluster) == set([n for tpl in ortholog_pairs for n in tpl]), f"some {rna_type} cluster nodes are missing in the orthologs pairs"
            # 2.3 - add record
            records.append({
                'cluster': cluster,
                'cluster_size': len(cluster),
                'strains': tuple(sorted(strains)),
                'num_strains': len(strains),
                'ortholog_pairs': sorted(ortholog_pairs_w_meta),
                'num_ortholog_pairs': len(ortholog_pairs_w_meta)
            })
        orthologs_df = pd.DataFrame(records)

        return orthologs_df

    def _log_n_dump_orthologs(self, rna_str: str, rna_type: str, all_orthologs_df: pd.DataFrame):
        out_path = create_dir_if_not_exists(join(self.config['analysis_output_dir'], "orthologs"))
        
        # 1 - general analysis
        num_clusters = len(all_orthologs_df)
        # cluster size distribution
        unq, counts = np.unique(all_orthologs_df['cluster_size'], return_counts=True)
        size_dist = " | ".join([f"{counts[i]} of size {unq[i]} ({int(round(counts[i]/num_clusters, 2)*100)} %)" for i in range(len(unq))])
        # strains distribution
        unq, counts = np.unique([s for clu in all_orthologs_df['strains'] for s in clu], return_counts=True)
        strain_2_num_orthologs = dict(zip(unq, counts))
        strain_dist = " | ".join([f"{counts[i]} includes {unq[i]} ({int(round(counts[i]/num_clusters, 2)*100)} %)" for i in range(len(unq))])
        # strains composition distribution
        unq, counts = np.unique(all_orthologs_df['strains'], return_counts=True)
        sorted_dict = dict(sorted(dict(zip(unq, counts)).items(), key=lambda item: item[1], reverse=True))
        strain_comp_dist = "\n   ".join([f"{counts} of composition {unq} ({int(round(counts/num_clusters, 2)*100)} %)" for unq, counts in sorted_dict.items()])

        # 2 - per strain analysis
        per_strain_analysis = ""
        for strain, data in self.strains_data.items():
            num_rna = len(data[f'all_{rna_type}'])
            num_orthologs = strain_2_num_orthologs.get(strain, 0)
            per_strain_analysis = per_strain_analysis + f"\n  {strain}: {int(round(num_orthologs/num_rna, 2)*100)} % of {rna_str}s ({num_orthologs} out of {num_rna}) have orthologs "

        # 3 - log
        self.logger.info(
            f"----------------   {rna_str} \n"
            f"-------   General analysis \n"
            f"Number of clusters: {len(all_orthologs_df)} \n"
            f"Cluster size distribution: \n"
            f"  {size_dist} \n"
            f"Strains distribution: \n"
            f"  {strain_dist} \n"
            f"Strains composition distribution: \n"
            f"   {strain_comp_dist} \n"
            f"-------   Per strain analysis"
            f"{per_strain_analysis}"
        )
        # 4 - dump
        write_df(all_orthologs_df, join(out_path, f"{rna_str}_orthologs.csv"))

    def _dump_paralogs(self, strain: str, rna_type: str, rna_paralogs_clusters: List[Set[str]]):
        out_path = create_dir_if_not_exists(join(self.config['analysis_output_dir'], "paralogs"))
        out_file = join(out_path, f"{strain}_{rna_type}_paralogs_clusters.txt")

        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"Strain: {strain}\n")
            f.write(f"{rna_type} paralogs:\n")
            for cluster in rna_paralogs_clusters:
                f.write("\n")
                for rna_node_id in cluster:
                    node_info = self.G.nodes[rna_node_id]
                    f.write(f"  {rna_node_id}: {json.dumps(node_info, ensure_ascii=False)}\n")

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
    
    def _find_orthologs(self, strain_to_rna_list: Dict[str, List[str]], rna_str: str) -> List[Tuple[str]]:
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
        orthologs_df = self.srna_orthologs if rna_str == 'sRNA' else self.mrna_orthologs
        all_rna_orthologs = set()
        # Flatten all RNAs from strain_to_rna_list
        all_rnas = set()
        for rna_list in strain_to_rna_list.values():
            all_rnas.update(rna_list)
        # Iterate over clusters
        for cluster in orthologs_df['cluster']:
            # Find intersection with all_rnas
            rna_orthologs = tuple(sorted(set(ast.literal_eval(cluster)).intersection(all_rnas)))
            if len(rna_orthologs) > 1:
                all_rna_orthologs.update(rna_orthologs)
        return sorted(all_rna_orthologs)

    def _log_bp_rna_mapping(self, mapping: dict):
        """
        Generate a DataFrame with information about BPs and their related mRNAs and sRNAs

        Args:
            mapping (dict): A dictionary in the following format:
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
        # 1 - generate df
        records = []
        for bp_id, strain_dict in mapping.items():
            strains = sorted(strain_dict.keys())
            num_strains = len(strains)
            related_mRNAs = {}
            num_related_mRNAs = {}
            related_sRNAs = {}
            num_related_sRNAs = {}
            for strain, mrna_to_srnas in strain_dict.items():
                related_mRNAs[strain] = sorted(mrna_to_srnas.keys())
                num_related_mRNAs[strain] = len(mrna_to_srnas)
                # flatten sRNA lists for all mRNAs in this strain
                srna_set = set()
                for srna_list in mrna_to_srnas.values():
                    srna_set.update(srna_list)
                related_sRNAs[strain] = sorted(srna_set)
                num_related_sRNAs[strain] = len(srna_set)
            records.append({
                'bp_id': bp_id,
                'strains': strains,
                'num_strains': num_strains,
                'related_mRNAs_and_sRNAs': strain_dict,
                'related_mRNAs': related_mRNAs,
                'num_related_mRNAs': num_related_mRNAs,
                'related_sRNAs': related_sRNAs,
                'num_related_sRNAs': num_related_sRNAs
            })
        df = pd.DataFrame(records)

        # 2 - identify orthologs
        for rna_str in ['sRNA', 'mRNA']:
            df[f'related_{rna_str}_orthologs'] = list(map(self._find_orthologs, df[f'related_{rna_str}s'], np.repeat(rna_str, len(df))))

        # 3 - log
        self.logger.info(f"--------- BP to RNAs mapping\n{df.head()}")
        # Optionally, dump to file if needed
        # out_path = self.config['analysis_output_dir']
        # df.to_csv(os.path.join(out_path, "bp_to_rnas_mapping.csv"), index=False)
    
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
        out_path = self.config['analysis_output_dir']
        self.logger.info(f"Dumping metadata to {out_path}")
        os.makedirs(out_path, exist_ok=True)

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
            file_path = os.path.join(out_path, f"metadata_per_srna_{strain}.csv")
            df.to_csv(file_path, index=False)
            # self.logger.debug(f"Metadata for strain {strain} dumped to {file_path}")
    
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
        self.logger.info(f"applying enrichment (finding significant BPs)")
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

                ################################################################
                # TODO: decide how to filter the BPs (pre or post correction, which threshold, etc.)
                ################################################################
                # 6 - find significant BPs for the sRNA - use adjusted p-values
                significant_srna_bps = []
                for bp, meta in bp_to_meta.items():
                    # bp_p_value = meta['p_value']
                    bp_p_value = meta['adj_p_value']
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
                ################################################################
                ################################################################
                
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