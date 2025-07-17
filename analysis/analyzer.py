from typing import Tuple, Set
import pandas as pd
import numpy as np
from pathlib import Path
from utils.general import read_df, write_df
import networkx as nx
from pyvis.network import Network
from scipy.stats import hypergeom, false_discovery_control
import json
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

        self._analyze_rna_clustering()
        # --------------   run analysis   --------------        
        # 1 - Generate a mapping of sRNA to biological processes (BPs)
        bp_mapping = self._generate_srna_bp_mapping()
        self._log_mapping(bp_mapping)
        # 2 - Enrichment (per strain): per sRNA, find and keep only significant biological processes (BPs) that its targets invovlved in.
        bp_mapping_post_en, meta = self._apply_enrichment(bp_mapping)
        self._log_mapping(bp_mapping_post_en)
        # 3 - dump metadata
        if dump_meta:
            self._dump_metadata(meta)
    
    def _analyze_rna_clustering(self):
        print("Analyzing RNA clustering...")
        for strain in self.U.strains:
            # Get mRNA nodes for the strain
            mrna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.mrna and d['strain'] == strain]
            # Get sRNA nodes for the strain
            srna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.srna and d['strain'] == strain]

            # Count sRNA nodes with paralogues (sRNA-sRNA)
            srna_with_paralogues = [
                srna for srna in srna_nodes if any(self.U.are_paralogs(self.G, srna, n, strain) for n in self.G.neighbors(srna))
            ]

            mrna_with_paralogues = [
                mrna for mrna in mrna_nodes if any(self.U.are_paralogs(self.G, mrna, n, strain) for n in self.G.neighbors(mrna) if self.G.nodes[n]['type'] == self.U.mrna)
            ]

            for srna in srna_with_paralogues:
                print()
                print(f"Strain: {strain}")
                print(f"sRNA: {srna}: {self.G.nodes[srna]}")
                print("Paralogues: ")
                for n in self.G.neighbors(srna): 
                    if self.U.are_paralogs(self.G, srna, n, strain):
                        print(f"  {n}: {self.G.nodes[n]}")
            
            # TODO: review K12 mRNAs: EG10085 (ascb), EG10114 (bglb)
            for mrna in mrna_with_paralogues:
                print()
                print(f"Strain: {strain}")
                print(f"mRNA: {mrna}: {self.G.nodes[mrna]}")
                print("Paralogues: ")
                for n in self.G.neighbors(mrna): 
                    if self.G.nodes[n]['type'] == self.U.mrna and self.U.are_paralogs(self.G, mrna, n, strain):
                        print(f"  {n}: {self.G.nodes[n]}")

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
            self._log_mrna_with_bp(strain, unq_targets_without_bp)

        return bp_mapping

    def _log_mrna_with_bp(self, strain: str, unq_targets_without_bp: Set[str]):
            ips_annot = self.ips_go_annotations.get(strain, None)
            if ips_annot is not None:
                mrna_w_mf = set(ips_annot[pd.notnull(ips_annot['MF_go_xrefs'])]['mRNA_accession_id'])
                mrna_w_cc = set(ips_annot[pd.notnull(ips_annot['CC_go_xrefs'])]['mRNA_accession_id'])
                mf_count = np.intersect1d(unq_targets_without_bp, mrna_w_mf).size
                cc_count = np.intersect1d(unq_targets_without_bp, mrna_w_cc).size
            self.logger.info(f"Strain: {strain}, Number of unique mRNAs targets without BPs: {len(unq_targets_without_bp)} (where {mf_count} have ips MF annotation, {cc_count} have ips CC annotation)")

    def _log_mapping(self, mapping: dict):
        for strain, srna_bp_mapping in mapping.items():
            srna_count = len(srna_bp_mapping)
            mrna_targets = [mrna for srna_targets in srna_bp_mapping.values() for mrna in srna_targets.keys()]
            unique_mrna_targets = set(mrna_targets)
            bp_list = [bp for srna_targets in srna_bp_mapping.values() for bps in srna_targets.values() for bp in bps]
            unique_bps = set(bp_list)

            self.logger.info(
                f"Strain: {strain} \n"
                f"  Number of sRNA keys: {srna_count} \n"
                f"  Number of mRNA targets (with BP): {len(mrna_targets)} \n"
                f"  Number of unique mRNA targets (with BP): {len(unique_mrna_targets)} \n"
                f"  Number of BPs: {len(bp_list)} \n"
                f"  Number of unique BPs: {len(unique_bps)}"
            )
    
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
            self.logger.info(f"Metadata for strain {strain} dumped to {file_path}")
    
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