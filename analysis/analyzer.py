from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
from utils.general import read_df, write_df
import networkx as nx
from pyvis.network import Network
from scipy.stats import hypergeom
import json
import sys
import os

ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

class Analyzer:
    def __init__(self, config, logger, graph_builder):
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
        
        # graph
        self.strains = graph_builder.get_strains()
        self.G = graph_builder.get_graph()

        # TODO: remove after testing
        self.strains_data = graph_builder.strains_data

        # node types
        # GO
        self._bp = graph_builder._bp
        self._mf = graph_builder._mf
        self._cc = graph_builder._cc
        # mRNA
        self._mrna = graph_builder._mrna
        # sRNA
        self._srna = graph_builder._srna

        # edge types
        # GO --> GO
        self._part_of = graph_builder._part_of
        self._regulates = graph_builder._regulates
        self._neg_regulates = graph_builder._neg_regulates
        self._pos_regulates = graph_builder._pos_regulates
        self._is_a = graph_builder._is_a
        self._sub_property_of = graph_builder._sub_property_of
        # mRNA --> GO
        self._annotated = graph_builder._annotated
        # sRNA --> mRNA     
        self._targets = graph_builder._targets

    def run_analysis(self):
        """
        GO node is represented as a dict item:
            <id_str> : {'type': <str>, 'lbl': <str>, 'meta': <dict>}
        mRNA/sRNA node is represented as a dict item:
            <id_str> : {'type': <str>, 'strain': <str>, 'locus_tag': <str>, 'name': <str>, 'synonyms': <List[str]>, 'start': <float>, 'end': <float>, 'strand': <str>, 'sequence': <str>}
        edge is represented as a dict item:
            (<id_str>, <id_str>) : {'type': <str>}
        """
        self.logger.info(f"running analysis")
        bp_mapping = self._generate_srna_bp_mapping()
        self._log_mapping(bp_mapping)
        bp_mapping_post_en = self._apply_enrichment(bp_mapping)
        self._log_mapping(bp_mapping_post_en)
        
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
        for strain in self.strains:
            # Filter sRNA nodes for the current strain
            # d = self.strains_data[strain]['unq_inter'][self.strains_data[strain]['unq_inter']['sRNA_accession_id'] == 'G0-16636']
            srna_bp_mapping = {}
            srna_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self._srna and d['strain'] == strain]
            for srna in srna_nodes:
                targets_to_bp = {}
                srna_targets = [n for n in self.G.neighbors(srna) if self.G.nodes[n]['type'] == self._mrna and self.G[srna][n]['type'] == self._targets]
                for target in srna_targets:
                    # Find the biological processes associated with the target
                    bp_nodes = [n for n in self.G.neighbors(target) if self.G.nodes[n]['type'] == self._bp and self.G[target][n]['type'] == self._annotated]
                    if bp_nodes:
                        targets_to_bp[target] = bp_nodes
                if targets_to_bp:
                    srna_bp_mapping[srna] = targets_to_bp
            bp_mapping[strain] = srna_bp_mapping

        return bp_mapping

    def _log_mapping(self, mapping: dict):
        for strain, srna_data in mapping.items():
            srna_count = len(srna_data)
            mrna_targets = [mrna for srna_targets in srna_data.values() for mrna in srna_targets.keys()]
            unique_mrna_targets = set(mrna_targets)
            bp_list = [bp for srna_targets in srna_data.values() for bps in srna_targets.values() for bp in bps]
            unique_bps = set(bp_list)

            self.logger.info(
                f"Strain: {strain} \n"
                f"  Number of sRNA keys: {srna_count} \n"
                f"  Number of mRNA targets: {len(mrna_targets)} \n"
                f"  Number of unique mRNA targets: {len(unique_mrna_targets)} \n"
                f"  Number of BPs: {len(bp_list)} \n"
                f"  Number of unique BPs: {len(unique_bps)}"
            )

    def _apply_enrichment(self, mapping: dict) -> dict:
        """
        Enrichment (per strain): 
            per sRNA, find and keep only significant biological processes (BPs) that its targets invovlved in.
            significant BPs are found using a hypergeometric test.
        """
        self.logger.info(f"applying enrichment (finding significant BPs)")
        for strain, srna_data in mapping.items():
            srna_count = len(srna_data)
            mrna_targets = [mrna for srna_targets in srna_data.values() for mrna in srna_targets.keys()]
            unique_mrna_targets = set(mrna_targets)
            bp_list = [bp for srna_targets in srna_data.values() for bps in srna_targets.values() for bp in bps]
            unique_bps = set(bp_list)

            self.logger.info(
                f"Strain: {strain} \n"
                f"  Number of sRNA keys: {srna_count} \n"
                f"  Number of mRNA targets: {len(mrna_targets)} \n"
                f"  Number of unique mRNA targets: {len(unique_mrna_targets)} \n"
                f"  Number of BPs: {len(bp_list)} \n"
                f"  Number of unique BPs: {len(unique_bps)}"
            )
            #####################
            test_pv = self._run_hypergeometric_test(M=52, n=26, N=12, k=7)
        
        return
    
    def _run_hypergeometric_test(self, M: int, n: int, N: int, k: int) -> float:
        """
        Hypergeometric test to find the cumulative probability of observing k or more marked elements in a selection of N elements 
        from a population of M elements, where n is the number of marked elements in the population.
        fornmally:

        P(X >= k) = sum_{i=k}^{min(n, N)} (n choose i) * (M-n choose N-i) / (M choose N)
        
        Parameters:
        M (int): Total number of elements in the population.
        n (int): Number of marked elements in the population (i.e. genes associated to this biological process).
        N (int): Size of the selection.
        k (int): Number of marked elements in the selection (i.e. genes of the group of interest that are associated to this biological process).
        """
        test_pv = hypergeom(M=M, n=n, N=N).sf(k-1)

        """
        g = 75  ## Number of submitted genes
        k = 59  ## Size of the selection, i.e. submitted genes with at least one annotation in GO biological processes
        m = 611  ## Number of "marked" elements, i.e. genes associated to this biological process
        N = 13588  ## Total number of genes with some annotation in GOTERM_BP_FAT.
        n = N - m  ## Number of "non-marked" elements, i.e. genes not associated to this biological process
        x = 19  ## Number of "marked" elements in the selection, i.e. genes of the group of interest that are associated to this biological process

        # Python
        hypergeom(M=N, n=m, N=k).sf(x-1)
        """

        return test_pv
