# from dbm.ndbm import library
from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
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
        self.bp_clustering_config = self.config['bp_clustering_config']

        self.ecoli_k12_nm = data_loader.ecoli_k12_nm
        # self.vibrio_nm = data_loader.vibrio_nm
        # self.pseudomonas_nm = data_loader.pseudomonas_nm

        self.strains_data = data_loader.strains_data
        self.srna_acc_col = data_loader.srna_acc_col
        self.mrna_acc_col = data_loader.mrna_acc_col
        self.go_embeddings_data = data_loader.go_embeddings_data
        self.clustering_data = data_loader.clustering_data
        
        self.ontology = ontology
        self.curated_go_ids_missing_in_ontology = set()
        self.bp_id_to_cluster = {}
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
        self.run_n_dump_stats_of_homology_edges = False   # RNA Homology Detection Between Strain Pairs (Statistics)
        self.run_n_dump_po2vec_clustering = True   # PO2Vec-based Clustering of BPs

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
       
        # self._add_po2vec_embeddings_and_clusters_to_bp_nodes()
        self._run_wang()

        # TODO: update
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
        if self.run_n_dump_stats_of_homology_edges:
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
    
    def _get_strain_str(self, strain_nm: str, strain_nodes: List[str]) -> str:
        _str = f"{strain_nm} \n (N = {len(strain_nodes)})"
        return _str

    def _dump_homology_edges_stats(self, rna_type: str):
        """Calc and dump statistics of homology edge types between RNA nodes of different strains"""
        self.logger.info(f"Calc and dump statistics - {rna_type} homology edges")
        strain_col = 'Strain'
        strain_to_rec = {}
        for (curr_strain, other_strain) in list(itertools.combinations(self.U.strains, 2)):
            self.logger.debug(f"---- comparing: {(curr_strain, other_strain)}")
            # 1 - get record of curr strain
            if not curr_strain in strain_to_rec:
                curr_node_ids = [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == curr_strain]
                strain_to_rec[curr_strain] = {strain_col: self._get_strain_str(curr_strain, curr_node_ids)}
            rec = strain_to_rec[curr_strain]
            # 2 - compare curr strain to other strain
            # 2.1 - get RNA ortholog pairs (between curr & other)
            other_node_ids = [n for n, d in self.G.nodes(data=True) if d['type'] == rna_type and d['strain'] == other_strain]
            ortholog_by_seq, ortholog_by_name, ortholog_by_seq_and_name = self._get_pairs_w_ortholog_edges_for_van(curr_node_ids, other_node_ids)
            # 2.2 - calculate numbers for Van diagram
            seq_only = len(ortholog_by_seq - ortholog_by_seq_and_name)
            name_only = len(ortholog_by_name - ortholog_by_seq_and_name)
            seq_and_name = len(ortholog_by_seq_and_name)
            _col_val = (seq_only, seq_and_name, name_only)
            # 3 - update record of curr strain (add a column for comparision with other)
            _col_nm = self._get_strain_str(other_strain,other_node_ids)
            rec.update({_col_nm: _col_val})
            strain_to_rec[curr_strain] = rec

        df = pd.DataFrame(list(strain_to_rec.values()))
        write_df(df, join(self.out_path_homology_pairs_stats, f"{rna_type}_homology_edges__{self.out_file_suffix}.csv"))
    
    def _get_pairs_w_ortholog_edges_for_van(self, node_ids_1: List[str], node_ids_2: List[str]) -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
        """ Get pairs of nodes that are linked with different types of ortholog edges.
            Checks all node pairs (n1, n2), where n1 in node_ids_1 and n2 in node_ids_2.

        Args:
            node_ids_1 (List[str]): 
            node_ids_2 (List[str]): 
            ortholog_edge_types (List[str]): types of ortholog edges to consider.

        Returns:
            Set[tuple]: ortholog_by_seq
            Set[tuple]: ortholog_by_name
            Set[tuple]: ortholog_by_seq_and_name
        """
        # 1 - fetch strains
        strains_1 = {n: self.G.nodes[n]['strain'] for n in node_ids_1}
        strains_2 = {n: self.G.nodes[n]['strain'] for n in node_ids_2}
        
        ortholog_by_seq, ortholog_by_name, ortholog_by_seq_and_name = [], [], []
        for (n1, n2) in [(n1, n2) for n1 in node_ids_1 for n2 in node_ids_2]:
            s1, s2 = strains_1[n1], strains_2[n2]
            _by_seq = self.U.are_orthologs_by_seq(self.G, n1, n2, s1, s2)
            _by_name = self.U.are_orthologs_by_name(self.G, n1, n2, s1, s2)
            if _by_seq:
                ortholog_by_seq.append((n1, n2))
            if _by_name:
                ortholog_by_name.append((n1, n2))
            if _by_seq and _by_name:
                ortholog_by_seq_and_name.append((n1, n2))

        return set(ortholog_by_seq), set(ortholog_by_name), set(ortholog_by_seq_and_name)

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

    def _add_po2vec_embeddings_and_clusters_to_bp_nodes(self):
        # 1 - add PO2Vec embeddings to BP nodes
        bp_to_po2vec_emb = self._add_po2vec_embeddings_to_bp_nodes()
        # 2 - cluster BPs based on PO2Vec embeddings
        if self.run_n_dump_po2vec_clustering:
            self._cluster_bps_based_on_po2vec_embeddings(bp_to_po2vec_emb)
        bp_to_po2vec_cluster = self._load_n_validate_po2vec_bp_clustering(bp_to_po2vec_emb)
        # 3 - add PO2Vec-based clusters to BP nodes & dump csv
        self._add_po2vec_clusters_to_bp_nodes_n_dump_csv(bp_to_po2vec_cluster)
    
    def _run_wang(self):
        """_summary_
        """
        # library(GOSemSim)
        # hsGO <- godata(ont="MF")
        # sim <- goSim("GO:0003677", "GO:0005524", semData = hsGO, measure = "Wang")
        # cat(sim)
        # https://yulab-smu.top/biomedical-knowledge-mining-book/

        # 6879	intracellular iron ion homeostasis
        # 6826	iron ion transport
        # 6275	regulation of DNA replication




        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr

        gosms = importr("GOSemSim")
        robjects.r('''
            library(GOSemSim)
            hsGOmf <- godata(ont="MF")
            hsGObp <- godata(ont="BP")
        ''')

        # Example: Calculate GO semantic similarity between two GO terms
        sim = robjects.r('goSim("GO:0003677", "GO:0005524", semData = hsGOmf, measure = "Wang")')
        print(f"Semantic similarity score: {sim[0]}")
        sim1 = robjects.r('goSim("GO:0008150", "GO:0009987", semData = hsGObp, measure = "Wang")')
        print(f"CHAT & CHAT    Semantic similarity score: {sim1[0]}")
        sim2 = robjects.r('goSim("GO:0006879", "GO:0006826", semData = hsGObp, measure = "Wang")')
        print(f"IRON & IRON    Semantic similarity score: {sim2[0]}")
        sim3 = robjects.r('goSim("GO:0006879", "GO:0006275", semData = hsGObp, measure = "Wang")')
        print(f"IRON & DNA replication    Semantic similarity score: {sim3[0]}")

    
    def _add_po2vec_embeddings_to_bp_nodes(self) -> Dict[str, np.ndarray]:
        self.logger.info(f"adding PO2Vec embeddings to BP nodes")
        node_id_to_po2vec_emb = self.go_embeddings_data['po2vec_embeddings']
        
        bp_to_po2vec_emb = {}
        for go_id, po2vec_emb in node_id_to_po2vec_emb.items():
            if self.G.has_node(go_id) and self.G.nodes[go_id]['type'] == self.U.bp:
                self.G = self.U.add_node_property_po2vec_embedding(self.G, go_id, po2vec_emb)
                bp_to_po2vec_emb[go_id] = po2vec_emb
        
        bp_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.bp]
        self.logger.info(f"BP: out of {len(bp_nodes)} nodes, {len(bp_to_po2vec_emb)} have PO2Vec embeddings ({(len(bp_to_po2vec_emb)/len(bp_nodes))*100:.2f}%)")

        return bp_to_po2vec_emb

    def _get_bp_clustering_params_dir_n_nm(self) -> tuple:
        linkage_method = self.bp_clustering_config['linkage_method']    # 'single'
        distance_metric = self.bp_clustering_config['distance_metric']  # 'cosine', 'dice', 'euclidean'
        threshold_dist_prec = self.bp_clustering_config['threshold_distance_percentile']  # 10, 20 ...
        _dir = create_dir_if_not_exists(join(self.config['bp_clustering_dir'], linkage_method))
        f_name = f"bp_clustering_{linkage_method}_{distance_metric}_threshold_percentile_{threshold_dist_prec}"
        return _dir, f_name, linkage_method, distance_metric, threshold_dist_prec
    
    def _cluster_bps_based_on_po2vec_embeddings(self, bp_to_po2vec_emb: Dict[str, np.ndarray]):
        self.logger.info(f"clustering BPs based on PO2Vec embeddings")
        # config params
        _dir, f_name, linkage_method, distance_metric, threshold_dist_prec = self._get_bp_clustering_params_dir_n_nm()
        self.logger.info(f"clustering params - linkage_method: {linkage_method}, distance_metric: {distance_metric}, threshold_dist_prec: {threshold_dist_prec}")

        # perform distance-based hierarchical clustering
        bp_ids = list(bp_to_po2vec_emb.keys())
        embeddings = np.array([bp_to_po2vec_emb[bp] for bp in bp_ids])  # Shape: (num_BPs, embedding_dim)

        distances = pdist(X=embeddings, metric=distance_metric)
        distance_threshold = np.percentile(a=distances, q=threshold_dist_prec)
        self.logger.info(f"distance threshold (at {threshold_dist_prec} percentile): {distance_threshold:.4f}")

        Z = linkage(y=distances, method=linkage_method)
        cluster_labels = fcluster(Z, distance_threshold, criterion='distance')

        # map GO IDs to their cluster labels
        bp_to_cluster = dict(zip(bp_ids, cluster_labels))
        with open(join(_dir, f'{f_name}.pickle'), 'wb') as handle:
            pickle.dump(bp_to_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"PO2Vec-based BP clustering saved to pickle: {join(_dir, f'{f_name}.pickle')}")
    
    def _load_n_validate_po2vec_bp_clustering(self, bp_to_po2vec_emb: Dict[str, np.ndarray]) -> Dict[str, int]:
        # 1 - load clustering from pickle
        _dir, f_name, _, _, _ = self._get_bp_clustering_params_dir_n_nm()
        with open(join(_dir, f'{f_name}.pickle'), 'rb') as handle:
            bp_to_cluster = pickle.load(handle)
        # 2 - validate clustering
        assert sorted(bp_to_cluster.keys()) == sorted(bp_to_po2vec_emb.keys()), "clustering keys do not match BP ids with PO2Vec embeddings"
        assert len(set(bp_to_cluster.values())) > 1, "clustering contains only one cluster"
        self.logger.info(f"loaded and validated PO2Vec-based BP clustering with {len(set(bp_to_cluster.values()))} clusters for {len(bp_to_cluster)} BPs ({f_name})")
        
        return bp_to_cluster

    def _add_po2vec_clusters_to_bp_nodes_n_dump_csv(self, bp_to_po2vec_cluster: Dict[str, int]):
        self.logger.info(f"adding PO2Vec-based clusters to BP nodes")               
        records = []
        for n, d in self.G.nodes(data=True):
            if d['type'] == self.U.bp and n in bp_to_po2vec_cluster:
                bp_cluster = int(bp_to_po2vec_cluster[n])
                self.G = self.U.add_node_property_po2vec_cluster(self.G, n, bp_cluster)
                records.append({'GO_ID': n, 'PO2Vec_cluster': bp_cluster, 'GO_label': d['lbl'], 'GO_definition': d['meta'].get('definition', {}).get('val', '')})     
        df = pd.DataFrame(records).sort_values(by=['PO2Vec_cluster', 'GO_ID'])
        # dump csv
        _dir, f_name, _, _, _ = self._get_bp_clustering_params_dir_n_nm()
        write_df(df, join(_dir, f'summary_{f_name}.csv'))
        self.logger.info(f"dumped summary_{f_name}.csv")

        # """
        # Iterates over all nodes in self.BP, self.MF, and self.CC and add their embeddings vectors.
        # node_id_to_emb: Dict[str, np.ndarray]

        # """
        # self.logger.info(f"adding PO2Vec embeddings to GO nodes")
        # # 1 - add PO2Vec embeddings to GO nodes
        # node_id_to_po2vec_emb = self.go_embeddings_data['po2vec_embeddings']
        # go_ids = list(node_id_to_po2vec_emb.keys())
        # embeddings = np.array([node_id_to_po2vec_emb[go_id] for go_id in go_ids])  # Shape: (num_GO, embedding_dim)
        
        # # Perform hierarchical clustering (use 'euclidean' distance and 'ward' method)
        # linkage_method = self.config.get('linkage_method', 'ward')
        # # Z = linkage(embeddings, method=linkage_method, metric='euclidean')
        # distance_metric = self.config.get('distance_metric', 'cosine')  #  'cosine', 'dice', 'euclidean'
        # distances = pdist(X=embeddings, metric=distance_metric)
        # z = linkage(y=distances, method=linkage_method)
        # # Determine clusters (you can adjust the threshold as needed)
        # threshold = self.config.get('clustering_threshold', 1.0)  # You can set this in your config
        # cluster_labels = fcluster(Z, threshold, criterion='distance')

        # # Map GO IDs to their cluster labels
        # go_id_to_cluster = dict(zip(go_ids, cluster_labels))


    # def _add_po2vec_embeddings_to_go_nodes(self):
    #     """
    #     Iterates over all nodes in self.BP, self.MF, and self.CC and add their embeddings vectors.
    #     node_id_to_emb: Dict[str, np.ndarray]

    #     """
    #     self.logger.info(f"adding PO2Vec embeddings to GO nodes")
    #     # 1 - add PO2Vec embeddings to GO nodes
    #     node_id_to_po2vec_emb = self.go_embeddings_data['po2vec_embeddings']
    #     for go_id, po2vec_emb in node_id_to_po2vec_emb.items():
    #         if self.G.has_node(go_id):
    #             self.G = self.U.add_node_property_po2vec_embedding(self.G, go_id, po2vec_emb)
    #     # 2 - log stats
    #     bp_nodes, bp_nodes_w_emb = 0, 0 
    #     for n, d in self.G.nodes(data=True):
    #         if d['type'] == self.U.bp:
    #             bp_nodes += 1
    #             if self.U.po2vec_emb in d:
    #                 bp_nodes_with_emb += 1
    #     self.logger.info(f"BP: out of {bp_nodes} nodes, {bp_nodes_w_emb} have PO2Vec embeddings ({(bp_nodes_w_emb/bp_nodes)*100:.2f}%)")

    def _add_po2vec_similarity_edges_between_go_nodes(self):
        self.logger.info(f"adding PO2Vec-based similarity edges between GO nodes")

        # emb_type = self.ontology.emb_type_po2vec
        # go_ids_with_emb = [n for n, d in self.G.nodes(data=True) if d['type'] == self.U.bp and emb_type in d]
        # self.logger.info(f"total GO terms with {emb_type}: {len(go_ids_with_emb)}")
        
        # # build KDTree for fast nearest neighbor search
        # go_id_to_emb = {n: self.G.nodes[n][emb_type] for n in go_ids_with_emb}
        # emb_matrix = np.array([go_id_to_emb[n] for n in go_ids_with_emb])
        # from sklearn.neighbors import NearestNeighbors
        # nbrs = NearestNeighbors(n_neighbors=self.config['num_similar_go_terms'] + 1, algorithm='auto').fit(emb_matrix)
        # distances, indices = nbrs.kneighbors(emb_matrix)

        # # add similarity edges
        # for i, go_id in enumerate(go_ids_with_emb):
        #     similar_indices = indices[i][1:]  # skip the first one (itself)
        #     similar_go_ids = [go_ids_with_emb[idx] for idx in similar_indices]
        #     for sim_go_id in similar_go_ids:
        #         # if not self.U.has_similarity_edge(self.G, go_id, sim_go_id):
        #         if not self.U.are_similar_by_po2vec(self.G, curr_go_id, other_go_id):
        #             self.G = self.U.add_edges_po2vec_similarity(self.G, curr_go_id, other_go_id, po2vec_similarity_score)

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
    