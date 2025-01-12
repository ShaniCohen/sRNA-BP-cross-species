from typing import Dict
import pandas as pd
import numpy as np
import random
from os.path import join, split, dirname, basename, abspath
import json
import sys
import os
from analysis.data_loading import data_load as dl
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# def networkx_example():
#     import networkx as nx

#     # Enable nx-cugraph backend
#     import os
#     os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"

#     # Create a graph and run an algorithm
#     G = nx.erdos_renyi_graph(1000, 0.01)
#     centrality = nx.betweenness_centrality(G)


def get_config(name, add_machine_to_conf=True):
    repo_path = dirname(abspath(__file__))
    conf_path = join(repo_path, 'configurations', f'{name}.json')
    with open(conf_path, 'r', encoding='utf-8') as f:
        _config = json.load(f)
    if add_machine_to_conf:
        _config['machine'] = 'cluster' if sys.platform == 'linux' else os.environ['COMPUTERNAME']

    return _config


def init_seed(seed: int = 2):
    np.random.seed(seed)
    random.seed(seed)
    return


def load_and_process_data(conf: Dict[str, str], ecoli_k12_nm: str, ecoli_epec_nm: str, salmonella_nm: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    # 1 - RNA and interactions data
    data = dl.load_rna_n_inter_data(conf, ecoli_k12_nm, ecoli_epec_nm, salmonella_nm)
    data = dl.align_rna_n_inter_data(data)
    # 2 - Gene Ontology data

    return data


def main():
    # 1 - set random seeds
    init_seed()

    # 2 - set version and config
    version = '0.0.1'
    logger.info(f"running main, version = {version}")
    conf = get_config(name="config")
    output_data_path = join(conf['output_data_path'][conf['machine']], version)

    # 3 - load + preprocess data
    ecoli_k12_nm, ecoli_epec_nm, salmonella_nm = 'ecoli_k12', 'ecoli_epec', 'salmonella'
    data = load_and_process_data(conf=conf, ecoli_k12_nm=ecoli_k12_nm, ecoli_epec_nm=ecoli_epec_nm, salmonella_nm=salmonella_nm)
    
    logger.info(f"--------------   done main   --------------")
    return


if __name__ == "__main__":
    main()
