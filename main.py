from typing import Dict
import pandas as pd
import numpy as np
import random
from os.path import join, split, dirname, basename, abspath
import json
import sys
import os
from analysis.data_loader import DataLoader
from analysis.ontology import Ontology
from analysis.graph_utils import GraphUtils
from analysis.graph_builder import GraphBuilder
from analysis.analyzer import Analyzer
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# from utils.logger import create_logger


ROOT_PATH = os.path.realpath(os.path.dirname(__file__))  # the root of the repository
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")


class AnalysisRunner:
    def __init__(self, version: str, config_path: str):
        print(f"running main, version = {version}")
        # self.logger = create_logger(logger_nm='AnalysisRunner')
        self.logger = logger
        self.logger.info(f"initializing AnalysisRunner, version = {version}")
        self.version = version

        # load & update configurations on-the-fly
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        # runner
        configs['runner']['input_data_path'] = configs['runner']['remote_data_path'] if 'shanisa' in ROOT_PATH else configs['runner']['local_data_path']
        configs['runner']['output_data_path'] = join(configs['runner']['input_data_path'], 'outputs')
        # data loader
        for _dir in ['interactions_dir', 'rna_dir', 'proteins_dir', 'go_annotations_dir', 'go_embeddings_dir', 'clustering_dir']:
            configs['data_loader'][_dir] = join(configs['runner']['input_data_path'], configs['data_loader'][_dir])
        # ontology
        for _dir in ['gene_ontology_dir']:
            configs['ontology'][_dir] = join(configs['runner']['input_data_path'], configs['ontology'][_dir])
        # graph utils
        # graph builder
        for _dir in ['builder_output_dir']:
            configs['graph_builder'][_dir] = join(configs['runner']['output_data_path'], configs['graph_builder'][_dir])
        # analyzer
        for _dir in ['analysis_output_dir']:
            configs['analyzer'][_dir] = join(configs['runner']['output_data_path'], configs['analyzer'][_dir])
        self.configs = configs

    def run(self):
        self.logger.info(f"--------------   run starts   --------------")

        data_loader = DataLoader(self.configs['data_loader'], self.logger)
        data_loader.load_and_process_data()
        
        ontology = Ontology(self.configs['ontology'], self.logger)
        ontology.load_go_ontology()
        ontology.create_ontology_nx_graphs()

        graph_utils = GraphUtils(self.configs['graph_utils'], self.logger, data_loader, ontology)

        graph_builder = GraphBuilder(self.configs['graph_builder'], self.logger, data_loader, ontology, graph_utils)
        graph_builder.build_graph()

        analyzer = Analyzer(self.configs['analyzer'], self.logger, graph_builder, graph_utils)
        analyzer.run_analysis()
        # Ad-hoc outputs
        # analyzer._run_ad_hoc_outputs_analysis_1(srnas_cluster = ('E2348C_ncR46', 'G0-8867', 'GcvB', 'gcvB', 'ncRNA0016'), 
        #                                         bp_str = "0006865__amino acid transport")
        
        self.logger.info(f"--------------   run completed   --------------")


if __name__ == "__main__":
    config_path = os.path.join(ROOT_PATH, 'configurations', 'config.json')
    analysis_runner = AnalysisRunner(version='0.0.1', config_path=config_path)
    analysis_runner.run()
