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
        for _dir in ['interactions_dir', 'rna_dir', 'go_annotations_dir']:
            configs['data_loader'][_dir] = join(configs['runner']['input_data_path'], configs['data_loader'][_dir])
        # ontology
        for _dir in ['gene_ontology_dir']:
            configs['ontology'][_dir] = join(configs['runner']['input_data_path'], configs['ontology'][_dir])
        
        self.configs = configs
    
    def _map_annotations_to_ontology(self, strain, data, ontology):
        self.logger.info(f"mapping annotations to ontology for {strain}")
        if 'all_mrna_w_curated_annot' in data.keys():
            # TODO: 
            # (1) match curated annotations to GO terms ontology (check with are PB, MF and CC)
            # (2) report stats
            self.logger.info(f"    all_mrna_w_curated_annot: {data['all_mrna_w_curated_annot'].shape}")
        if 'all_mrna_w_ips_annot' in data.keys():
            # TODO: 
            # (3) match curated annotations to GO terms ontology (check with are PB, MF and CC)
            # (4) report stats
            self.logger.info(f"    all_mrna_w_ips_annot: {data['all_mrna_w_ips_annot'].shape}")
        # TODO: 
        # (5) compare curated and IPS annotations (optional)
        # (6) start thinking about the analysis (Sahar PPT)
        # (7) describe proprocessing in the latex paper

    def run(self):
        self.logger.info(f"--------------   run starts   --------------")

        data_loader = DataLoader(self.configs['data_loader'], self.logger)
        data_loader.load_and_process_data()
        
        ontology = Ontology(self.configs['ontology'], self.logger)
        ontology.load_go_ontology()
        ontology.create_ontology_nx_graphs()

        for strain, data in data_loader.data.items():
            self._map_annotations_to_ontology(strain, data, ontology)

        
        self.logger.info(f"--------------   run completed   --------------")
        return


if __name__ == "__main__":
    config_path = os.path.join(ROOT_PATH, 'configurations', 'config.json')
    analysis_runner = AnalysisRunner(version='0.0.1', config_path=config_path)
    analysis_runner.run()
