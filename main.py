from typing import Dict
import pandas as pd
import numpy as np
import random
from os.path import join, split, dirname, basename, abspath
import json
import sys
import os
# from analysis import data_loader as dl
from analysis.data_loader import DataLoader
# import logging
# logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
from utils.logger import create_logger


ROOT_PATH = os.path.realpath(os.path.dirname(__file__))  # the root of the repository
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")


class AnalysisRunner:
    def __init__(self, version: str, config_path: str):
        print(f"running main, version = {version}")
        self.logger = create_logger(logger_nm='AnalysisRunner')
        self.logger.info(f"initializing AnalysisRunner, version = {version}")
        self.version = version
        self.config_path = config_path

        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.config = configs['runner']
        
        data_path = self.config['remote_data_path'] if 'shanisa' in ROOT_PATH else self.config['local_data_path']
        # update data_loader config on-the-fly
        self.data_loader_config = configs['data_loader']
        self.data_loader_config['input_data_path'] = data_path
        self.data_loader_config['output_data_path'] = join(data_path, 'outputs')
        
    def run(self):
        self.logger.info(f"--------------   run starts   --------------")

        data_loader = DataLoader(self.data_loader_config, self.logger)
        data = data_loader.load_and_process_data()
        # # 1 - set random seeds
        # init_seed()

        # # 2 - set version and config
        # version = '0.0.1'
        # logger.info(f"running main, version = {version}")
        # conf = get_config(name="config")
        # output_data_path = join(conf['output_data_path'][conf['machine']], version)

        # # 3 - load + preprocess data
        # ecoli_k12_nm, ecoli_epec_nm, salmonella_nm = 'ecoli_k12', 'ecoli_epec', 'salmonella'
        # data = load_and_process_data(conf=conf, ecoli_k12_nm=ecoli_k12_nm, ecoli_epec_nm=ecoli_epec_nm, salmonella_nm=salmonella_nm)
        
        self.logger.info(f"--------------   run completed   --------------")
        return


if __name__ == "__main__":
    config_path = os.path.join(ROOT_PATH, 'configurations', 'config.json')
    analysis_runner = AnalysisRunner(version='0.0.1', config_path=config_path)
    analysis_runner.run()
