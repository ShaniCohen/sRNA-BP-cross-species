from os.path import join
import json
import sys
import os
import pandas as pd
from typing import List
from analysis.data_loader import DataLoader
from analysis.ontology import Ontology
from analysis.graph_utils import GraphUtils
from analysis.graph_builder import GraphBuilder
from analysis.analyzer import Analyzer
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


ROOT_PATH = os.path.realpath(os.path.dirname(__file__))  # the root of the repository
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")


class Pipeline:
    def __init__(self, version: str, config_path: str):
        self.logger = logger
        self.logger.info(f"initializing Pipeline, version = {version}")
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
        #   input dirs
        for _dir in ['bp_clustering_dir']:
            configs['analyzer'][_dir] = join(configs['runner']['input_data_path'], configs['analyzer'][_dir])
        #   output dirs
        for _dir in ['analysis_output_dir']:
            configs['analyzer'][_dir] = join(configs['runner']['output_data_path'], configs['analyzer'][_dir])
        #   add ontology info
        configs['analyzer'].update(configs['ontology'])

        self.configs = configs
    
    def run(self, random_graph_seed: int = None):
        self.logger.info(f"--------------   run starts   --------------")
        if random_graph_seed is not None:
            self.logger.info(f"*****  Running analysis on a RANDOM graph (seed = {random_graph_seed})  *****")

        data_loader = DataLoader(self.configs['data_loader'], self.logger)
        data_loader.load_and_process_data()
        
        ontology = Ontology(self.configs['ontology'], self.logger)
        ontology.load_go_ontology()
        ontology.create_ontology_nx_graphs()

        graph_utils = GraphUtils(self.configs['graph_utils'], self.logger, data_loader, ontology)

        graph_builder = GraphBuilder(self.configs['graph_builder'], self.logger, data_loader, ontology, graph_utils)
        graph_builder.build_graph()

        analyzer = Analyzer(self.configs['analyzer'], self.logger, graph_builder, graph_utils, random_graph_seed)
        analyzer.run_analysis()
        
        self.logger.info(f"--------------   run completed   --------------")

    def run_p_value_calculation(self, dir_original_graph: str = None, dir_random_graphs: str = None, conf_str: str = 'k12_curated_ips', seeds: List[int] = list(range(1, 1001))):
        """_summary_

        Args:
            conf_str (str, optional): Can be either 'k12_curated_ips' or 'k12_curated_ips_w_Enrichment'.
        """
        self.logger.info(f"--------------   p-value calculation starts   --------------")
        dir_original_graph = dir_original_graph if dir_original_graph else join(self.configs['analyzer']['analysis_output_dir'], conf_str, 'Analysis_tool_1_sRNA_to_BP')
        dir_random_graphs = dir_random_graphs if dir_random_graphs else join(self.configs['analyzer']['analysis_output_dir'], conf_str, 'Random_graphs')
        
        original_res = pd.read_csv(join(dir_original_graph, f'sRNA-to-BP__Output__{conf_str}.csv'))
        for seed in seeds:
            dir_seed = join(dir_random_graphs, f'seed_{seed}', 'Analysis_tool_1_sRNA_to_BP')
            random_res = pd.read_csv(join(dir_seed, f'sRNA-to-BP__Output__{conf_str}.csv'))
            
            merged = original_res.merge(random_res, on=['sRNA', 'BP'], suffixes=('_original', '_random'))
            merged['p_value'] = merged.apply(lambda row: 1 if row['p_value_random'] <= row['p_value_original'] else 0, axis=1)
            merged.to_csv(join(dir_random_graphs, f'sRNA-to-BP__Output__v_random_graph_seed_{seed}__{conf_str}_with_p_values.csv'), index=False)
        self.logger.info(f"--------------   p-value calculation completed   --------------")


if __name__ == "__main__":
    ##### Random Graph Mode (for p-value calculation):  
    #   set a seed to get analysis results over a RANDOM graph
    #   otherwise results are provided for the real graph (seed=None)
    seed = int(os.getenv('SLURM_ARRAY_TASK_ID')) if os.getenv('SLURM_ARRAY_TASK_ID') else None
    # seed = seed + 500 if seed is not None else None

    config_path = os.path.join(ROOT_PATH, 'configurations', 'config.json')
    pipeline = Pipeline(version='0.0.1', config_path=config_path)
    # pipeline.run(random_graph_seed=seed)
    pipeline.run_p_value_calculation()

    print("done")


    # file_path = join(pipeline.configs['analyzer']['analysis_output_dir'], 'k12_curated_ips', 'Analysis_tool_1', 'sRNA-to-BP__Output__v_k12_curated_ips.csv')
    # import pandas as pd
    # df = pd.read_csv(file_path, encoding='utf-8')
    # pipeline.run()
