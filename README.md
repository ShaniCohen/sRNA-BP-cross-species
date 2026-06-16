# Comprehensive pipeline for cross-species analysis of sRNA-mediated regulation.

This repository is the official implementation of the paper 
[*"A multi-layer, multi-species graph-based framework for comparative analysis of sRNA-mediated regulation"*]().

<p align="center">
	<img src="/docs/Pipeline.png" width="800" />
</p>


## Overview

This repository provides a modular pipeline for integrating experimentally derived sRNA–mRNA interactions, orthology, functional annotations, and GO biological process clustering. It supports two analyses: (1) functional conservation among orthologous sRNAs and (2) convergent regulation by distinct sRNAs across species, uncovering signals that emerge only through multi-layer, cross-species integration.
The codebase is organized into reusable modules, allowing flexible execution and easy extension to new datasets or regulatory layers, enabling scalable, hypothesis-driven studies of post-transcriptional regulatory network evolution.

## Installation

- Clone the repository.
- Create a Conda environment using `environment.yml` (this will also install the dependencies listed in `requirements.txt`).
- Run the top-level runner

```bash
conda env create -f environment.yml
conda activate pipeline-env
python main.py
```

## Modules

- **Configurations**: [configurations/config.json](configurations/config.json)
	- Centralize runtime settings, input paths, and parameter defaults.

- **Top-level runner**: [main.py](main.py)
	- Orchestrates end-to-end pipeline runs and high-level configuration.
	- Example usage:

```py
if __name__ == "__main__":
		config_path = os.path.join(ROOT_PATH, 'configurations', 'config.json')
		pipeline = Pipeline(config_path=config_path)
		pipeline.run()
```

- **Data loading**: [analysis/data_loader.py](analysis/data_loader.py)
	- Read and preprocess per-strain RNA and interaction files, load protein mappings, GO annotations, embeddings (optional) and clustering inputs.
	- Inputs: raw CSVs and annotation files located under paths in `configs['data_loader']`.
	- Outputs: `DataLoader.strains_data` — a dict of per-strain dataframes (e.g. `all_mrna`, `all_srna`, `all_inter`) and convenience column names (`all_mrna_acc_col`).
	- Example usage:

```py
data_loader = DataLoader(configs['data_loader'], logger)
data_loader.load_and_process_data()
# use data_loader.strains_data[...] in downstream modules
```

- **Ontology**: [analysis/ontology.py](analysis/ontology.py)
	- Parse the Gene Ontology JSON, remove deprecated terms, map ontology properties and build NetworkX graphs for BP/MF/CC.
	- Inputs: GO JSON file path set in `configs['ontology']`.
	- Outputs: `Ontology.BP`, `Ontology.MF`, `Ontology.CC` — NetworkX graphs with node/edge attributes; helper maps such as `property_id_to_info`.
	- Example usage:

```py
ontology = Ontology(configs['ontology'], logger)
ontology.load_go_ontology()
ontology.create_ontology_nx_graphs()
# ontology.BP is ready for graph construction
```

- **Graph utilities**: [analysis/graph_utils.py](analysis/graph_utils.py)
	- Small helpers and validation functions for graph operations (node/edge types, adding RNA nodes/edges, querying nodes by strain, orthology/paralogy checks, PO2Vec helpers).
	- Inputs: an already-built NetworkX graph and module-level constants from `GraphUtils` (node/edge type strings).
	- Typical methods:
		- `add_node_rna(...)`, `add_edge_srna_mrna_inter(...)`, `add_edge_mrna_go_annot(...)`
		- `get_all_srna_nodes(G, strain)`, `get_all_mrna_nodes(G, strain)`, `are_paralogs(...)`, `are_orthologs_by_seq(...)`
	- Example usage inside GraphBuilder/Analyzer:

```py
U = GraphUtils(configs['graph_utils'], logger, data_loader, ontology)
U.add_node_rna(G, id, U.srna, strain, locus_tag, name, synonyms, start, end, strand, seq)
```

- **Graph construction**: [analysis/graph_builder.py](analysis/graph_builder.py)
	- Build the multi-layer NetworkX graph integrating GO terms (BP nodes), mRNA nodes, sRNA nodes and sRNA→mRNA interaction edges, mRNA annotation edges, and homology edges (orthologs/paralogs; sequence-based/name-based). Also supports adding PO2Vec embeddings and clustering of BPs.
	- Inputs: `DataLoader.strains_data`, `Ontology.BP` graphs, configuration under `configs['graph_builder']`.
	- Outputs: `GraphBuilder.G` (a `nx.MultiDiGraph`) with properly typed nodes and edges; CSV dumps under `builder_output_dir` when enabled.
	- Example usage:

```py
graph_builder = GraphBuilder(configs['graph_builder'], logger, data_loader, ontology, graph_utils)
graph_builder.build_graph()
G = graph_builder.get_graph()
```

- **Analysis**: [analysis/analyzer.py](analysis/analyzer.py)
	- Run multi-step analyses on the constructed graph — cluster RNA homologs/paralogs, compute BP↔RNA mappings, compute subgroup-level statistics, run enrichment tests, and produce output tables and visualizations.
	- Inputs: `GraphBuilder.get_graph()`, clustering parameters under `configs['analyzer']`, and optional random-graph seed for p-value estimation.
	- Outputs: CSV summary tables and per-tool result directories under `analysis_output_dir` (e.g. `Analysis_tool_1_sRNA_to_BP`, `Analysis_tool_2_BP_to_sRNA`, clustering trees, etc.).
	- Example usage:

```py
analyzer = Analyzer(configs['analyzer'], logger, graph_builder, graph_utils, random_seed=None)
analyzer.run_analysis()
```

Notes:
- All components read their settings from `configurations/config.json`. Update this file to control behavior (such as enabling enrichment or selecting the clustering linkage method).
- The set of strains to analyze is defined within the `DataLoader` module.
- The standard pipeline is:
```py
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
```

## Data

The complete input and output datasets are available at [Zenodo](https://zenodo.org/records/20718221).

---

Generated on: June 11, 2026
