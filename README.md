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
- The standard `Pipeline.run()` is:
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

## Cytoscape Graph Viewer

Interactive viewer for the sRNA → mRNA → BP regulatory graph, built with
[Cytoscape.js](https://js.cytoscape.org/). Nodes are arranged in three
columns by type (sRNA → mRNA → BP-terms), colored by cluster where applicable,
and edges are styled by relationship type.

<p align="center">
	<img src="/docs/Cytoscape_example.png" width="800" />
</p>

### Folder contents

```
visualization/cytoscape_graph_viewer/
├── index.html        # the viewer (this is what you open)
└── graph-data.json   # the graph data (must sit next to index.html)
```

Both files **must be in the same folder**. `index.html` loads
`graph-data.json` via a relative path (`fetch('graph-data.json')`), so if
the JSON file is moved, renamed, or placed elsewhere, the page will fail
to load the graph.

**Example.** For each sRNA subgroup, the analysis pipeline exports a
dedicated JSON file for Cytoscape visualization, named
`sRNA-to-BP__Mappings__Cluster_<N>__Subgroup_<M>__cytoscape.json` (the
trailing `cytoscape` marks it as the Cytoscape-ready export). To visualize the
subgroup containing sRNA homologs `ecoli_epec__E2348C_ncR06__chix`,
`ecoli_k12__G0-9382__chix`, `klebsiella__chiX__chix`, and
`salmonella__ncRNA0003__chix` (Cluster 16, Subgroup 1), take the
corresponding export file, `sRNA-to-BP__Mappings__Cluster_16__Subgroup_1__cytoscape.json`,
rename it to `graph-data.json`, and place it in
`visualization/cytoscape_graph_viewer/` alongside `index.html`.

### Option 1 — VS Code integrated browser (Live Preview)

The quickest option if you're already working in VS Code.

1. Install the **Live Preview** extension (publisher: Microsoft) from
   the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`), if not already
   installed.
2. In the file Explorer sidebar, right-click `index.html`.
3. Select **"Show Preview"**.
4. The graph opens in a browser tab *inside* the VS Code window, served
   over a local `http://` address that Live Preview manages
   automatically — no terminal commands needed.

> If you don't see "Show Preview" in the right-click menu, the extension
> may not be installed yet, or you may be looking at the "Open with Live
> Server" option from a different extension instead — see Option 2 if
> so, since that one launches an external browser tab rather than an
> integrated one.

### Option 2 — Run a local server from the terminal

Works regardless of editor, and is the most reliable fallback.

1. Open a terminal and navigate to this folder:
   ```bash
   cd path/to/visualization/cytoscape_graph_viewer
   ```
2. Start a local server. Pick whichever tool you already have installed:

   **Python (most common):**
   ```bash
   python3 -m http.server 8000 --bind 127.0.0.1
   ```
   *(On Windows, use `python` or `py` instead of `python3` if that's
   what resolves on your system.)*

   **Node.js:**
   ```bash
   npx serve .
   ```

   **PHP:**
   ```bash
   php -S localhost:8000
   ```
3. Open the printed address in your browser, e.g.:
   ```
   http://localhost:8000/index.html
   ```
4. Stop the server with `Ctrl+C` in the terminal when you're done.

### Reading the graph

- **Shape = node type** — circle (sRNA), triangle (mRNA), square (BP),
  arranged left-to-right in that order.
- **Color = cluster membership** — nodes belonging to a homolog cluster
  (sRNA/mRNA) or a shared BP-cluster are colored to match; ungrouped
  nodes stay gray. A legend listing each cluster's color is built
  automatically on the right, based on whatever `cluster`/`color`
  values are present in `graph-data.json`.
- **Line style = edge type** — solid lines are "interacts with," dashed
  lines are "annotated," per the legend.
- **Click a node** to fade everything except its immediate neighborhood
  and highlight its edges in red. **Click empty canvas** to reset.

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| Stuck on "Loading graph…" past ~8 seconds | You opened the file directly (`file://`) instead of via a server — use Option 1 or 2 above. The page will also show a diagnostic message after 8s. |
| "Failed to load graph: HTTP 404" | `graph-data.json` isn't in the same folder as `index.html`, or is misnamed. |
| Graph loads but looks unstyled/plain | Check your internet connection — `index.html` loads the Cytoscape.js library itself from a CDN (`cdnjs.cloudflare.com`), so it needs network access even when served locally. |
| Nothing happens on click | Make sure you're clicking directly on a node (not just near it) — Cytoscape's hit detection is based on the actual rendered shape. |

---

Generated on: September 24, 2026
