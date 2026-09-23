#!/usr/bin/env python3
"""
Generate a Cytoscape.js-compatible graph-data.json file from a Python dict
that maps sRNAs -> mRNAs -> BPs, with optional clusters for any of
the three node types.
"""

import colorsys
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

BpValue = Union[str, List[str]]
MrnaMap = Dict[str, BpValue]
SrnaMap = Dict[str, MrnaMap]
ClusterMap = Dict[str, List[str]]

NODE_TYPES = ("sRNA", "mRNA", "BP")


def _as_list(value: BpValue) -> List[str]:
    """Normalize a value that may be a single string or a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _tree_to_srna_map(tree: Dict[str, Any]) -> SrnaMap:
        srna_map: SrnaMap = {}
        for srna, srna_data in tree.items():
            _strain_nm = srna.split('__')[0]
            _srna = f"{_strain_nm}__{srna.split('__')[2]}"
            _mrna_to_bps = {}
            for mrna, mrna_data in srna_data['targets_info'].items():
                _mrna = f"{_strain_nm}__{mrna.split('__')[1]}"
                _bps = [bp[2] for bp in mrna_data['BP_info']]
                assert len(_bps) == len(set(_bps)), f"Duplicate BPs found for {srna} -> {mrna}: {_bps}"
                _mrna_to_bps[_mrna] = _bps
            srna_map[_srna] = _mrna_to_bps
        return srna_map


def _cluster_list_to_map(clusters: List[Tuple[str, ...]], prefix: str) -> ClusterMap:
    """Convert an ordered list of cluster members to the graph cluster map shape."""
    return {
        f"{prefix}-{index}": list(members)
        for index, members in enumerate(clusters, start=1)
    }


def _bp_cluster_list_to_map(clusters: List[Tuple[Any, Any, str]], prefix: str) -> ClusterMap:
    """Group multi-label BP clusters from (cluster ID, BP ID, BP label) records.

    Clusters containing only one label are omitted because they do not define
    a shared BP cluster.
    """
    cluster_to_labels: Dict[Any, List[str]] = {}
    for cluster_id, _, label in clusters:
        cluster_to_labels.setdefault(cluster_id, []).append(label)

    cluster_map: ClusterMap = {}
    for cluster_id, labels in cluster_to_labels.items():
        if len(labels) > 1:
            cluster_map[f"{prefix}-{len(cluster_map) + 1}"] = labels
    return cluster_map


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """h in [0, 360), s and l in [0, 1] -> '#rrggbb'."""
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def _generate_cluster_colors(
    n: int, saturation: float = 0.65, lightness: float = 0.5
) -> List[str]:
    """
    Generate `n` visually distinct colors by spacing hues around the color
    wheel using the golden angle, so consecutive colors never look similar
    no matter how many are requested.
    """
    golden_angle = 137.508
    return [_hsl_to_hex((i * golden_angle) % 360, saturation, lightness) for i in range(n)]


def _assign_cluster_colors_globally(
    ntype_to_cluster_map: Dict[str, ClusterMap]
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    Build label -> (cluster_name, color) per node type, drawing every
    cluster's color from a single shared sequence so colors never repeat
    across node types (e.g. an sRNA cluster and a BP cluster can never end
    up the same color).

    Returns: {node_type: {member_label: (cluster_name, color)}}
    """
    # Collect (node_type, cluster_name) pairs in a stable, deterministic
    # order across all three types, then hand out one color per pair.
    all_cluster_keys: List[Tuple[str, str]] = []
    for ntype in NODE_TYPES:
        for cluster_name in ntype_to_cluster_map.get(ntype, {}):
            all_cluster_keys.append((ntype, cluster_name))

    colors = _generate_cluster_colors(len(all_cluster_keys))
    key_to_color = dict(zip(all_cluster_keys, colors))

    lookup: Dict[str, Dict[str, Tuple[str, str]]] = {t: {} for t in NODE_TYPES}
    for ntype in NODE_TYPES:
        for cluster_name, members in ntype_to_cluster_map.get(ntype, {}).items():
            color = key_to_color[(ntype, cluster_name)]
            for member_label in members:
                if member_label in lookup[ntype]:
                    print(
                        f"Warning: '{member_label}' is listed in multiple {ntype} "
                        f"clusters; keeping '{lookup[ntype][member_label][0]}', "
                        f"ignoring '{cluster_name}'."
                    )
                    continue
                lookup[ntype][member_label] = (cluster_name, color)
    return lookup


def _build_graph(
    srna_map: SrnaMap,
    srna_clusters: Optional[ClusterMap] = None,
    mrna_clusters: Optional[ClusterMap] = None,
    bp_clusters: Optional[ClusterMap] = None) -> Dict[str, Any]:

    node_id: Dict[str, str] = {}   # label -> node id (e.g. "checkout-sRNA" -> "n0")
    node_type: Dict[str, str] = {} # label -> type
    nodes: List[Dict[str, Any]] = [] # each dict in the list represents a node -> {id, label, type, cluster?, color?}
    edges: List[Dict[str, Any]] = []
    edge_keys = set()

    cluster_lookup = _assign_cluster_colors_globally({
        "sRNA": srna_clusters or {},
        "mRNA": mrna_clusters or {},
        "BP": bp_clusters or {},
    })

    def get_or_create_node(label: str, ntype: str) -> str:
        if label in node_id:
            return node_id[label]
        nid = f"n{len(nodes)}"
        node_id[label] = nid
        node_type[label] = ntype

        data: Dict[str, Any] = {"id": nid, "label": label, "type": ntype}
        lookup = cluster_lookup.get(ntype, {})
        if label in lookup:  # if this node belongs to a cluster, add cluster name and color
            cluster_name, color = lookup[label]
            data["cluster"] = cluster_name
            data["color"] = color

        nodes.append({"data": data})
        return nid

    def add_edge(source_id: str, target_id: str, edge_type: str) -> None:

        edge_key = (source_id, target_id, edge_type)
        if edge_key in edge_keys:
            return
        edge_keys.add(edge_key)

        eid = f"e{len(edges)}"
        edges.append({
            "data": {
                "id": eid,
                "source": source_id,
                "target": target_id,
                "edgeType": edge_type,
            }
        })

    for srna_label, mrnas_to_bps in srna_map.items():
        srna_id = get_or_create_node(srna_label, "sRNA")
        for mrna_label, bp_value in mrnas_to_bps.items():
            mrna_id = get_or_create_node(mrna_label, "mRNA")
            add_edge(srna_id, mrna_id, "interacts with")

            for bp_label in _as_list(bp_value):
                bp_id = get_or_create_node(bp_label, "BP")
                add_edge(mrna_id, bp_id, "annotated")

    return {"nodes": nodes, "edges": edges}


def adjust_data_for_cytoscape_graph_viewer(tree: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
    """_summary_

    Args:
        tree (Dict[str, Any]): _description_
        row (pd.Series): _description_

    Returns:
        Dict[str, Any]: _description_
    """

    if row['srna_subgroup'] == ('ecoli_k12__EG30098__spf', 'salmonella__ncRNA0077__spf', 'vibrio__Spot42__spot42'):
        print(row) #TODO: remove
    
    srna_map: SrnaMap = _tree_to_srna_map(tree)
    srna_clusters: ClusterMap = {
        "sRNA-cluster-1": [f"{x.split('__')[0]}__{x.split('__')[2]}" for x in row['srna_subgroup']],
    }
    mrna_clusters: ClusterMap = _cluster_list_to_map(row['homolog_clusters_of_targets_annotated_with_shared_bp_clusters'], "mRNA-cluster")
    bp_clusters: ClusterMap = _bp_cluster_list_to_map(row['shared_bp_clusters'], "BP-cluster")
    
    graph_data = _build_graph(srna_map, srna_clusters, mrna_clusters, bp_clusters)
    return graph_data
