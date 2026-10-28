"""Evidence paths: the Path record built by the PPR retriever and the helpers its context formatter uses."""
from typing import Dict, List, Optional, Tuple


class Path:
    """A path through the graph: its current node, the nodes visited, and the steps taken."""

    def __init__(self, start_node_id: str, is_fact: bool):
        self.current_node = start_node_id
        self.visited_nodes = {start_node_id}
        self.fact_depth = 1 if is_fact else 0
        self.full_depth = 1
        # one (source_id, edge_type, edge_data, target_id) tuple per step
        self.history = []

    def branch(self, target_node_id: str, is_fact: bool, edge_type: str, edge_data: dict) -> "Path":
        """A new Path one step further (this one is left unchanged)."""
        new_path = Path(start_node_id=target_node_id, is_fact=is_fact)
        new_path.visited_nodes = set(self.visited_nodes)
        new_path.visited_nodes.add(target_node_id)
        new_path.fact_depth = self.fact_depth + (1 if is_fact else 0)
        new_path.full_depth = self.full_depth + 1
        new_path.history = list(self.history)
        new_path.history.append((self.current_node, edge_type, edge_data, target_node_id))
        return new_path


def get_path_nodes(path: Path) -> List[str]:
    """The ordered node ids of a path."""
    if not path.history:
        return [path.current_node]
    nodes = [path.history[0][0]]
    for _, _, _, target in path.history:
        nodes.append(target)
    return nodes


def build_edge_lookup(path: Path) -> Dict[Tuple[str, str], Tuple[str, dict]]:
    """(source_id, target_id) -> (edge_type, edge_data) for every step of a path."""
    return {(src, tgt): (edge_type, edge_data) for src, edge_type, edge_data, tgt in path.history}


def extract_bursts(nodes: List[str], edge_cache: set) -> List[List[str]]:
    """The contiguous runs of a node sequence whose edges are not in edge_cache. Each new edge is added to
    edge_cache as soon as it is seen, so an edge repeated within the same path is written once too."""
    bursts: List[List[str]] = []
    current_burst: Optional[List[str]] = None
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if (a, b) not in edge_cache and (b, a) not in edge_cache:
            if current_burst is None:
                current_burst = [a, b]
            else:
                current_burst.append(b)
            edge_cache.add((a, b))
            edge_cache.add((b, a))
        elif current_burst is not None:
            bursts.append(current_burst)
            current_burst = None
    if current_burst is not None:
        bursts.append(current_burst)
    return bursts
