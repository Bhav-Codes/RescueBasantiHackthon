# src/coolant_path.py
from typing import Dict, List, Tuple, Optional
import heapq
import math

EPS = 1e-12

def dijkstra_deterministic(nodes: List[Dict], edges: List[Dict], start_id: str, target_id: str) -> Tuple[List[str], float]:
    """
    nodes: [{"id":...}, ...]
    edges: [{"from":u,"to":v,"time":t,...}, ...] (we expect undirected conduits; edges can be added both ways)
    Returns: (path_list, total_time) or ([], inf) if unreachable.
    Tie-breakers:
      1) minimize total time
      2) minimize hops among equal-time paths
      3) deterministic choice: lexicographically smaller predecessors
    """
    node_ids = {n["id"] for n in nodes}
    if start_id not in node_ids or target_id not in node_ids:
        raise ValueError("start/target must be valid node ids")
    # adjacency
    adj = {nid: [] for nid in node_ids}
    for e in edges:
        u, v = e["from"], e["to"]
        t = float(e["time"])
        # add both directions
        adj[u].append((v, t))
        adj[v].append((u, t))
    dist = {nid: math.inf for nid in node_ids}
    hops = {nid: 10**18 for nid in node_ids}
    parent: Dict[str, Optional[str]] = {nid: None for nid in node_ids}
    dist[start_id] = 0.0
    hops[start_id] = 0
    heap = [(0.0, 0, start_id)]
    while heap:
        t_u, h_u, u = heapq.heappop(heap)
        if t_u > dist[u] + EPS: continue
        if abs(t_u - dist[u]) <= EPS and h_u > hops[u]: continue
        if u == target_id:
            break
        for v, w in adj[u]:
            t_v = t_u + w
            h_v = h_u + 1
            if t_v + EPS < dist[v]:
                dist[v] = t_v
                hops[v] = h_v
                parent[v] = u
                heapq.heappush(heap, (t_v, h_v, v))
            elif abs(t_v - dist[v]) <= EPS:
                if h_v < hops[v]:
                    hops[v] = h_v
                    parent[v] = u
                    heapq.heappush(heap, (t_v, h_v, v))
                elif h_v == hops[v]:
                    if parent[v] is None or (u < parent[v]):
                        parent[v] = u
                        heapq.heappush(heap, (t_v, h_v, v))
    if dist[target_id] == math.inf:
        return [], math.inf
    # reconstruct path
    path = []
    cur = target_id
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, dist[target_id]