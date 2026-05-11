from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from collections import deque
import uuid
import functools
from langgraph.graph import StateGraph, START, END
from .contract import Contract, NodeMeta, Edge, LLMCall
from . import decorators as D
from .rid import *


@dataclass
class _NodeRecord:
    name: str
    fn: Any


@dataclass
class _ConditionalEdgeAnnotation:
    src: str
    destinations: List[List[str]]
    frontiers: Optional[List[List[str]]]


class GraphProxy:
    """
    A thin wrapper that mirrors the StateGraph API we care about and
    records node/edge structure while you build the graph.

    Usage:
        sg = StateGraph(DocState)
        g = GraphProxy(sg)
        g.add_node("router", router)
        g.add_node("write", write)
        g.add_edge("router", "write")
        g.set_entry_point("router")
        g.set_finish_point("write")
        CONTRACT = g.build_contract()
        graph = g.materialize()   # returns the original StateGraph
    """
    def __init__(self, state_graph: StateGraph):
        self._g = state_graph
        self._nodes: Dict[str, _NodeRecord] = {'START': _NodeRecord(name='START', fn=None), 'END': _NodeRecord(name='END', fn=None)}
        self._edges: List[Edge] = []
        self._conditional_edge_annotations: List[_ConditionalEdgeAnnotation] = []
        self._contract_edge_entries: List[Any] = []
        self._entry_point: Optional[str] = None

    def _record_edge(self, edge: Edge):
        self._edges.append(edge)
        self._contract_edge_entries.append(edge)

    def _record_conditional_annotation(self,
                                       annotation: _ConditionalEdgeAnnotation):
        self._conditional_edge_annotations.append(annotation)
        self._contract_edge_entries.append(annotation)

    def _remove_recorded_edge(self, edge: Edge):
        for index, existing_edge in enumerate(self._edges):
            if existing_edge is edge:
                del self._edges[index]
                break
        for index, entry in enumerate(self._contract_edge_entries):
            if entry is edge:
                del self._contract_edge_entries[index]
                break

    def _copy_grouped_node_names(self,
                                 groups: List[List[str]],
                                 field_name: str,
                                 *,
                                 allow_empty_groups: bool = False) -> List[List[str]]:
        if not groups:
            raise ValueError(f"{field_name} must contain at least one group")

        normalized_groups: List[List[str]] = []
        for group in groups:
            if not group and not allow_empty_groups:
                raise ValueError(f"{field_name} groups cannot be empty")
            normalized_group: List[str] = []
            for node_name in group:
                if not isinstance(node_name, str):
                    raise ValueError(
                        f"{field_name} entries must be node names (strings)")
                normalized_group.append(node_name)
            normalized_groups.append(normalized_group)
        return normalized_groups

    def _resolve_contract_edges(self) -> List[Edge]:
        resolved_edges: List[Edge] = []
        for entry in self._contract_edge_entries:
            if isinstance(entry, Edge):
                resolved_edges.append(entry)
                continue

            annotation = entry
            for group_index, destinations in enumerate(annotation.destinations):
                label = f"conditional_{group_index}"
                for dst in destinations:
                    resolved_edges.append(
                        Edge(src=annotation.src, dst=dst, label=label))
        return resolved_edges

    def _dedupe_edges(self, edges: List[Edge]) -> List[Edge]:
        seen_edges = set()
        unique_edges: List[Edge] = []
        for edge in edges:
            edge_tuple = (edge.src, edge.dst, edge.label)
            if edge_tuple in seen_edges:
                continue
            seen_edges.add(edge_tuple)
            unique_edges.append(edge)
        return unique_edges

    def _build_adjacency(self, edges: List[Edge]) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.src, [])
            if edge.dst not in adjacency[edge.src]:
                adjacency[edge.src].append(edge.dst)
        return adjacency

    def _validate_conditional_annotation_nodes(
            self, annotation: _ConditionalEdgeAnnotation,
            contract_nodes: Dict[str, NodeMeta]):
        if annotation.src not in contract_nodes:
            raise ValueError(
                f"annotate_conditional_edge source '{annotation.src}' was not added to the graph")

        for group in annotation.destinations:
            for node_name in group:
                if node_name not in contract_nodes:
                    raise ValueError(
                        f"annotate_conditional_edge destination '{node_name}' was not added to the graph")

        if annotation.frontiers is None:
            return

        for group in annotation.frontiers:
            for node_name in group:
                if node_name not in contract_nodes:
                    raise ValueError(
                        f"annotate_conditional_edge frontier '{node_name}' was not added to the graph")

    def _compute_non_llm_reachability_owners(
            self, destinations: List[List[str]], contract_nodes: Dict[str, NodeMeta],
            adjacency: Dict[str, List[str]]) -> Dict[str, Set[int]]:
        owners: Dict[str, Set[int]] = {}

        for group_index, dest_group in enumerate(destinations):
            queue = deque(dest_group)
            seen: Set[str] = set()

            while queue:
                node_name = queue.popleft()
                if node_name in seen:
                    continue
                seen.add(node_name)

                node = contract_nodes[node_name]
                if node.llm_calls:
                    continue

                owners.setdefault(node_name, set()).add(group_index)
                for child_name in adjacency.get(node_name, []):
                    queue.append(child_name)

        return owners

    def _infer_first_llm_frontier(
            self, group_index: int, destinations: List[str],
            contract_nodes: Dict[str, NodeMeta], adjacency: Dict[str, List[str]],
            non_llm_reachability_owners: Dict[str, Set[int]]) -> List[str]:
        frontier: List[str] = []
        queue = deque(destinations)
        seen: Set[str] = set()

        while queue:
            node_name = queue.popleft()
            if node_name in seen:
                continue
            seen.add(node_name)

            node = contract_nodes[node_name]
            if node.llm_calls:
                if node_name not in frontier:
                    frontier.append(node_name)
                continue

            owners = non_llm_reachability_owners.get(node_name, set())
            if owners and owners != {group_index}:
                # If mutually-exclusive branches merge before any LLM node, the
                # first downstream LLM is no longer uniquely attributable to one
                # branch. Leave that case to the explicit frontier override.
                continue

            for child_name in adjacency.get(node_name, []):
                queue.append(child_name)

        return frontier

    def _apply_conditional_edge_annotations(
            self, contract_nodes: Dict[str, NodeMeta], edges: List[Edge]):
        adjacency = self._build_adjacency(edges)

        for annotation in self._conditional_edge_annotations:
            self._validate_conditional_annotation_nodes(annotation, contract_nodes)
            non_llm_reachability_owners = self._compute_non_llm_reachability_owners(
                annotation.destinations, contract_nodes, adjacency)

            for group_index, destinations in enumerate(annotation.destinations):
                seq_var = f"seq_{uuid.uuid4().hex[:8]}"
                contract_nodes[annotation.src].writes.add(seq_var)

                if annotation.frontiers is None:
                    frontier_nodes = self._infer_first_llm_frontier(
                        group_index, destinations, contract_nodes, adjacency,
                        non_llm_reachability_owners)
                else:
                    frontier_nodes = list(dict.fromkeys(annotation.frontiers[group_index]))

                for frontier_name in frontier_nodes:
                    frontier_node = contract_nodes[frontier_name]
                    if not frontier_node.llm_calls:
                        raise ValueError(
                            f"annotate_conditional_edge frontier '{frontier_name}' must be an LLM node")
                    for llm_call in frontier_node.llm_calls:
                        if seq_var not in llm_call.reads:
                            llm_call.reads.append(seq_var)

    def _ensure_rid_node(self):
        def __gp_ensure_rid(state, config):
            rid = uuid.uuid4().hex
            # Get graph_run_id from config (thread_id) for global RID store
            graph_run_id = None
            if config:
                configurable = config.get("configurable", {})
                graph_run_id = configurable.get("thread_id")
            if not graph_run_id:
                graph_run_id = state.get("thread_id") if isinstance(state, dict) else None
            if not graph_run_id:
                graph_run_id = uuid.uuid4().hex  # Fallback
            
            set_rid(rid, graph_run_id)
            # Store both in state for parallel nodes
            return {"__gp_rid": rid, "__gp_graph_run_id": graph_run_id}
        return __gp_ensure_rid

    def _wrap_node_with_rid_usage_check(self, fn, node_name: str):
        @functools.wraps(fn)
        def _wrapped(state, config=None, *a, **kw):
            reset_rid_used()
            out = fn(state, config=config, *a, **kw)
            if not was_rid_used():
                print(f"[GraphProxy] Warning: node '{node_name}' appears to call an SDK "
                      f"but never accessed RID; add X-Run-Id=get_rid() to request headers.")
            return out
        return _wrapped

    def _wrap_node_with_node_name(self, fn, node_name: str):
        @functools.wraps(fn)
        def _wrapped(state, config=None, *a, **kw):
            # Restore RID context for parallel execution (may be in different thread)
            graph_run_id = None
            if config:
                configurable = config.get("configurable", {})
                graph_run_id = configurable.get("thread_id")
            if not graph_run_id and isinstance(state, dict):
                graph_run_id = state.get("__gp_graph_run_id") or state.get("thread_id")
            
            if graph_run_id:
                # This will restore RID from global store if needed
                from .rid import set_graph_run_id
                set_graph_run_id(graph_run_id)
            
            set_node_name(node_name)
            return fn(state, config=config, *a, **kw)
        return _wrapped

    # --- pass-through build methods (record + forward) ---
    def add_node(self, name: str, fn: Any):
        # Wrap all nodes to set node name context
        fn = self._wrap_node_with_node_name(fn, name)
        
        # Check if this is an LLM node based on decorators
        is_llmish = bool(getattr(fn, D.META_LLM_CALLS, []))

        if is_llmish:
            found = wrap_llm_attrs_on_fn(fn)
            if not found:
                found = wrap_llm_referenced_globals(fn)
            if not found:
                # Likely raw SDK; check if they at least reference get_rid in source
                mentions = fn_mentions_get_rid(fn)
                if mentions is False:
                    print(f"[GraphProxy] Warning: {name} has no detectable LLM handle "
                          f"and no get_rid() mention; ensure SDK headers include the RID.")
                # Optional runtime check
                fn = self._wrap_node_with_rid_usage_check(fn, name)

        self._nodes[name] = _NodeRecord(name=name, fn=fn)
        return self._g.add_node(name, fn)

    def add_edge(self, src: Any, dst: Any, *, label: Optional[str] = None):
        # record only if both ends are strings
        if isinstance(src, str) and isinstance(dst, str):
            src_text = src
            dst_text = dst
            if src == START:
                src_text = 'START'
            if dst == END:
                dst_text = 'END'
            self._record_edge(Edge(src=src_text, dst=dst_text, label=label))
        else:
            raise ValueError(f"Invalid edge: {src} -> {dst}")
        return self._g.add_edge(src, dst)

    def annotate_conditional_edge(self,
                                  src: str,
                                  destinations: List[List[str]],
                                  *,
                                  frontiers: Optional[List[List[str]]] = None):
        """
        Annotate the routing structure for a conditional edge before calling add_conditional_edges.
        Args:
            src: Source node name
            destinations: List of fanout groups. Each group is a list of destination strings.
                        For single destinations, use a list with one element: [["single_dest"]]
                        For fanout to multiple nodes: [["dest1", "dest2"]]
                        For multiple fanout groups: [["group1a", "group1b"], ["group2"]]
            frontiers: Optional explicit LLM frontier groups. When omitted, GraphProxy
                        infers the first downstream LLM nodes that still belong uniquely
                        to each conditional branch.
        """
        normalized_destinations = self._copy_grouped_node_names(
            destinations, "destinations")
        normalized_frontiers = None
        if frontiers is not None:
            normalized_frontiers = self._copy_grouped_node_names(
                frontiers, "frontiers", allow_empty_groups=True)
            if len(normalized_frontiers) != len(normalized_destinations):
                raise ValueError(
                    "frontiers must have the same number of groups as destinations")

        self._record_conditional_annotation(
            _ConditionalEdgeAnnotation(
                src=src,
                destinations=normalized_destinations,
                frontiers=normalized_frontiers,
            ))

    def add_conditional_edges(self, *args, **kwargs):
        """
        Pass-through to the underlying StateGraph's add_conditional_edges.
        Routing structure should be annotated beforehand using annotate_conditional_edge().
        """
        return self._g.add_conditional_edges(*args, **kwargs)

    def set_entry_point(self, name: str):
        self._entry_point = name
        # Add RID guard node
        guard = "__gp_ensure_rid"
        self._g.add_node(guard, self._ensure_rid_node())
        self._g.set_entry_point(guard)
        self._g.add_edge(guard, name)
        
        # Record edges for contract
        self._record_edge(Edge(src='START', dst=guard, label=None))
        self._record_edge(Edge(src=guard, dst=name, label=None))
        return self._g

    def set_finish_point(self, name: str):
        self._record_edge(Edge(src=name, dst='END', label=None))
        return self._g.set_finish_point(name)

    def attach_graph(self, 
                     node_name: str, 
                     subgraph: 'GraphProxy'):
        """
        Attach a subgraph GraphProxy at a specific node in this graph.
        
        Args:
            node_name: The node in this graph where the subgraph will be attached
            subgraph: The GraphProxy instance to attach
        
        This method merges the subgraph into the current graph by:
        1. Adding all nodes from the subgraph to the current graph's representation
        2. Adding all edges from the subgraph to the current graph's representation
        3. Connecting the attachment point to the subgraph entry
        4. Connecting the subgraph exit to the next node in the current graph
        
        Note: This only affects the GraphProxy representation for contract generation.
        The underlying StateGraph objects remain unchanged and should be called normally.
        """
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' not found in current graph")
        
        # Remove the original attachment node and its edges
        if node_name in self._nodes:
            del self._nodes[node_name]
        
        # Add all nodes from subgraph (with namespace to avoid conflicts)
        namespace = f"{node_name}_"
        for sub_node_name, sub_node_rec in subgraph._nodes.items():
            if sub_node_name in ['START', 'END']:
                continue 
            new_name = f"{namespace}{sub_node_name}"
            self._nodes[new_name] = sub_node_rec
        
        # Find the actual first and last nodes in the subgraph
        sub_first_nodes = [edge.dst for edge in subgraph._edges if edge.src == 'START']
        sub_last_nodes = [edge.src for edge in subgraph._edges if edge.dst == 'END']
        
        # Add all internal edges from subgraph (with namespace)
        for edge in subgraph._edges:
            if (edge.src != 'START' and edge.dst != 'END'):
                # Internal subgraph edge
                new_edge = Edge(
                    src=f"{namespace}{edge.src}",
                    dst=f"{namespace}{edge.dst}",
                    label=edge.label
                )
                self._record_edge(new_edge)
        
        # Now connect the incoming and outgoing edges properly
        if sub_first_nodes:
            for edge in self._edges:
                if edge.dst == node_name:
                    for sub_first_node in sub_first_nodes:
                        new_edge = Edge(
                            src=edge.src,
                            dst=f"{namespace}{sub_first_node}",
                            label=edge.label
                        )
                        self._record_edge(new_edge)
                    self._remove_recorded_edge(edge)
        
        if sub_last_nodes:
            for edge in self._edges:
                if edge.src == node_name:
                    for sub_last_node in sub_last_nodes:
                        new_edge = Edge(
                            src=f"{namespace}{sub_last_node}",
                            dst=edge.dst,
                            label=edge.label
                        )
                        self._record_edge(new_edge)
                    self._remove_recorded_edge(edge)

    # --- outputs ---
    def materialize(self):
        """Return the underlying StateGraph to the caller."""
        return self._g

    def build_contract(self) -> Contract:
        c = Contract(entry='START', end='END')
        # nodes
        for name, rec in self._nodes.items():
            fn = rec.fn
            writes = set(getattr(fn, D.META_WRITES, set()))
            # llm calls
            llm_specs = list(getattr(fn, D.META_LLM_CALLS, []))
            llm_calls: List[LLMCall] = []
            for spec in llm_specs:
                llm_calls.append(
                    LLMCall(
                        model=spec.get("model"),
                        reads=list(spec.get("reads", [])),
                        static_vars=list(spec.get("static_vars", [])),
                    )
                )
            c.nodes[name] = NodeMeta(
                name=name,
                fn_qualname=f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', getattr(fn, '__name__', str(fn)))}",
                writes=writes,
                llm_calls=llm_calls,
            )
        
        # Add RID guard node to contract
        c.nodes["__gp_ensure_rid"] = NodeMeta(
            name="__gp_ensure_rid",
            fn_qualname="GraphProxy._ensure_rid_node",
            writes=set(),
            llm_calls=[],
        )

        unique_edges = self._dedupe_edges(self._resolve_contract_edges())
        self._apply_conditional_edge_annotations(c.nodes, unique_edges)
        
        # Check if all writes and reads are empty, add dummy variables if so
        all_writes_empty = all(not node.writes for node in c.nodes.values())
        all_reads_empty = all(not llm_call.reads for node in c.nodes.values() for llm_call in node.llm_calls)
        
        if all_writes_empty and all_reads_empty:
            dummy_var = "__dummy_dependency"
            for node in c.nodes.values():
                node.writes.add(dummy_var)
                for llm_call in node.llm_calls:
                    llm_call.reads.append(dummy_var)
        
        c.edges.extend(unique_edges)
        return c
