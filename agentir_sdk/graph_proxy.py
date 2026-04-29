from typing import Any, Dict, List, Optional
from dataclasses import dataclass
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
        self._entry_point: Optional[str] = None

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
            self._edges.append(Edge(src=src_text, dst=dst_text, label=label))
        else:
            raise ValueError(f"Invalid edge: {src} -> {dst}")
        return self._g.add_edge(src, dst)

    def annotate_conditional_edge(self, src: str, destinations: List[List[str]]):
        """
        Annotate the routing structure for a conditional edge before calling add_conditional_edges.
        This method adds a random variable to the source node's writes and the destination node's reads to enforce sequentiality.
        Args:
            src: Source node name
            destinations: List of fanout groups. Each group is a list of destination strings.
                        For single destinations, use a list with one element: [["single_dest"]]
                        For fanout to multiple nodes: [["dest1", "dest2"]]
                        For multiple fanout groups: [["group1a", "group1b"], ["group2"]]
        """
        for i, dest_group in enumerate(destinations):
            for dst in dest_group:
                # Generate a random variable name for sequentiality enforcement
                random_var = f"seq_{uuid.uuid4().hex[:8]}"
                
                # Mark as conditional edge
                label = f"conditional_{i}" if len(destinations) > 1 else "conditional"
                self._edges.append(Edge(src=src, dst=dst, label=label))
                
                # Add the random variable to source node's writes
                if src in self._nodes:
                    src_node = self._nodes[src]
                    if src_node.fn is not None:
                        if not hasattr(src_node.fn, D.META_WRITES):
                            setattr(src_node.fn, D.META_WRITES, set())
                        getattr(src_node.fn, D.META_WRITES).add(random_var)
                
                # Add the random variable to destination node's reads
                if dst in self._nodes:
                    dst_node = self._nodes[dst]
                    if dst_node.fn is not None:
                        if hasattr(dst_node.fn, D.META_LLM_CALLS):
                            llm_calls = getattr(dst_node.fn, D.META_LLM_CALLS)
                            if llm_calls:
                                llm_calls[0].setdefault('reads', []).append(random_var)

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
        self._edges.append(Edge(src='START', dst=guard, label=None))
        self._edges.append(Edge(src=guard, dst=name, label=None))
        return self._g

    def set_finish_point(self, name: str):
        self._edges.append(Edge(src=name, dst='END', label=None))
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
                self._edges.append(new_edge)
        
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
                        self._edges.append(new_edge)
                    self._edges.remove(edge)
        
        if sub_last_nodes:
            for edge in self._edges:
                if edge.src == node_name:
                    for sub_last_node in sub_last_nodes:
                        new_edge = Edge(
                            src=f"{namespace}{sub_last_node}",
                            dst=edge.dst,
                            label=edge.label
                        )
                        self._edges.append(new_edge)
                    self._edges.remove(edge)

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
        
        # edges - remove duplicates while preserving order
        seen_edges = set()
        unique_edges = []
        for edge in self._edges:
            edge_tuple = (edge.src, edge.dst, edge.label)
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                unique_edges.append(edge)
        
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
