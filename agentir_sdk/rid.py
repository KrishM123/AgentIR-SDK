"""
RID (Run ID) functionality for transport-level tracing in LangGraph.

This module provides:
- RID management with thread-local storage + global fallback
- LLM wrapper for automatic RID injection
- LLM discovery and wrapping utilities
- Static analysis for RID usage detection
"""
import threading
from typing import Optional, Dict, Any
import ast
import inspect
import dis
import textwrap

# Thread-local storage for RID
_thread_local = threading.local()

# Global RID store keyed by graph_run_id for parallel node execution
# This handles cases where LangGraph runs parallel nodes in different threads
_global_rid_store: Dict[str, str] = {}
_global_rid_lock = threading.Lock()


def set_rid(rid: str, graph_run_id: Optional[str] = None) -> None:
    """Proxy internal: seed the run RID and reset usage flag."""
    _thread_local.rid = rid
    _thread_local.rid_used = False
    
    # Also store in global dict if graph_run_id provided
    if graph_run_id:
        _thread_local.graph_run_id = graph_run_id
        with _global_rid_lock:
            _global_rid_store[graph_run_id] = rid


def set_graph_run_id(graph_run_id: str) -> None:
    """Set the graph run ID for this thread to lookup RID from global store."""
    _thread_local.graph_run_id = graph_run_id
    # Try to restore RID from global store
    with _global_rid_lock:
        if graph_run_id in _global_rid_store:
            _thread_local.rid = _global_rid_store[graph_run_id]


def get_rid() -> Optional[str]:
    """Zero-arg getter for SDK users & helpers. Marks RID as used."""
    _thread_local.rid_used = True
    
    # First try thread-local
    rid = getattr(_thread_local, 'rid', None)
    if rid:
        return rid
    
    # Fallback: try to get from global store using graph_run_id
    graph_run_id = getattr(_thread_local, 'graph_run_id', None)
    if graph_run_id:
        with _global_rid_lock:
            rid = _global_rid_store.get(graph_run_id)
            if rid:
                _thread_local.rid = rid  # Cache in thread-local
                return rid
    
    return None


def cleanup_rid(graph_run_id: str) -> None:
    """Clean up RID from global store when graph execution completes."""
    with _global_rid_lock:
        _global_rid_store.pop(graph_run_id, None)


def reset_rid_used() -> None:
    """Proxy internal: reset the usage flag."""
    _thread_local.rid_used = False


def was_rid_used() -> bool:
    """Proxy internal: for runtime diagnostics after a node finishes."""
    return getattr(_thread_local, 'rid_used', False)


def set_node_name(node_name: str) -> None:
    """Proxy internal: set the current node name."""
    _thread_local.node_name = node_name


def get_node_name() -> Optional[str]:
    """Get the current node name."""
    return getattr(_thread_local, 'node_name', None)

def with_rid_per_run(base_llm):
    """Idempotent wrapper that injects X-Run-Id per call via runtime config."""
    if getattr(base_llm, "_gp_wrapped", False):
        return base_llm

    # Allow overriding headers at runtime where supported
    try:
        base_llm = base_llm.configurable_fields(default_headers=True)
    except Exception:
        pass  # some runnables won't expose this; we still delegate

    class _LLMWithRID:
        _gp_wrapped = True  # idempotence flag

        def __init__(self, base): 
            self.base = base

        def _view(self, config):
            rid = get_rid()
            node_name = get_node_name()
            headers = {}
            if rid:
                headers["X-Run-Id"] = rid
            if node_name:
                headers["X-Node-Name"] = node_name
            try:
                return self.base.with_config(
                    configurable={"default_headers": headers}
                )
            except Exception:
                return self.base

        def invoke(self, x, *, config=None):
            view = self._view(config)
            try:
                return view.invoke(x, config=config)
            except TypeError:
                # Fallback for LLMs that don't accept config parameter
                return view.invoke(x)

        async def ainvoke(self, x, *, config=None):
            view = self._view(config)
            try:
                return await view.ainvoke(x, config=config)
            except TypeError:
                # Fallback for LLMs that don't accept config parameter
                return await view.ainvoke(x)

        def stream(self, x, *, config=None):
            view = self._view(config)
            try:
                return view.stream(x, config=config)
            except TypeError:
                # Fallback for LLMs that don't accept config parameter
                return view.stream(x)

        # Add batch/abatch/astream here if your graphs use them.

    return _LLMWithRID(base_llm)

def looks_like_llm(obj) -> bool:
    """Narrow duck-typing for LCEL-like runnables."""
    return any(hasattr(obj, m) for m in ("invoke", "ainvoke", "stream", "batch"))

def wrap_llm_attrs_on_fn(fn) -> bool:
    """Replace any function attribute that looks like an LLM."""
    changed = False
    for name, val in list(vars(fn).items()):
        if looks_like_llm(val) and not getattr(val, "_gp_wrapped", False):
            setattr(fn, name, with_rid_per_run(val))
            changed = True
    return changed

def wrap_llm_referenced_globals(fn) -> bool:
    """Replace any referenced global binding that looks like an LLM."""
    code = getattr(fn, "__code__", None)
    names = set(getattr(code, "co_names", ()))
    changed = False
    for name in names:
        if name in fn.__globals__:
            obj = fn.__globals__[name]
            if looks_like_llm(obj) and not getattr(obj, "_gp_wrapped", False):
                fn.__globals__[name] = with_rid_per_run(obj)
                changed = True
    return changed

def fn_mentions_get_rid(fn) -> Optional[bool]:
    """True: found call to get_rid. False: definitely not seen. None: unknown."""
    try:
        base = inspect.unwrap(fn)
        src = inspect.getsource(base)
        # Nested/decorated functions can come with leading indentation.
        tree = ast.parse(textwrap.dedent(src))

        class V(ast.NodeVisitor):
            seen = False
            def visit_Call(self, node: ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id == "get_rid":
                    self.seen = True
                elif isinstance(f, ast.Attribute) and f.attr == "get_rid":
                    self.seen = True
                self.generic_visit(node)

        v = V()
        v.visit(tree)
        if v.seen:
            return True
    except (OSError, SyntaxError):
        pass  # no source (REPL/compiled)

    try:
        for op in dis.get_instructions(fn):
            if op.opname in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_ATTR", "LOAD_METHOD"} and op.argval == "get_rid":
                return True
    except Exception:
        return None

    return False
