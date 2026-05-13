"""
RID (Run ID) functionality for transport-level tracing in LangGraph.

This module provides:
- RID management with thread-local storage + global fallback
- LLM wrapper for automatic RID injection
- Scheduler header binding for custom invoke wrappers
- LLM discovery and wrapping utilities
- Static analysis for RID usage detection
"""
import copy
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
_SCHEDULER_RID_MODULUS = 2**31


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


def get_scheduler_rid() -> Optional[str]:
    """Return the scheduler-ready integer RID string for the current run."""
    rid = get_rid()
    if rid is None:
        return None
    return str(int(rid, 16) % _SCHEDULER_RID_MODULUS)


def get_scheduler_headers() -> Dict[str, str]:
    """Build scheduler request headers from the current GraphProxy context."""
    headers: Dict[str, str] = {}
    rid = get_scheduler_rid()
    node_name = get_node_name()
    if rid is not None:
        headers["rid"] = rid
    if node_name is not None:
        headers["node-name"] = node_name
    return headers


class SchedulerHeaderBindableMixin:
    """
    Small opt-in protocol for user-owned scheduler clients.

    GraphProxy can wrap these objects without changing how nodes call their
    custom `.invoke()` methods. The bound copy carries the RID-derived
    scheduler headers for one graph-executed call.
    """

    def with_scheduler_headers(self, headers: Dict[str, str]):
        bound = copy.copy(self)
        bound._gp_scheduler_headers = dict(headers)
        return bound

    def bound_scheduler_headers(self) -> Dict[str, str]:
        headers = getattr(self, "_gp_scheduler_headers", None)
        if not headers or "rid" not in headers or "node-name" not in headers:
            raise ValueError(
                "Scheduler headers are not bound; this client must run inside "
                "a GraphProxy-managed node execution")
        return dict(headers)


def _supports_runtime_header_config(obj) -> bool:
    return callable(getattr(obj, "with_config", None))


def _supports_scheduler_header_binding(obj) -> bool:
    return callable(getattr(obj, "with_scheduler_headers", None))


def _supports_rid_injection(obj) -> bool:
    return looks_like_llm(obj) and (
        _supports_runtime_header_config(obj) or
        _supports_scheduler_header_binding(obj)
    )


def _accepts_config_argument(fn) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True

    if "config" in signature.parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

def with_rid_per_run(base_llm):
    """Idempotent wrapper that injects X-Run-Id per call via runtime config."""
    if getattr(base_llm, "_gp_wrapped", False):
        return base_llm

    if not _supports_rid_injection(base_llm):
        return base_llm

    runtime_config_base = base_llm
    if _supports_runtime_header_config(base_llm):
        # Some LCEL objects need this preparatory step before with_config accepts
        # default_headers. Custom scheduler wrappers take the separate binding path.
        try:
            runtime_config_base = base_llm.configurable_fields(default_headers=True)
        except Exception:
            runtime_config_base = base_llm

    class _LLMWithRID:
        _gp_wrapped = True  # idempotence flag

        def __init__(self, base): 
            self.base = base

        def _view(self, config):
            if _supports_scheduler_header_binding(self.base):
                return self.base.with_scheduler_headers(get_scheduler_headers())

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

        def _call_bound(self, method_name: str, *args, **kwargs):
            config = kwargs.get("config")
            view = self._view(config)
            method = getattr(view, method_name)

            call_kwargs = kwargs
            if "config" in kwargs and not _accepts_config_argument(method):
                call_kwargs = dict(kwargs)
                call_kwargs.pop("config")

            return method(*args, **call_kwargs)

        def invoke(self, *args, **kwargs):
            return self._call_bound("invoke", *args, **kwargs)

        async def ainvoke(self, *args, **kwargs):
            return await self._call_bound("ainvoke", *args, **kwargs)

        def stream(self, *args, **kwargs):
            return self._call_bound("stream", *args, **kwargs)

        # Add batch/abatch/astream here if your graphs use them.

    return _LLMWithRID(runtime_config_base)

def looks_like_llm(obj) -> bool:
    """Narrow duck-typing for LCEL-like runnables."""
    return any(hasattr(obj, m) for m in ("invoke", "ainvoke", "stream", "batch"))

def wrap_llm_attrs_on_fn(fn) -> bool:
    """Replace any function attribute that looks like an LLM."""
    found = False
    for name, val in list(vars(fn).items()):
        if getattr(val, "_gp_wrapped", False):
            found = True
            continue
        if _supports_rid_injection(val):
            wrapped = with_rid_per_run(val)
            if wrapped is not val:
                setattr(fn, name, wrapped)
            found = True
    return found

def wrap_llm_referenced_globals(fn) -> bool:
    """Replace any referenced global binding that looks like an LLM."""
    def _wrap_referenced_globals(inner_fn, seen_fn_ids) -> bool:
        base = inspect.unwrap(inner_fn)
        fn_id = id(base)
        if fn_id in seen_fn_ids:
            return False
        seen_fn_ids.add(fn_id)

        code = getattr(base, "__code__", None)
        names = set(getattr(code, "co_names", ()))
        found = False

        for name in names:
            if name not in base.__globals__:
                continue

            obj = base.__globals__[name]
            if getattr(obj, "_gp_wrapped", False):
                found = True
                continue

            if _supports_rid_injection(obj):
                wrapped = with_rid_per_run(obj)
                if wrapped is not obj:
                    base.__globals__[name] = wrapped
                found = True
                continue

            if inspect.isfunction(obj) and _wrap_referenced_globals(obj, seen_fn_ids):
                found = True

        return found

    return _wrap_referenced_globals(fn, set())

def fn_mentions_get_rid(fn) -> Optional[bool]:
    """True: found call to get_rid. False: definitely not seen. None: unknown."""
    base = inspect.unwrap(fn)
    try:
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
        for op in dis.get_instructions(base):
            if op.opname in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_ATTR", "LOAD_METHOD"} and op.argval == "get_rid":
                return True
    except Exception:
        return None

    return False
