from __future__ import annotations
from typing import Callable, Iterable, Optional

# We keep metadata on function objects under reserved attributes.
# These are aggregated at graph-build time by GraphProxy/Analyzer.

META_WRITES = "__lg_writes__"
META_LLM_CALLS = "__lg_llm_calls__"

class _LLMCallSpec(dict):
    # tiny carrier used by the decorator; converted into contract.LLMCall later
    pass


def writes(*keys: str):
    def deco(fn: Callable):
        setattr(fn, META_WRITES, set(keys) | getattr(fn, META_WRITES, set()))
        return fn
    return deco


def llm_call(*,
             model: Optional[str] = None,
             reads: Optional[Iterable[str]] = None,
             static_vars: Optional[Iterable[str]] = None):
    """
    Annotate an LLM call site inside a node function without changing its behavior.

    - name: logical name for the call (unique within the node)
    - model: optional model identifier
    - reads: state keys interpolated into the prompt
    - static_vars: static substitution values for the prompt

    You can stack multiple @llm_call on the same function to describe multiple call sites.
    """
    r = list(reads or [])
    sv = list(static_vars or [])

    def deco(fn):
        spec = _LLMCallSpec(
            model=model,
            reads=r,
            static_vars=sv,
        )
        setattr(fn, META_LLM_CALLS, [spec] + getattr(fn, META_LLM_CALLS, []))
        return fn
    return deco