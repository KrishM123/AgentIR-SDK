# AgentIR LangGraph Annotation Guide

This guide explains how to annotate LangGraph nodes so external schedulers and analyzers can reason about your graph.

The SDK sources referenced below live in `agentir_sdk/`.

## What You Get

With annotations plus `GraphProxy`, you can expose:

- Which state keys each node writes
- Which state keys each LLM call reads
- Static prompt components per call
- Conditional routing structure
- A machine-readable graph contract (`Contract.to_dict()`)

## Core API

### 1) `@writes(*keys: str)`

Declare state keys produced by a node.

```python
from agentir_sdk.decorators import writes

@writes("draft", "messages")
def composer(state, config=None):
    ...
    return {"draft": text, "messages": [...]}
```

Guidance:

- List only keys the node really produces.
- Keep names aligned with your `TypedDict` state.
- Include keys even if they are sometimes empty.

Scheduling implication:

- `writes` defines outputs that downstream nodes may depend on.

### 2) `@llm_call(model=None, reads=None, static_vars=None)`

Declare one LLM call site inside a node.

```python
from agentir_sdk.decorators import llm_call

@llm_call(
    model="gpt-4o-mini",
    reads=["task_brief", "research_notes"],
    static_vars=["You are a precise writing assistant."]
)
def composer(state, config=None):
    ...
```

Parameters:

- `model`: logical model identifier
- `reads`: state keys used to build that prompt
- `static_vars`: static prompt constants/templates

You can stack multiple `@llm_call` decorators on one function for multiple call sites.

Scheduling implication:

- `reads` exposes data dependencies for that call.
- `model` and `static_vars` provide stable metadata about call shape.

## Build Graphs with `GraphProxy`

Use `GraphProxy` as the builder wrapper around `StateGraph`.

```python
from langgraph.graph import StateGraph
from agentir_sdk.graph_proxy import GraphProxy

workflow = StateGraph(MyState)
G = GraphProxy(workflow)

G.add_node("router", router)
G.add_node("composer", composer)
G.add_edge("router", "composer")

G.set_entry_point("router")
G.set_finish_point("composer")

contract = G.build_contract()
graph = G.materialize().compile()
```

## Other Available Features

`GraphProxy` also includes:

- `attach_graph(node_name, subgraph)`: merge a subgraph view into the parent contract representation
- entry-point RID guard insertion via `set_entry_point(...)` for per-run tracing context
- LLM handle detection and wrapping helpers to improve RID and node-name propagation

`agentir_sdk/rid.py` provides low-level helpers (`get_rid`, node-name context helpers, and wrappers) if you build custom LLM adapters.

## Conditional Routing Annotation

If you use `add_conditional_edges`, annotate route possibilities first:

```python
def route(state):
    return "qa" if state.get("mode") == "qa" else "planner"

G.annotate_conditional_edge("router", [["qa"], ["planner"]])
G.add_conditional_edges("router", route, {
    "qa": "qa",
    "planner": "planner",
})
```

Why this matters:

- It makes branch possibilities explicit in the generated contract.
- It improves dependency visibility for schedulers and analyzers.

If a conditional branch first passes through non-LLM nodes, `GraphProxy` will
infer the first downstream LLM frontier for that branch automatically. When a
branch merges with another branch before the first LLM node, that frontier is
ambiguous; use `frontiers=` to pin it explicitly:

```python
G.annotate_conditional_edge(
    "router",
    [["non_llm_gateway"], ["direct_llm"]],
    frontiers=[["writer", "critic"], ["direct_llm"]],
)
```

## Contract Shape

`build_contract()` returns a `Contract` object with:

- `entry`, `end`
- `nodes[name].writes`
- `nodes[name].llm_calls[]` with `model`, `reads`, `static_vars`
- `edges[]` (`src`, `dst`, optional `label`)

Serialize via:

```python
payload = contract.to_dict()
```

## High-Level Scheduling Implications

Without exposing internal policy details, annotations generally enable:

- better dependency tracking between nodes
- more accurate readiness analysis for downstream work
- clearer branch and fanout structure for conditional paths
- better observability of LLM call footprints

Poor or missing annotations usually mean less precise planning and fewer optimization opportunities.

## Best Practices

- Keep `writes` minimal and accurate.
- Keep `reads` specific to actual prompt inputs.
- Use stable names for state keys and model IDs.
- Annotate every node that performs LLM calls.
- Annotate conditional structures before adding conditional edges.
- Rebuild contracts after graph topology or annotation changes.

## Common Mistakes

- Declaring `writes` keys that are never returned.
- Omitting keys in `reads` that are used in prompt construction.
- Forgetting to annotate conditional route options.
- Treating `static_vars` as dynamic runtime data.

## End-to-End Example

```python
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from agentir_sdk.decorators import writes, llm_call
from agentir_sdk.graph_proxy import GraphProxy

class DocState(TypedDict, total=False):
    messages: list
    route: str
    draft: str

@writes("route", "messages")
@llm_call(model="router-model", reads=["messages"], static_vars=["route_prompt_v1"])
def router(state: DocState, config=None):
    user = state["messages"][-1].content
    route = "qa" if "qa" in user.lower() else "write"
    return {"route": route, "messages": [AIMessage(content=f"[route] {route}")]}

@writes("draft", "messages")
@llm_call(model="writer-model", reads=["messages"], static_vars=["writer_prompt_v2"])
def writer(state: DocState, config=None):
    return {"draft": "hello", "messages": [AIMessage(content="[writer] done")]}

@writes("draft", "messages")
@llm_call(model="qa-model", reads=["messages"], static_vars=["qa_prompt_v1"])
def qa(state: DocState, config=None):
    return {"draft": "answer", "messages": [AIMessage(content="[qa] done")]}

wf = StateGraph(DocState)
G = GraphProxy(wf)

G.add_node("router", router)
G.add_node("writer", writer)
G.add_node("qa", qa)

G.set_entry_point("router")

G.annotate_conditional_edge("router", [["writer"], ["qa"]])
G.add_conditional_edges("router", lambda s: "qa" if s.get("route") == "qa" else "writer", {
    "writer": "writer",
    "qa": "qa",
})

G.set_finish_point("writer")
G.set_finish_point("qa")

contract = G.build_contract()
```
