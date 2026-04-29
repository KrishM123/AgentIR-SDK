# AgentIR SDK

This folder is a copy-ready SDK export. Move it into a separate repository and install it from there when running AgentIR graph benchmarks.

## Purpose

- annotate graph nodes with read/write metadata
- build scheduler-facing graph contracts
- propagate RID and node-name metadata into model calls
- provide lightweight client-side logging helpers

## What Belongs Here

- decorators in `agentir_sdk/decorators.py`
- contract data structures in `agentir_sdk/contract.py`
- `GraphProxy` and RID helpers in `agentir_sdk/graph_proxy.py` and `agentir_sdk/rid.py`
- client-side logging in `agentir_sdk/client_logger.py`
- usage guidance in `agentir_sdk/interface.md`

## Install After Copying

```bash
pip install -e <path-to-agentir-sdk-repo>
```

## Important Invariants

- `@writes` and `@llm_call` annotations should reflect the real graph contract.
- `GraphProxy.build_contract()` is the source of truth for scheduler-facing contract serialization.
- RID propagation should remain explicit and observable; missing RID or node-name metadata is a correctness problem.
