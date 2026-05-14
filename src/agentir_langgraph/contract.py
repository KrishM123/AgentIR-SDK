from typing import List, Set, Dict
from dataclasses import dataclass, field


@dataclass
class LLMCall:
    model: str = None
    reads: List[str] = field(default_factory=list)
    static_vars: List[str] = field(default_factory=list)

@dataclass
class NodeMeta:
    name: str
    fn_qualname: str
    writes: Set[str] = field(default_factory=set)
    llm_calls: List[LLMCall] = field(default_factory=list)

@dataclass
class Edge:
    src: str
    dst: str
    label: str = None

@dataclass
class Contract:
    entry: str = None
    end: str = None
    nodes: Dict[str, NodeMeta] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    
    def to_dict(self):
        """Convert the contract to a JSON-serializable dictionary."""
        return {
            "entry": self.entry,
            "end": self.end,
            "nodes": {
                name: {
                    "writes": list(node.writes),
                    "llm_calls": [
                        {
                            "model": call.model,
                            "reads": list(call.reads),
                            "static_vars": list(call.static_vars)
                        } for call in node.llm_calls
                    ]
                } for name, node in self.nodes.items()
            },
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "label": edge.label
                } for edge in self.edges
            ]
        }