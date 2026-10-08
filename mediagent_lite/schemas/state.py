"""LangGraph shared state definition.

Why TypedDict for LangGraph state?
LangGraph requires a TypedDict (not a Pydantic model) for its StateGraph.
The typed fields ensure each node's reducer knows the expected shape.
We store Pydantic objects inside, but the container is a TypedDict.

Key teaching point: state is IMMUTABLE per step. Each node returns a dict
of UPDATES only (not the full state). LangGraph merges them using reducers.
Default reducer is 'last wins'. Lists use the 'append' reducer if annotated.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from mediagent_lite.schemas.clinical import StructuredCase
from mediagent_lite.schemas.evidence import EvidenceBundle
from mediagent_lite.schemas.hypothesis import HypothesisList
from mediagent_lite.schemas.report import FusedReport


class TraceEntry(TypedDict):
    """A single structured trace entry for one node execution."""

    node: str
    inputs_summary: str  # brief string summary (not full objects, to keep trace readable)
    outputs_summary: str
    latency_ms: float
    iteration: int
    tokens_used: Optional[int]  # None if provider doesn't report
    timestamp: str  # ISO format


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph state machine.

    Design notes:
    - raw_input: the original text, never modified after entry
    - iteration: incremented by AdaptiveOptimizer each loop
    - query_history: Optimizer reads this to avoid repeating queries
    - trace: append-only log of node executions (list reducer)
    """

    # Input
    raw_input: str
    config_flags: dict[str, Any]  # ablation flags (e.g. use_rag, use_web, use_fusion)

    # Intermediate state (each set by the corresponding node)
    structured_case: Optional[StructuredCase]
    hypotheses: Optional[HypothesisList]
    evidence_bundles: list[EvidenceBundle]
    final_report: Optional[FusedReport]

    # Feedback loop control
    confidence: float  # top confidence from current iteration
    iteration: int  # current loop count (0-indexed)
    query_history: list[str]  # previous queries, used by optimizer to diversify

    # Observability
    trace: Annotated[list[TraceEntry], lambda a, b: a + b]  # append-only
    run_id: str  # unique per invocation
    error: Optional[str]  # set if any node fails gracefully
