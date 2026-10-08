"""LangGraph node functions.

These are thin wrappers that adapt our Agent classes to the LangGraph
StateGraph signature: `def node(state: dict) -> dict`
They return dictionaries representing the UPDATES to be applied to the state.
"""

import time
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig

from mediagent_lite.agents.clarifier import ClinicalTextClarifier
from mediagent_lite.agents.rag_analyzer import SymptomRAGAnalyzer
from mediagent_lite.schemas.state import AgentState, TraceEntry


def _create_trace(node_name: str, start_time: float, inputs: str, outputs: str, iteration: int) -> TraceEntry:
    """Helper to generate a standardized trace entry."""
    return TraceEntry(
        node=node_name,
        inputs_summary=inputs,
        outputs_summary=outputs,
        latency_ms=round((time.time() - start_time) * 1000, 2),
        iteration=iteration,
        tokens_used=None,  # Not tracked in Lite version by default
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


def clarifier_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Node wrapper for ClinicalTextClarifier."""
    start = time.time()
    agent = ClinicalTextClarifier()
    
    # Process
    raw_text = state["raw_input"]
    structured_case = agent.process(raw_text, config=config)
    
    # Trace
    in_sum = f"Text ({len(raw_text)} chars)"
    out_sum = f"Case (cc: {structured_case.chief_complaint}, {len(structured_case.symptoms)} syms)"
    trace = _create_trace("ClinicalTextClarifier", start, in_sum, out_sum, state.get("iteration", 0))
    
    # Return updates
    return {
        "structured_case": structured_case,
        "trace": [trace]  # list reducer will append this
    }


def rag_analyzer_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Node wrapper for SymptomRAGAnalyzer."""
    start = time.time()
    agent = SymptomRAGAnalyzer()
    
    case = state["structured_case"]
    if not case:
        raise ValueError("RAG Analyzer requires a structured_case in state.")
        
    config_flags = state.get("config_flags", {})
    iteration = state.get("iteration", 0)
    
    # If we are in a feedback loop, use the latest query from history
    override_query = None
    if iteration > 0 and state.get("query_history"):
        override_query = state["query_history"][-1]
        
    # Process
    hypotheses = agent.process(case, config_flags, override_query, config=config)
    
    # Trace
    in_sum = override_query if override_query else f"Case (cc: {case.chief_complaint})"
    out_sum = f"Found {len(hypotheses.hypotheses)} hypotheses. Top: {hypotheses.hypotheses[0].diagnosis}"
    trace = _create_trace("SymptomRAGAnalyzer", start, in_sum, out_sum, iteration)
    
    # If the RAG completely failed to find evidence, we might want to halt early.
    # We pass it downstream for now.
    
    return {
        "hypotheses": hypotheses,
        "trace": [trace]
    }
