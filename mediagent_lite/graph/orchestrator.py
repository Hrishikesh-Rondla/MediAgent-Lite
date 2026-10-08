"""LangGraph orchestrator."""

import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from mediagent_lite.agents.evidence_scanner import EvidenceScanner
from mediagent_lite.agents.fusion import FusionAgent
from mediagent_lite.agents.optimizer import QueryOptimizer
from mediagent_lite.config.settings import get_settings

# Import nodes
from mediagent_lite.graph.nodes import _create_trace, clarifier_node, rag_analyzer_node
from mediagent_lite.schemas.state import AgentState

# --- Wrap remaining agents in node functions ---

def evidence_scanner_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.time()
    agent = EvidenceScanner()
    
    case = state["structured_case"]
    hypotheses = state["hypotheses"]
    
    # Run evidence scanner concurrently for each hypothesis
    # This is a major speedup and demonstrates multi-agent parallel execution
    bundles = []
    if hypotheses and hypotheses.hypotheses:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(agent.process, case, h, config)
                for h in hypotheses.hypotheses[:3]  # Only scan top 3 to save time/API calls
            ]
            bundles = [f.result() for f in futures]
            
    trace = _create_trace("EvidenceScanner (Parallel)", start, f"{len(bundles)} hypotheses", f"{sum(len(b.records) for b in bundles)} evidence records", state.get("iteration", 0))
    
    return {
        "evidence_bundles": bundles,
        "trace": [trace]
    }


def fusion_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.time()
    agent = FusionAgent()
    
    report = agent.process(
        state["structured_case"],
        state["hypotheses"],
        state["evidence_bundles"],
        config=config
    )
    
    trace = _create_trace("FusionAgent", start, "RAG + PubMed", f"Top conf: {report.diagnoses[0].confidence:.2f}" if report.diagnoses else "No dx", state.get("iteration", 0))
    
    return {
        "final_report": report,
        "trace": [trace]
    }


def optimizer_node(state: AgentState, config: RunnableConfig) -> dict:
    start = time.time()
    agent = QueryOptimizer()
    
    iteration = state.get("iteration", 0) + 1
    
    new_query = agent.process(
        state["structured_case"],
        state["final_report"],
        state.get("query_history", []),
        config=config
    )
    
    trace = _create_trace("QueryOptimizer", start, f"Iter {iteration}", f"New query: {new_query}", iteration)
    
    return {
        "query_history": [new_query], # Reducer appends
        "iteration": iteration,
        "trace": [trace]
    }

# --- Define Edges ---

def evaluate_confidence(state: AgentState) -> str:
    """Conditional edge logic: Decide whether to loop or end."""
    settings = get_settings()
    tau = settings.tau
    max_iter = settings.max_iter
    
    report = state.get("final_report")
    iteration = state.get("iteration", 0)
    
    # If no report, or max iterations reached, stop
    if not report or not report.diagnoses or iteration >= max_iter:
        return "end"
        
    # Check if the top diagnosis meets the confidence threshold
    top_conf = report.diagnoses[0].confidence
    if top_conf >= tau:
        return "end"
        
    # Otherwise, loop to optimizer
    return "optimize"


# --- Build Graph ---

def build_graph() -> StateGraph:
    """Constructs the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Clarifier", clarifier_node)
    workflow.add_node("RAG", rag_analyzer_node)
    workflow.add_node("Scanner", evidence_scanner_node)
    workflow.add_node("Fusion", fusion_node)
    workflow.add_node("Optimizer", optimizer_node)
    
    # Define normal linear flow
    workflow.set_entry_point("Clarifier")
    workflow.add_edge("Clarifier", "RAG")
    workflow.add_edge("RAG", "Scanner")
    workflow.add_edge("Scanner", "Fusion")
    
    # Define conditional feedback loop
    workflow.add_conditional_edges(
        "Fusion",
        evaluate_confidence,
        {
            "end": END,
            "optimize": "Optimizer"
        }
    )
    
    # Loop back
    workflow.add_edge("Optimizer", "RAG")
    
    # Compile
    return workflow.compile()
