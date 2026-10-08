"""Re-export AgentState from schemas for backward compatibility."""
from mediagent_lite.schemas.state import AgentState, TraceEntry

__all__ = ["AgentState", "TraceEntry"]
