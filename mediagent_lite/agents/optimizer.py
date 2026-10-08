"""Adaptive Query Optimizer Agent.

Responsibility: If the Fusion Agent's confidence is too low, this agent
rewrites the search query to try and find better evidence in the next iteration.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from mediagent_lite.llm_factory import get_llm
from mediagent_lite.schemas.clinical import StructuredCase
from mediagent_lite.schemas.report import FusedReport


class QueryOptimizer:
    """Rewrites queries for the feedback loop."""

    def __init__(self):
        # We use a fast, lightweight model (like Llama 8B) for simple rewriting
        self.llm = get_llm("optimizer")
        
        self.chain = ChatPromptTemplate.from_messages([
            ("system", """You are a medical search optimizer.
The previous search failed to yield high confidence diagnoses.
Based on the patient case and the failed report, write ONE new, improved search query.
Focus on different synonyms, broader categories, or specific combinations of the symptoms.
Return ONLY the raw query string. No quotes, no explanation."""),
            ("human", "Case:\n{case}\n\nFailed Report:\n{report}\n\nPrevious Queries:\n{history}\n\nNew Query:")
        ]) | self.llm

    def process(
        self, 
        case: StructuredCase, 
        report: FusedReport, 
        query_history: list[str],
        config: Optional[RunnableConfig] = None
    ) -> str:
        """Generate a new query."""
        
        result = self.chain.invoke({
            "case": case.model_dump_json(exclude_none=True),
            "report": report.model_dump_json(exclude_none=True),
            "history": "\n".join(query_history) if query_history else "None"
        }, config=config)
        
        return result.content.strip().strip('"').strip("'")
