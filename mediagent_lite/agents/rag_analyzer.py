"""Symptom RAG Analyzer Agent.

Responsibility: Take a structured case, build a query, retrieve supporting
chunks from Qdrant, and propose ranked diagnoses grounded in those chunks.
"""

from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

from mediagent_lite.llm_factory import get_llm
from mediagent_lite.schemas.clinical import StructuredCase
from mediagent_lite.schemas.hypothesis import (
    DiagnosisHypothesis,
    HypothesisList,
    SupportingChunk,
)
from mediagent_lite.rag.retriever import ClinicalRetriever


class SymptomRAGAnalyzer:
    """Proposes diagnoses based on RAG retrieval."""

    def __init__(self):
        self.llm = get_llm("rag_analyzer")
        self.retriever = ClinicalRetriever()
        
        # We want the LLM to output our HypothesisList schema
        self.chain = self._build_prompt() | self.llm.with_structured_output(HypothesisList)

    def _build_query(self, case: StructuredCase) -> str:
        """Convert structured case into a dense search query."""
        terms = []
        if case.chief_complaint:
            terms.append(case.chief_complaint)
            
        for sym in case.symptoms:
            term = sym.name
            if sym.anatomical_region:
                term = f"{sym.anatomical_region} {term}"
            terms.append(term)
            
        for lab in case.labs:
            if lab.interpretation in ["high", "low", "critical"]:
                terms.append(f"{lab.interpretation} {lab.name}")
                
        # Fallback if somehow empty
        if not terms:
            return "general medical query"
            
        return " ".join(terms)

    def _format_docs_for_prompt(self, docs_with_scores: list[tuple[Document, float]]) -> str:
        """Format retrieved documents into a string for the prompt."""
        formatted = []
        for i, (doc, score) in enumerate(docs_with_scores):
            # We must expose the qdrant_id and score so the LLM can cite them
            qdrant_id = doc.metadata.get("qdrant_id", f"unknown_{i}")
            source = doc.metadata.get("source", "unknown")
            text = doc.page_content.replace("\n", " ")
            
            entry = f"[CHUNK_ID: {qdrant_id} | SCORE: {score:.2f} | SOURCE: {source}]\n{text}\n"
            formatted.append(entry)
            
        return "\n".join(formatted)

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert diagnostic assistant.
I will give you a patient's structured case and a set of retrieved medical text chunks.

Your job is to propose a ranked list of possible diagnoses.

CRITICAL RULES:
1. You MUST ONLY propose a diagnosis if it is supported by at least one of the provided chunks.
2. For each diagnosis, you MUST cite the specific `chunk_id` of the supporting chunk(s), the exact `similarity_score` provided, and a brief `text_span` from the chunk that justifies the diagnosis.
3. If the retrieved chunks do not support any diagnosis for the patient's symptoms, do not hallucinate one.
4. Rank the diagnoses by likelihood (1 = most likely).
5. YOU MUST LIMIT YOUR OUTPUT TO A MAXIMUM OF 3 HYPOTHESES. DO NOT GENERATE MORE THAN 3.

CRITICAL: You MUST respond with valid JSON matching this exact structure:
{{
  "hypotheses": [
    {{
      "diagnosis": "Disease A",
      "rank": 1,
      "reasoning": "Symptoms match...",
      "supporting_chunks": [
        {{
          "chunk_id": "chunk-123",
          "text_span": "text from chunk...",
          "similarity_score": 0.85
        }}
      ]
    }},
    {{
      "diagnosis": "Disease B",
      "rank": 2,
      "reasoning": "Also possible...",
      "supporting_chunks": [
        {{
          "chunk_id": "chunk-456",
          "text_span": "different chunk text...",
          "similarity_score": 0.62
        }}
      ]
    }}
  ]
}}
"""),
            ("human", """Patient Case:
{case_json}

Retrieved Medical Text Chunks:
{context}

Based ONLY on the retrieved chunks, propose a ranked list of diagnoses.
""")
        ])

    def _filter_unsupported_hypotheses(self, hypothesis_list: HypothesisList) -> HypothesisList:
        """Rule engine: silently drop hypotheses that failed to cite a chunk.
        
        Why do this here and not in Pydantic? 
        Because we don't want to fail the entire validation chain just because 
        the LLM included one hallucinated diagnosis at rank 5. We just drop it.
        """
        valid_hyps = []
        for h in hypothesis_list.hypotheses:
            if len(h.supporting_chunks) > 0:
                valid_hyps.append(h)
                
        # Re-rank them sequentially
        for i, h in enumerate(valid_hyps):
            h.rank = i + 1
            
        if not valid_hyps:
            # If the LLM hallucinated everything, we return a fallback rather than crashing
            fallback_chunk = SupportingChunk(
                chunk_id="none",
                text_span="No supporting evidence found in vector DB.",
                similarity_score=0.0
            )
            fallback_hyp = DiagnosisHypothesis(
                diagnosis="Unknown (No Retrieval Support)",
                rank=1,
                reasoning="The retrieved context did not contain information matching the patient's symptoms.",
                supporting_chunks=[fallback_chunk]
            )
            valid_hyps.append(fallback_hyp)
            
        return HypothesisList(hypotheses=valid_hyps)

    def process(
        self, 
        case: StructuredCase, 
        config_flags: dict[str, Any],
        override_query: Optional[str] = None,
        config: Optional[RunnableConfig] = None
    ) -> HypothesisList:
        """Execute RAG analysis."""
        
        # 1. Build Query (or use override from optimizer loop)
        query = override_query if override_query else self._build_query(case)
        
        # 2. Retrieve
        docs_with_scores = self.retriever.retrieve(query, config_flags)
        
        # If retrieval completely fails (even after fallback)
        if not docs_with_scores:
            fallback_chunk = SupportingChunk(
                chunk_id="none",
                text_span="No documents retrieved.",
                similarity_score=0.0
            )
            return HypothesisList(
                hypotheses=[
                    DiagnosisHypothesis(
                        diagnosis="Unknown (Retrieval Failed)",
                        rank=1,
                        reasoning="Vector database returned no results for the query.",
                        supporting_chunks=[fallback_chunk]
                    )
                ]
            )
            
        # 3. Format Context
        context = self._format_docs_for_prompt(docs_with_scores)
        
        # 4. Generate Hypotheses
        result: HypothesisList = self.chain.invoke(
            {
                "case_json": case.model_dump_json(exclude_none=True),
                "context": context
            },
            config=config
        )
        
        # 5. Filter ungrounded hallucinations
        filtered_result = self._filter_unsupported_hypotheses(result)
        
        return filtered_result
