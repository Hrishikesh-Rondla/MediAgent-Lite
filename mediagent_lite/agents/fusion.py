"""Fusion Agent.

Responsibility: Take the retrieved RAG hypotheses and the dynamically
scanned PubMed evidence, detect conflicts, synthesize a final report,
and calculate a confidence score.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from mediagent_lite.config.settings import get_settings
from mediagent_lite.llm_factory import get_llm
from mediagent_lite.schemas.clinical import StructuredCase
from mediagent_lite.schemas.evidence import EvidenceBundle
from mediagent_lite.schemas.hypothesis import HypothesisList
from mediagent_lite.schemas.report import DiagnosisReport, FusedReport


class FusionAgent:
    """Synthesizes final diagnostic report from RAG and external evidence."""

    def __init__(self):
        # Fusion is complex, it uses the strongest model (Gemini Pro by default)
        self.llm = get_llm("fusion")
        self.settings = get_settings()
        
        self.chain = self._build_prompt() | self.llm.with_structured_output(FusedReport)

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are the Lead Diagnostician (Fusion Node).
You will receive a patient case, hypotheses proposed by a RAG system, and external evidence scanned from PubMed.

Your job:
1. Synthesize this into a final FusedReport.
2. Evaluate conflicts: Does the new PubMed evidence contradict the RAG hypotheses? If so, flag a conflict.
3. Calculate a rough initial confidence score (0.0 to 1.0) for each diagnosis. The system will refine this later, but provide your best clinical estimate based on the evidence.
4. NEVER output a generic symptom (e.g., 'Abdominal pain', 'Chest pain') as a final diagnosis. If the RAG hypothesis is just a symptom, refine it to the underlying disease (e.g., 'Appendicitis', 'Myocardial Infarction').
5. BIOLOGICAL PLAUSIBILITY: You must ruthlessly reject biologically impossible diagnoses. If a RAG hypothesis proposes a female-only disease (e.g., Adnexal Torsion, Ectopic Pregnancy) for a MALE patient, you MUST reject it entirely.
"""),
            ("human", """Patient Case:
{case_json}

RAG Hypotheses:
{hypotheses_json}

PubMed Evidence Scans:
{evidence_json}

Generate the final synthesis report.
""")
        ])

    def _calculate_confidence(self, report: DiagnosisReport, rag_hyp: Optional[dict], pubmed_ev_list: list[EvidenceBundle]) -> float:
        """Applies the deterministic confidence formula from the paper.
        
        C = w1 * S_rag + w2 * S_ev + w3 * S_concordance
        """
        w1, w2, w3 = self.settings.confidence_weights
        
        # 1. RAG Similarity Score (max score of supporting chunks)
        s_rag = 0.0
        if rag_hyp and rag_hyp.get("supporting_chunks"):
            s_rag = max((chunk.get("similarity_score", 0.0) for chunk in rag_hyp["supporting_chunks"]), default=0.0)
            
        # 2. Evidence Score (sum of weights of supporting evidence, capped at 1.0)
        s_ev = 0.0
        for bundle in pubmed_ev_list:
            if report.diagnosis in bundle.weighted_scores:
                s_ev += bundle.weighted_scores[report.diagnosis]
        s_ev = max(0.0, min(1.0, s_ev))
        
        # 3. Concordance (Does RAG and PubMed agree?)
        # Simple heuristic: If s_rag > 0.5 and s_ev > 0.5, concordance is high.
        # If one is high and the other is 0, concordance is low.
        s_concordance = 0.0
        if s_rag > 0.3 and s_ev > 0.3:
            s_concordance = 1.0
        elif s_rag > 0.3 or s_ev > 0.3:
            s_concordance = 0.5
            
        confidence = (w1 * s_rag) + (w2 * s_ev) + (w3 * s_concordance)
        
        # If there's an explicit conflict flagged by the LLM, heavily penalize confidence
        if report.has_conflict:
            confidence *= 0.5
            
        final_conf = max(0.0, min(1.0, confidence))
        
        # FIX: We must overwrite the LLM's hallucinated scores with our deterministic math
        report.retrieval_similarity = s_rag
        report.weighted_evidence_score = s_ev
        report.concordance = s_concordance
        report.confidence = final_conf
        
        return final_conf

    def process(
        self, 
        case: StructuredCase, 
        hypotheses: HypothesisList, 
        evidence_bundles: list[EvidenceBundle],
        config: Optional[RunnableConfig] = None
    ) -> FusedReport:
        """Run the fusion process and calculate deterministic confidence."""
        
        # 1. LLM Synthesis
        result: FusedReport = self.chain.invoke({
            "case_json": case.model_dump_json(exclude_none=True),
            "hypotheses_json": hypotheses.model_dump_json(exclude_none=True),
            "evidence_json": [b.model_dump(exclude_none=True) for b in evidence_bundles]
        }, config=config)
        
        # 2. Overwrite LLM's guessed confidence with our deterministic math formula
        hyp_dict = {h.diagnosis: h.model_dump() for h in hypotheses.hypotheses}
        
        # In EvidenceBundle, weighted_scores holds {diagnosis_name: score}
        # To pass the correct bundle to the calculate_confidence function, we need the bundle itself, 
        # but the schema changes mean a single bundle might contain evidence for multiple hypotheses,
        # or multiple bundles might contain evidence for the same hypothesis.
        # We simplify this by just passing the list of bundles and letting the calculator handle it.
        
        for dx_report in result.diagnoses:
            rag_h = hyp_dict.get(dx_report.diagnosis)
            
            computed_conf = self._calculate_confidence(dx_report, rag_h, evidence_bundles)
            dx_report.confidence = computed_conf
            
        # Sort by the new computed confidence
        result.diagnoses.sort(key=lambda x: x.confidence, reverse=True)
        
        return result
