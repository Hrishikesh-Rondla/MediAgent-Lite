"""Streamlit UI for MediAgent-Lite.

This provides an interactive chat-like interface to run cases through
the LangGraph orchestrator and visualize the internal trace.
"""

import streamlit as st
import traceback

from langchain_core.runnables import RunnableConfig

from mediagent_lite.graph.orchestrator import build_graph
from mediagent_lite.config.settings import get_settings


st.set_page_config(
    page_title="MediAgent-Lite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state (but build the graph fresh every time to avoid cached configuration errors)
st.session_state.graph = build_graph()
st.session_state.settings = get_settings()

if "history" not in st.session_state:
    st.session_state.history = []

import os

# --- Sidebar ---
with st.sidebar:
    st.title("🩺 MediAgent-Lite")
    st.markdown("""
    A teaching-oriented implementation of an adaptive Multi-Agent CDSS.
    
    **Architecture:**
    1. **Clarifier**: Extracts structured data.
    2. **RAG**: Proposes diagnoses from local DB.
    3. **Scanner**: Fetches PubMed evidence.
    4. **Fusion**: Synthesizes and scores.
    5. **Optimizer**: Rewrites query if score < tau.
    """)
    
    st.markdown("---")
    st.markdown("### Config overrides (Runtime)")
    # Allow overriding some config dynamically
    current_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    provider_options = ["gemini", "groq", "ollama", "fake"]
    idx = provider_options.index(current_provider) if current_provider in provider_options else 0
    
    provider = st.selectbox("LLM Provider", provider_options, index=idx)
    
    # If the user changed the provider in the UI, update the env var and rebuild graph
    if provider != current_provider:
        os.environ["DEFAULT_LLM_PROVIDER"] = provider
        st.session_state.graph = build_graph()
        st.rerun()
        
    st.markdown("---")
    st.markdown("### Demo Cases")
    if st.button("Cardiology Case"):
        st.session_state.demo_prompt = "65yo M c/o sudden crushing chest pain radiating to jaw, diaphoresis. BP 160/90, HR 110."
    if st.button("Neurology Case"):
        st.session_state.demo_prompt = "72yo F presents with sudden right-sided weakness, facial droop, and slurred speech starting 45 mins ago."
    
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# --- Main Layout ---
st.title("Clinical Case Analysis")

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            # Display final report
            st.markdown("### Final Diagnostic Report")
            report = msg["report"]
            if report and report.diagnoses:
                for i, dx in enumerate(report.diagnoses[:3]):
                    with st.expander(f"#{i+1}: {dx.diagnosis} (Confidence: {dx.confidence:.2f})", expanded=(i==0)):
                        st.progress(dx.confidence)
                        st.markdown(f"**Explanation:** {dx.explanation}")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RAG Score", f"{dx.retrieval_similarity:.2f}")
                        col2.metric("Evidence Score", f"{dx.weighted_evidence_score:.2f}")
                        col3.metric("Concordance", f"{dx.concordance:.2f}")
                        
                        if dx.has_conflict:
                            st.warning(f"⚠️ Conflict Detected: {dx.conflict_detail}")
            else:
                st.error("No diagnoses could be determined.")
                
            # Display trace
            if msg.get("trace"):
                with st.expander("🔍 View Internal Graph Trace", expanded=False):
                    for t in msg["trace"]:
                        st.markdown(f"**[{t['iteration']}] {t['node']}** ({t['latency_ms']}ms)")
                        st.text(f"In:  {t['inputs_summary']}\nOut: {t['outputs_summary']}")


# Input area
prompt = st.chat_input("Enter clinical case (e.g., '45yo M c/o sudden severe chest pain...'):")
if "demo_prompt" in st.session_state:
    prompt = st.session_state.demo_prompt
    del st.session_state.demo_prompt

if prompt:
    
    # Add user message
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Process
    with st.chat_message("assistant"):
        with st.status("Analyzing case...", expanded=True) as status:
            try:
                # Set up the initial state
                initial_state = {
                    "raw_input": prompt,
                    "structured_case": None,
                    "hypotheses": None,
                    "evidence_bundles": [],
                    "final_report": None,
                    "trace": [],
                    "query_history": [],
                    "iteration": 0,
                    "config_flags": {}
                }
                
                final_state = None
                for output in st.session_state.graph.stream(initial_state, stream_mode="values"):
                    # When stream_mode="values", output is the full state dictionary at that step
                    final_state = output
                    # Find the latest trace entry to show what happened
                    if "trace" in output and output["trace"]:
                        latest_trace = output["trace"][-1]
                        st.write(f"✅ Step completed.")
                        st.caption(f"↳ {latest_trace['outputs_summary']} ({latest_trace['latency_ms']}ms)")
                            
                status.update(label="Analysis complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Error occurred", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())
                final_state = None
                
        if final_state is not None:
            try:
                full_state = final_state
                
                report = full_state.get("final_report")
                trace = full_state.get("trace", [])
                
                st.session_state.history.append({
                    "role": "assistant",
                    "report": report,
                    "trace": trace
                })
                st.rerun()
            except Exception as e:
                st.error("Failed to generate final report.")
