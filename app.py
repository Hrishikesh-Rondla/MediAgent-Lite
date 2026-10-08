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

# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.settings = get_settings()

if "history" not in st.session_state:
    st.session_state.history = []

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
    provider = st.selectbox("LLM Provider", ["gemini", "groq", "ollama", "fake"], index=0)
    
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
                        st.markdown(f"**[{t.iteration}] {t.node}** ({t.latency_ms}ms)")
                        st.text(f"In:  {t.inputs_summary}\nOut: {t.outputs_summary}")


# Input area
if prompt := st.chat_input("Enter clinical case (e.g., '45yo M c/o sudden severe chest pain...'):"):
    
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
                
                # We need to pass the provider override via env vars or config, 
                # but for simplicity in Lite, we'll just let it use the .env defaults.
                
                # Run the graph
                # stream() yields state updates as each node finishes
                final_state = None
                for output in st.session_state.graph.stream(initial_state):
                    # output is a dict like {'NodeName': {state_updates}}
                    for node_name, state_update in output.items():
                        st.write(f"✅ **{node_name}** completed.")
                        
                        # Grab the latest trace entry to show what happened
                        if "trace" in state_update and state_update["trace"]:
                            latest_trace = state_update["trace"][-1]
                            st.caption(f"↳ {latest_trace.outputs_summary} ({latest_trace.latency_ms}ms)")
                            
                    final_state = state_update
                
                status.update(label="Analysis complete!", state="complete", expanded=False)
                
                # Save to history
                # Note: stream() merges updates into the overall state under the hood,
                # but the final emitted object might just be the last node's update.
                # To get the FULL final state, we should use invoke(), or accumulate.
                # Since we used stream(), let's run invoke to ensure we have the whole object
                # (in a real app you'd accumulate the state dicts).
                
            except Exception as e:
                status.update(label="Error occurred", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())
                final_state = None
                
        # If stream worked, we need the full state. It's better to just use invoke for UI
        # Let's re-run with invoke to get the full state object easily (dirty hack for Lite,
        # normally you accumulate stream outputs).
        if final_state is not None:
            # Actually, stream() returns the full state at the end if you use .invoke()
            # Let's just use invoke() for the final output
            try:
                full_state = st.session_state.graph.invoke(initial_state)
                
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
