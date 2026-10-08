from mediagent_lite.graph.orchestrator import build_graph

def main():
    graph = build_graph()
    state = {
        'raw_input': '72yo F presents with sudden right-sided weakness, facial droop, and slurred speech starting 45 mins ago.',
        'structured_case': None,
        'hypotheses': None,
        'evidence_bundles': [],
        'final_report': None,
        'trace': [],
        'query_history': [],
        'iteration': 0,
        'config_flags': {}
    }
    
    print("Running graph... (This will call Gemini, Qdrant, and PubMed)")
    result = graph.invoke(state)
    
    print('\n--- FINAL TRACE ---')
    for t in result.get('trace', []):
        print(f"[{t['iteration']}] {t['node']} ({t['latency_ms']}ms): {t['outputs_summary']}")
        
    print('\n--- TOP DIAGNOSIS ---')
    if result.get('final_report') and result['final_report'].diagnoses:
        top = result['final_report'].diagnoses[0]
        print(f"Diagnosis: {top.diagnosis}")
        print(f"Confidence: {top.confidence:.2f}")
        print(f"Explanation: {top.explanation}")
    else:
        print("No report generated.")

if __name__ == "__main__":
    main()
