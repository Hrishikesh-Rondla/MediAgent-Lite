# Error Analysis

Sampled failures from Config E (Full pipeline + loop):

| ID | Category | Notes |
|---|---|---|
| medqa_1 | Model reasoning error | Failed to map symptoms to correct option |
| medqa_2 | Answer-mapping error | Selected wrong letter despite correct text |
| medqa_3 | Retrieval miss | Chunk not in top_k |
| medqa_4 | Wrong evidence | PubMed returned irrelevant abstracts |
| medqa_5 | Schema/repair failure | Clamped values but logic still failed |
