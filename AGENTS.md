 Knowledge Gap Analyzer (KGA)

## Project objective

Develop a Python application that analyzes an internal
technical documentation corpus.

The intended audience is an experienced engineer who
understands common technologies but is unfamiliar with
the company's internal systems.

The long-term objective is to identify missing
organizational knowledge and documentation gaps.

## Current milestone: V0.1

Implement company-specific concept discovery.

### Required capabilities

1. Read UTF-8 Markdown, TXT and AsciiDoc documents.
2. Extract terminology and concepts using an LLM.
3. Classify concepts as:
   - General technical knowledge
   - Company-specific knowledge
   - Uncertain
4. Detect internal applications, identifiers, acronyms,
   incidents, configurations and operational concepts.
5. Preserve source document references and supporting
   text passages.
6. Normalize and deduplicate extracted concepts.
7. Export structured JSON and CSV reports.
8. Provide a command-line interface.

### Architecture

- Python 3.10+
- OpenAI-compatible API client
- Pydantic data models
- Configurable model and API endpoint
- No dependency on Open WebUI internals
- No database required for V0.1

### Important constraints

- Never invent source references.
- Every extracted concept must be traceable to the corpus.
- Do not classify concepts as missing knowledge in V0.1.
- Do not implement a knowledge graph yet.
- Do not modify Open WebUI.
- Do not upload or transmit documents except through
  the explicitly configured LLM endpoint.
- Keep API credentials out of logs and reports.
- Include unit tests.
- Prefer simple, maintainable code.

### Expected output

- concepts.json
- concepts.csv
- Execution summary
- Basic statistics by concept category

### Future milestones

V0.2: Missing explanation detection
V0.3: Knowledge dependency analysis
V0.4: Gap prioritization
V0.5: RAG evaluation and Q&A generation
V0.6: Continuous monitoring


