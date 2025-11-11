# CrewAI Expert

A CrewAI-powered RAG (Retrieval-Augmented Generation) system that answers questions about CrewAI documentation using multi-agent collaboration.

## Overview

This project implements a sophisticated question-answering system for CrewAI documentation with two main capabilities:

1. **Document Processing**: Fetches CrewAI documentation from GitHub, chunks content, and creates vector embeddings stored in Qdrant
2. **Question Answering**: Uses a multi-agent crew (linguist + CrewAI expert) to provide accurate, contextual answers to CrewAI-related questions

## Architecture

- **Flow-based Design**: Built with CrewAI Flows for orchestration
- **Vector Search**: Qdrant-powered semantic search over documentation
- **Multi-agent Crew**: Linguist for language detection + CrewAI expert for domain knowledge
- **Async Processing**: Concurrent document chunking and embedding generation

## Prerequisites

- Python >=3.10, <3.14
- OpenAI API key
- Qdrant instance (URL and API key)
- GitHub token (optional, for accessing private repos)

## Environment Setup

Set the following environment variables:

```bash
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION_PREFIX=crewai_docs
GITHUB_AUTH_KEY=your_github_token
```

## Installation

```bash
pip install uv
uv pip install -e .
```

## Usage

### Update Documentation Embeddings

```bash
crewai run --inputs='{"run_type": "update_embeddings"}'
```

This fetches CrewAI docs from GitHub, processes them, and stores embeddings in Qdrant.

### Answer Questions

```bash
crewai run --inputs='{"run_type": "answer_prompt", "prompt": "How do I create custom tools in CrewAI?"}'
```

### Direct Execution

```python
from crewai_expert.main import kickoff
kickoff()
```

## Project Structure

```
src/crewai_expert/
├── main.py                 # Flow orchestration
├── clients/
│   └── github_client.py    # GitHub API integration
├── crews/
│   └── answer_crewai_prompt/
│       ├── answer_crewai_prompt_crew.py
│       └── config/
│           ├── agents.yaml # Agent definitions
│           └── tasks.yaml  # Task definitions
├── services/
│   └── doc_files_chunking_service.py  # Document processing
├── tools/
│   └── custom_tool.py      # CrewAI tools
├── types/                  # Pydantic models
└── utils/
    └── mdx_chunker.py      # MDX document chunking
```

## Dependencies

- crewai[google-genai,tools]==1.4.1
- openai>=1.109.1
- qdrant-client>=1.15.1
- semchunk>=3.2.3
- tiktoken>=0.11.0

## License

See project license file.
