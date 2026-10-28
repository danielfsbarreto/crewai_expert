# CrewAI Expert

A CrewAI-powered RAG (Retrieval-Augmented Generation) system that answers questions about CrewAI documentation using multi-agent collaboration.

## Overview

This project implements a sophisticated question-answering system for CrewAI documentation with two main capabilities:

1. **Document Processing**: Fetches CrewAI documentation from GitHub, chunks content, and creates vector embeddings stored in MongoDB
2. **Question Answering**: Uses a multi-agent crew (linguist + CrewAI expert) to provide accurate, contextual answers to CrewAI-related questions

## Architecture

- **Flow-based Design**: Built with CrewAI Flows for orchestration
- **Vector Search**: MongoDB Atlas Vector Search over documentation
- **Multi-agent Crew**: Linguist for language detection + CrewAI expert for domain knowledge
- **Async Processing**: Concurrent document chunking and embedding generation

## Prerequisites

- Python >=3.10, <3.14
- OpenAI API key
- Anthropic API key
- MongoDB Atlas cluster (connection string, database, and collection with vector search)
- GitHub token (optional, for higher API rate limits)

## Environment Setup

Set the following environment variables:

```bash
ANTHROPIC_API_KEY=your_model_key
OPENAI_API_KEY=your_openai_key
MONGODB_CONNECTION_STRING=your_mongo_connection_string
MONGODB_DATABASE_NAME=your_mongo_database
MONGODB_COLLECTION_NAME=your_mongo_collection
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

This fetches CrewAI docs from GitHub, processes them, and stores embeddings in MongoDB.

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

- crewai[anthropic, tools]>=1.15.14,<1.16
- crewai-tools[mongodb]
- semchunk>=3.2.3
- tiktoken<0.9.dev0
- tqdm>=4.67.1

## License

See project license file.
