# RAG Evaluation

Evaluate a RAG (Retrieval Augmented Generation) system with custom metrics

This project now supports **multiple LLM providers** including:
- **Ollama** (local models - recommended for testing)
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic Claude
- Google Gemini

## Quick Start

### 1. Choose Your LLM Provider

#### Option A: Use Ollama (Local Models - No API Key Required)

1. Install Ollama from https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama2
   # Or use other models like: mistral, mixtral, phi, etc.
   ```
3. Start Ollama (it runs on http://localhost:11434 by default)

The project is configured to use Ollama by default!

#### Option B: Use Cloud LLM Providers (Requires API Key)

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Or use Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Run the Evaluation

Using `uv`:

```bash
uv run python evals.py
```

Or using `pip`:

```bash
python evals.py
```

### 4. Export Results to CSV

Using `uv`:

```bash
uv run python export_csv.py
```

Or using `pip`:

```bash
python export_csv.py
```

## Project Structure

```
rag_eval/
├── README.md           # This file
├── pyproject.toml      # Project configuration
├── llm_client.py       # LLM abstraction layer (supports multiple providers)
├── rag.py              # RAG application code
├── evals.py            # Evaluation workflow
├── __init__.py         # Makes this a Python package
├── logs/               # RAG execution logs and traces
└── evals/              # Evaluation-related data
    ├── datasets/       # Test datasets
    └── experiments/    # Experiment results (CSVs saved here)
```

## Architecture

The project uses a modular architecture:

1. **llm_client.py**: Abstraction layer supporting multiple LLM providers
   - `BaseLLMClient`: Base interface for all LLM clients
   - `OpenAIClient`: Wrapper for OpenAI models
   - `OllamaClient`: Wrapper for local Ollama models

2. **rag.py**: RAG system implementation
   - Uses the LLM abstraction to work with any provider
   - Includes retrieval and generation components
   - Full tracing and logging support

3. **evals.py**: Evaluation framework
   - Uses Ragas for metrics and experiments
   - Can use different models for RAG vs evaluation
   - Saves results to CSV for analysis

## Customization

### Modify the LLM Provider

The project supports using different models for:
1. **RAG generation** (answering questions based on documents)
2. **Evaluation metrics** (grading the RAG responses)

Edit `evals.py` to configure your preferred models:

```python
from llm_client import OllamaClient, OpenAIClient
from ragas.llms import llm_factory

# For RAG generation - Choose one:
# Option 1: Ollama (default)
rag_llm_client = OllamaClient(model="llama2", base_url="http://localhost:11434")

# Option 2: OpenAI
from openai import OpenAI
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
rag_llm_client = OpenAIClient(client=openai_client, model="gpt-4o")

# For evaluation metrics - Choose one:
# Option 1: Ollama (default)
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
eval_llm = llm_factory("llama2", client=ollama_client)

# Option 2: Anthropic Claude
from anthropic import Anthropic
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
eval_llm = llm_factory("claude-3-5-sonnet-20241022", client=anthropic_client)

# Option 3: Google Gemini
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
eval_llm = llm_factory("gemini-1.5-pro", client=genai)

# Option 4: OpenAI
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
eval_llm = llm_factory("gpt-4o", client=openai_client)
```

### Customize Test Cases

Edit the `load_dataset()` function in `evals.py` to add or modify test cases.

### Change Evaluation Metrics

Update the `my_metric` definition in `evals.py` to use different grading criteria.

## Documentation

Visit https://docs.ragas.io for more information.
