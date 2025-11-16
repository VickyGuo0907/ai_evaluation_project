# AI Evaluation Project

This project demonstrates how to evaluate Large Language Models (LLMs) and RAG (Retrieval-Augmented Generation) systems using three popular evaluation frameworks: **DeepEval**, **MLflow**, and **Ragas**. All evaluations are primarily designed to work with **local models via Ollama and LM Studio**, though cloud-based LLM providers are also supported as alternatives.

## Table of Contents

- [Overview](#overview)
- [Platform Comparison](#platform-comparison)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Evaluation Frameworks](#evaluation-frameworks)
  - [DeepEval](#deepeval-evaluation)
  - [MLflow](#mlflow-evaluation)
  - [Ragas](#ragas-evaluation)
- [When to Use Which Platform](#when-to-use-which-platform)
- [Resources](#resources)

## Overview

This project provides hands-on examples for evaluating LLM applications using three different frameworks:

- **DeepEval**: Focused on LLM-as-judge evaluations with comprehensive metrics
- **MLflow**: Full ML lifecycle management with LLM evaluation capabilities
- **Ragas**: Specialized in RAG system evaluation with custom metrics

Each framework has unique strengths, and this project helps you understand which one fits your specific use case.

### Local-First LLM Approach

This project is designed with a **local-first philosophy**, primarily using:

- **Ollama**: Fast, lightweight local LLM server with easy model management
- **LM Studio**: User-friendly GUI application with OpenAI-compatible API

**Why Local Models?**
- Privacy and data security
- No API costs
- Full control over model selection
- Offline operation capability
- Faster iteration during development
- No rate limits

Cloud LLM providers (OpenAI, Anthropic, Google) are supported as alternatives, but the examples and default configurations focus on Ollama and LM Studio.

## Platform Comparison

| Feature | DeepEval | MLflow | Ragas |
|---------|----------|---------|-------|
| **Primary Focus** | LLM evaluation metrics | ML lifecycle management | RAG system evaluation |
| **Ease of Use** | High - Simple API | Medium - Full platform | High - RAG-focused |
| **Built-in Metrics** | 15+ metrics | Standard ML + LLM | RAG-specific metrics |
| **Custom Metrics** | Yes | Yes | Yes |
| **Batch Evaluation** | Yes | Yes | Yes |
| **UI Dashboard** | Cloud-based | Local web UI | CSV/DataFrame |
| **Experiment Tracking** | Yes | Yes (extensive) | Yes |
| **Model Deployment** | No | Yes | No |
| **RAG Support** | Yes | Yes | Specialized |
| **Local LLM Support** | Yes (Ollama, LM Studio) | Yes (various) | Yes (Ollama, LM Studio) |
| **Best For** | Comprehensive LLM testing | End-to-end ML/LLM workflows | RAG-specific evaluation |

## Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **Ollama** (for local LLM evaluation - recommended)
   ```bash
   # macOS
   brew install ollama

   # Or download from https://ollama.ai

   # Start Ollama service
   ollama serve
   ```

3. **LM Studio** (alternative local LLM option)
   - Download from https://lmstudio.ai
   - Install and launch the application
   - Download your preferred models through the LM Studio UI
   - Start the local server (default: http://localhost:1234)
   - LM Studio provides an OpenAI-compatible API endpoint

4. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

### Optional: Cloud LLM API Keys

If you prefer cloud-based models as an alternative to local Ollama/LM Studio models:

- **OpenAI**: Get API key from https://platform.openai.com
- **Anthropic Claude**: Get API key from https://console.anthropic.com
- **Google Gemini**: Get API key from https://ai.google.dev

**Note**: Cloud providers are optional. This project is designed to work primarily with Ollama and LM Studio.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ai_evaluation_project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Local Models

**Option A: Using Ollama (Recommended)**

```bash
# For DeepEval examples
ollama pull gpt-oss:20b

# For Ragas examples
ollama pull llama2

# Alternative models you can try
ollama pull llama3
```

**Option B: Using LM Studio**

1. Launch LM Studio application
2. Navigate to the "Discover" tab
3. Search and download models such as:
   - Llama 3 (8B or 70B variants)
   - gemma-3-27b-it
4. Go to the "Local Server" tab
5. Select your downloaded model
6. Click "Start Server" (default: http://localhost:1234)
7. The server provides an OpenAI-compatible API endpoint

### 5. Configure Environment Variables

Create a `.env` file in each subdirectory to configure your LLM provider:

**For Ollama (Primary - Local)**:
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2  # or mistral, mixtral, llama3, etc.

# Optional: Set default model for specific use cases
OLLAMA_EVAL_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**For LM Studio (Primary - Local)**:
```bash
# LM Studio Configuration (OpenAI-compatible)
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio  # LM Studio doesn't require a real key
LM_STUDIO_MODEL=llama-3-8b  # Name of your loaded model in LM Studio

# Alternative: Use generic provider setting
LLM_PROVIDER=lmstudio
LLM_BASE_URL=http://localhost:1234/v1
```

**For OpenAI (Alternative - Cloud)**:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o  # or gpt-3.5-turbo, gpt-4-turbo, etc.
```

**For Anthropic Claude (Alternative - Cloud)**:
```bash
# Anthropic Configuration
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**For Google Gemini (Alternative - Cloud)**:
```bash
# Google Gemini Configuration
GOOGLE_API_KEY=your-google-key
GEMINI_MODEL=gemini-1.5-pro
```

**For MLflow with Ollama (mlflow_eval/)**:
```bash
# Ollama API Configuration
OLLAMA_API_BASE=http://localhost:11434/v1
OLLAMA_API_KEY=no_need

# Model Configuration
RESULT_MODEL_NAME=llama3.2:3b              # Model for generating responses
LLM_AS_JUDGE_MODEL_NAME=openai:/gemma:27b   # Model for evaluation scoring

# MLflow Tracking
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

**Example `.env` file for local-first setup**:
```bash
# Primary: Use Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Backup: LM Studio (if Ollama unavailable)
# LLM_PROVIDER=lmstudio
# OPENAI_API_BASE=http://localhost:1234/v1
# OPENAI_API_KEY=lm-studio
```

## Quick Start

### Test DeepEval

```bash
cd deepeval_eval
python llm_as_judge_evaluation.py
# Or
python rag_evaluation.py
```

### Test MLflow

```bash
cd mlflow_eval

# Make sure Ollama is running with required models
ollama pull llama3.2:3b
ollama pull gemma:27b

# Start MLflow UI (optional but recommended for viewing results)
./mlflow_server.sh

# In a new terminal, run the evaluation
python rag_evaluation.py

# View results at http://localhost:5000
```

### Test Ragas

```bash
cd ragas_eval
python evals.py
# Export results
python export_csv.py
```

## Project Structure

```
ai_evaluation_project/
├── README.md                      # This file
├── requirements.txt               # Common dependencies
│
├── deepeval_eval/                 # DeepEval examples
│   ├── README.md                  # Detailed DeepEval guide
│   ├── .env                       # DeepEval configuration
│   ├── llm_as_judge_evaluation.py # LLM-as-judge examples
│   ├── rag_evaluation.py          # RAG evaluation examples
│   └── deepeval_results/          # Evaluation results
│
├── mlflow_eval/                   # MLflow examples
│   ├── .env                       # MLflow configuration (Ollama/LM Studio)
│   ├── mlflow_server.sh           # MLflow UI server script
│   ├── rag_evaluation.py          # RAG evaluation with MLflow
│   ├── dataset_utils.py           # Dataset management utilities
│   ├── mlflow.db                  # MLflow tracking database (SQLite)
│   ├── mlruns/                    # MLflow experiment data
│   └── eval_data/                 # Evaluation datasets
│       └── test_cases_1.yaml      # YAML test case definitions
│
└── ragas_eval/                    # Ragas examples
    ├── README.md                  # Detailed Ragas guide
    ├── llm_client.py              # Multi-provider LLM client
    ├── rag.py                     # RAG system implementation
    ├── evals.py                   # Evaluation workflow
    ├── test_ollama.py             # Ollama connectivity test
    ├── logs/                      # Execution logs
    └── evals/                     # Evaluation data and results
        ├── datasets/              # Test datasets
        └── experiments/           # Experiment results (CSV)
```

## Evaluation Frameworks

### DeepEval Evaluation

**Location**: `deepeval_eval/`

**What it offers**:
- 15+ pre-built evaluation metrics
- LLM-as-judge evaluation
- RAG system evaluation
- Batch evaluation support
- Bias, toxicity, and hallucination detection

**Key Features**:
- Answer Relevancy
- Faithfulness
- Hallucination Detection
- Contextual Relevancy
- Contextual Precision & Recall
- Bias & Toxicity metrics

**Use Cases**:
- Comprehensive LLM output quality assessment
- Production monitoring with multiple metrics
- Bias and safety evaluation
- Quick prototyping of evaluation pipelines

**Getting Started**:
```bash
cd deepeval_eval
python llm_as_judge_evaluation.py
```

**Learn More**: See [deepeval_eval/README.md](deepeval_eval/README.md) for detailed documentation.

### MLflow Evaluation

**Location**: `mlflow_eval/`

**What it offers**:
- Full ML lifecycle management
- Experiment tracking and versioning
- Model registry and deployment
- Built-in LLM evaluation metrics
- Web UI for visualization
- Trace logging and debugging
- YAML-based dataset management
- Custom and built-in scorers

**Key Features**:
- Integrated experiment tracking with SQLite backend
- Model versioning and deployment
- Rich web UI for exploration
- Production-ready MLOps platform
- Custom metric support (accuracy_scorer)
- Built-in LLM scorers (Correctness, Guidelines)
- YAML-driven test case management
- Local Ollama model integration

**Implementation Details**:

The MLflow evaluation system includes:

1. **Dataset Management** (`dataset_utils.py`):
   - Load test cases from YAML files
   - Create and configure evaluation datasets
   - Automatic dataset versioning and tagging
   - Metadata tracking for test case management

2. **Test Cases** (`eval_data/test_cases_1.yaml`):
   - Structured YAML format for test cases
   - Includes questions, contexts, and expectations
   - Expected responses, facts, and accuracy thresholds
   - Metadata tracking (version, description, last_updated)

3. **Evaluation Scorers** (`rag_evaluation.py`):
   - **Correctness**: LLM-as-judge scoring using local models
   - **Guidelines**: Validates responses against language/quality guidelines
   - **accuracy_scorer**: Custom scorer with exact/partial matching (1.0/0.5/0.0 scale)

4. **Model Configuration**:
   - Uses Ollama for local LLM inference
   - Separate models for generation (`llama3.2:3b`) and judging (`gemma-3-27b-it`)
   - OpenAI-compatible API integration
   - SQLite tracking database for persistent storage

**Use Cases**:
- End-to-end ML/LLM workflow management
- Team collaboration on model evaluation
- Production model deployment
- Long-term experiment tracking
- A/B testing of models
- YAML-driven regression testing
- Local LLM evaluation without cloud dependencies

**Getting Started**:
```bash
cd mlflow_eval

# Configure your models in .env file
# RESULT_MODEL_NAME=llama3.2:3b
# LLM_AS_JUDGE_MODEL_NAME=openai:/gemma-3-27b-it

# Start MLflow UI (optional but recommended)
./mlflow_server.sh

# Run evaluation with test cases
python rag_evaluation.py

# View results at http://localhost:5000
```

**MLflow UI**: Access the tracking UI at `http://localhost:5000` to explore experiments, compare runs, and visualize metrics. The UI provides detailed views of:
- Scorer metrics (Correctness, Guidelines, Accuracy)
- Test case results and comparisons
- Dataset versions and metadata
- Run parameters and artifacts

**Creating Custom Test Cases**:

The YAML test case format allows you to define comprehensive test scenarios:

```yaml
test_cases:
  - inputs:
      question: "What is the capital of France?"
      context: "general_knowledge"
    expectations:
      expected_response: "Paris is the capital of France."
      expected_facts:
        - "Paris"
        - "capital of France"
      tone: "factual"
      must_include: ["Paris", "paris"]
      must_not_include: []
      accuracy: 1.0

metadata:
  version: "1.1"
  last_updated: "2025-11-15"
  description: "Test cases for RAG evaluation"
  total_cases: 6
```

Each test case includes:
- **inputs**: Question and context for the LLM
- **expectations**: Expected response, facts, tone, and validation rules
- **metadata**: Version tracking and dataset description

The `dataset_utils.py` module automatically loads these test cases and creates MLflow datasets with proper versioning and tagging.

### Ragas Evaluation

**Location**: `ragas_eval/`

**What it offers**:
- Specialized RAG evaluation framework
- Multi-provider LLM support (Ollama, OpenAI, Claude, Gemini)
- Custom metric creation
- Dataset-driven evaluation
- CSV export for analysis

**Key Features**:
- RAG-specific metrics (faithfulness, answer relevancy, context precision/recall)
- Flexible LLM provider abstraction
- Easy dataset management
- Automated experiment tracking
- Result export to CSV

**Use Cases**:
- Evaluating RAG applications
- Comparing different retrieval strategies
- Testing different embedding models
- Document retrieval quality assessment
- Custom RAG metric development

**Getting Started**:
```bash
cd ragas_eval
python evals.py
# Export to CSV
python export_csv.py
```

**Learn More**: See [ragas_eval/README.md](ragas_eval/README.md) for detailed documentation.

## When to Use Which Platform

### Choose DeepEval if you need:
- Quick setup for LLM evaluation
- Pre-built metrics for common evaluation tasks
- LLM-as-judge evaluations
- Safety metrics (bias, toxicity)
- Simple batch evaluation
- Cloud dashboard for team collaboration

### Choose MLflow if you need:
- Full ML lifecycle management
- Model versioning and deployment
- Team collaboration features
- Production-grade tracking
- Integration with existing MLOps pipelines
- Long-term experiment management
- Local web UI for exploration
- YAML-based test case management for regression testing
- Custom and built-in scorer combinations
- SQLite-backed persistent storage
- LLM-as-judge evaluation with local models

### Choose Ragas if you need:
- RAG-specific evaluation metrics
- Flexibility in LLM providers
- Custom metric development
- Lightweight evaluation framework
- CSV-based result analysis
- Research and experimentation focus

### Use Multiple Platforms if:
- You want to compare evaluation approaches
- Different teams prefer different tools
- You need both development (Ragas) and production (MLflow) workflows
- You want comprehensive coverage (DeepEval metrics + MLflow tracking + Ragas RAG focus)

## Customization Tips

### Using Different LLM Providers

All three frameworks support multiple LLM providers. **This project primarily uses Ollama and LM Studio for local LLM evaluation**, but cloud providers are also supported.

**Ollama (Local - Primary Option)**:
```python
# Already configured in all examples
# Just ensure Ollama is running
ollama serve
```

**LM Studio (Local - Primary Option)**:
```python
# Start LM Studio server on http://localhost:1234
# Use OpenAI-compatible endpoint
import openai
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"  # LM Studio doesn't require a real key

# Or use with LangChain/frameworks that support OpenAI
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
```

**OpenAI (Cloud - Alternative)**:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
# Update model references to "gpt-4o" or "gpt-3.5-turbo"
```

**Anthropic Claude (Cloud - Alternative)**:
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key"
# Update model references to "claude-3-5-sonnet-20241022"
```

### Adding Custom Metrics

Each framework supports custom metrics:
- **DeepEval**: Extend `BaseMetric` class
- **MLflow**: Use `mlflow.evaluate()` with custom metric functions or create custom `@scorer` decorated functions
- **Ragas**: Create custom `Metric` classes

Example MLflow custom scorer (`mlflow_eval/rag_evaluation.py:48-74`):
```python
@scorer(name="accuracy_scorer")
def accuracy_scorer(outputs: Any, expectations: dict[str, Any]):
    """Custom scorer with exact/partial matching"""
    # Returns 1.0 for exact match, 0.5 for partial, 0.0 for incorrect
    # Implementation details in mlflow_eval/rag_evaluation.py
```

### Modifying Test Datasets

**For MLflow** (YAML-based):
Create or edit YAML files in `mlflow_eval/eval_data/`:
```yaml
test_cases:
  - inputs:
      question: "Your question here"
      context: "category or context"
    expectations:
      expected_response: "Expected answer"
      expected_facts: ["fact1", "fact2"]
      must_include: ["keyword1"]
      must_not_include: ["unwanted"]
```

Then load using `dataset_utils.py`:
```python
from dataset_utils import create_evaluation_dataset

dataset, metadata = create_evaluation_dataset(
    experiment_ids=[experiment_id],
    dataset_name="my_test_set",
    test_case_file_path="eval_data/my_test_cases.yaml"
)
```

**For DeepEval and Ragas**:
Edit the respective Python files to add your own:
- Questions and expected answers
- Documents for RAG systems
- Evaluation criteria
- Threshold values

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### LM Studio Connection Issues

```bash
# Check if LM Studio server is running
curl http://localhost:1234/v1/models

# If not running:
# 1. Open LM Studio application
# 2. Go to "Local Server" tab
# 3. Select a model
# 4. Click "Start Server"
```

**Common LM Studio Issues**:
- **Port conflict**: Change the port in LM Studio settings if 1234 is in use
- **Model not loaded**: Ensure a model is selected before starting the server
- **Slow responses**: LM Studio performance depends on your hardware; use smaller models for faster inference

### Model Not Found

```bash
# List available models
ollama list

# Pull required model
ollama pull llama2
```

### Port Already in Use (MLflow)

```bash
# Change port in mlflow_server.sh
mlflow server --port 5001  # Use different port
```

### Memory Issues with Large Models

- Use smaller models (e.g., `llama2:7b` instead of `gpt-oss:20b`)
- Reduce batch sizes
- Close other applications
- Ensure sufficient RAM (16GB+ recommended for large models)

### MLflow-Specific Issues

**Dataset Loading Errors**:
```bash
# Ensure YAML file exists and is valid
cat mlflow_eval/eval_data/test_cases_1.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('mlflow_eval/eval_data/test_cases_1.yaml'))"
```

**SQLite Database Locked**:
```bash
# Stop any running MLflow servers
pkill -f "mlflow server"

# Restart MLflow UI
cd mlflow_eval
./mlflow_server.sh
```

**Model Not Found in Ollama**:
```bash
# Check available models
ollama list

# Pull required models for MLflow eval
ollama pull llama3.2:3b      # For RESULT_MODEL_NAME
ollama pull gemma:27b         # For LLM_AS_JUDGE_MODEL_NAME
```

**Import Error for dataset_utils**:
```bash
# Make sure you're in the mlflow_eval directory
cd mlflow_eval
python rag_evaluation.py

# Or use absolute imports if running from project root
```

## Resources

### Documentation
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Ragas Documentation](https://docs.ragas.io)
- [Ollama Documentation](https://ollama.ai/docs)
- [LM Studio Documentation](https://lmstudio.ai/docs)

### Related Tools
- [ChromaDB](https://docs.trychroma.com/) - Vector database for RAG
- [LangChain](https://python.langchain.com/) - LLM application framework
- [LlamaIndex](https://docs.llamaindex.ai/) - Data framework for LLMs

### Learning Resources
- [Understanding RAG Systems](https://python.langchain.com/docs/tutorials/rag/)
- [LLM Evaluation Best Practices](https://www.anthropic.com/research/measuring-model-performance)
- [Building Production RAG Systems](https://www.run.ai/guides/machine-learning-engineering/retrieval-augmented-generation)

## Next Steps

1. **Experiment with Examples**: Run all three frameworks and compare results
2. **Try Different Models**: Test various Ollama models or cloud providers
3. **Create Custom Datasets**: Add your own test cases relevant to your domain
4. **Build Custom Metrics**: Develop evaluation metrics specific to your needs
5. **Integrate into Pipelines**: Add evaluation to your CI/CD workflows
6. **Scale Up**: Move from local testing to production monitoring

## Contributing

Contributions are welcome! Feel free to:
- Add new evaluation examples
- Support additional LLM providers
- Improve documentation
- Report issues or suggest features

## License

This project is provided for educational and demonstration purposes.

---

**Happy Evaluating!** Choose the framework that best fits your needs, or use all three to get comprehensive insights into your LLM applications.