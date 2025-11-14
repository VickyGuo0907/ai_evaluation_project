# AI Evaluation Project

This project demonstrates how to evaluate Large Language Models (LLMs) and RAG (Retrieval-Augmented Generation) systems using three popular evaluation frameworks: **DeepEval**, **MLflow**, and **Ragas**. All evaluations can be performed using local models via Ollama or cloud-based LLM providers.

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
| **Local LLM Support** | Yes (Ollama) | Yes (various) | Yes (Ollama) |
| **Best For** | Comprehensive LLM testing | End-to-end ML/LLM workflows | RAG-specific evaluation |

## Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **Ollama** (for local LLM evaluation)
   ```bash
   # macOS
   brew install ollama

   # Or download from https://ollama.ai

   # Start Ollama service
   ollama serve
   ```

3. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

### Optional: Cloud LLM API Keys

If you prefer cloud-based models over local Ollama models:

- **OpenAI**: Get API key from https://platform.openai.com
- **Anthropic Claude**: Get API key from https://console.anthropic.com
- **Google Gemini**: Get API key from https://ai.google.dev

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

### 4. Pull Ollama Models (if using local models)

```bash
# For DeepEval examples
ollama pull gpt-oss:20b

# For Ragas examples
ollama pull llama2

# Alternative models you can try
ollama pull mistral
ollama pull mixtral
ollama pull phi
```

### 5. Configure Environment Variables (Optional)

If using cloud LLM providers, create a `.env` file in each subdirectory:

```bash
# For OpenAI
OPENAI_API_KEY=your-openai-key

# For Anthropic
ANTHROPIC_API_KEY=your-anthropic-key

# For Google Gemini
GOOGLE_API_KEY=your-google-key
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
# Start MLflow UI (optional)
./mlflow_server.sh
# Run evaluation
python rag_evaluation.py
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
│   ├── .env                       # MLflow configuration
│   ├── mlflow_server.sh           # MLflow UI server script
│   ├── rag_evaluation.py          # RAG evaluation with MLflow
│   ├── mlflow.db                  # MLflow tracking database
│   ├── mlruns/                    # MLflow experiment data
│   └── eval_data/                 # Evaluation datasets
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

**Key Features**:
- Integrated experiment tracking
- Model versioning and deployment
- Rich web UI for exploration
- Production-ready MLOps platform
- Custom metric support
- Distributed evaluation

**Use Cases**:
- End-to-end ML/LLM workflow management
- Team collaboration on model evaluation
- Production model deployment
- Long-term experiment tracking
- A/B testing of models

**Getting Started**:
```bash
cd mlflow_eval
# Start UI (optional)
./mlflow_server.sh
# Run evaluation
python rag_evaluation.py
# View results at http://localhost:5000
```

**MLflow UI**: Access the tracking UI at `http://localhost:5000` to explore experiments, compare runs, and visualize metrics.

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

All three frameworks support multiple LLM providers:

**Ollama (Local)**:
```python
# Already configured in all examples
# Just ensure Ollama is running
```

**OpenAI**:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
# Update model references to "gpt-4o" or "gpt-3.5-turbo"
```

**Anthropic Claude**:
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key"
# Update model references to "claude-3-5-sonnet-20241022"
```

### Adding Custom Metrics

Each framework supports custom metrics:
- **DeepEval**: Extend `BaseMetric` class
- **MLflow**: Use `mlflow.evaluate()` with custom metric functions
- **Ragas**: Create custom `Metric` classes

See individual README files for examples.

### Modifying Test Datasets

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

## Resources

### Documentation
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Ragas Documentation](https://docs.ragas.io)
- [Ollama Documentation](https://ollama.ai/docs)

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