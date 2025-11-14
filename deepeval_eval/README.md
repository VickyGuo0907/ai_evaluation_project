# DeepEval with Local Ollama Model - Sample Project

This project demonstrates how to use DeepEval for LLM evaluation with a local Ollama model (gpt-oss:20b). It includes examples for both LLM as Judge evaluation and RAG (Retrieval-Augmented Generation) evaluation.

## Features

- **LLM as Judge Evaluation**: Evaluate LLM responses using various metrics
  - Answer Relevancy
  - Faithfulness
  - Hallucination Detection
  - Bias and Toxicity
  - Batch Evaluation

- **RAG Evaluation**: Build and evaluate a complete RAG system
  - Vector storage with ChromaDB
  - Document retrieval
  - Response generation
  - Contextual Relevancy
  - Contextual Precision and Recall
  - Faithfulness to retrieved context

## Prerequisites

1. **Ollama**: Install and run Ollama
   ```bash
   # Install Ollama (macOS)
   brew install ollama

   # Or download from https://ollama.ai

   # Start Ollama service
   ollama serve
   ```

2. **Python 3.8+**: Make sure you have Python installed
   ```bash
   python --version
   ```

## Installation

1. Clone or navigate to this directory:
   ```bash
   cd deepeval_eval
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv

   # Activate on macOS/Linux
   source venv/bin/activate

   # Activate on Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull the Ollama model:
   ```bash
   ollama pull gpt-oss:20b
   ```

## Configuration

The project uses environment variables configured in the `.env` file. The default configuration is already set up:

```env
# Ollama Configuration
OLLAMA_HOST=http://127.0.0.1:11434

# DeepEval Configuration
DEEPEVAL_RESULTS_FOLDER=./deepeval_results
DEEPEVAL_TELEMETRY_OPT_OUT=1
```

You can modify these settings as needed.

## Usage

### 1. LLM as Judge Evaluation

Run the LLM as Judge evaluation examples:

```bash
python llm_as_judge_evaluation.py
```

This script demonstrates:
- **Example 1**: Basic answer relevancy evaluation
- **Example 2**: Faithfulness evaluation (checking if output is grounded in context)
- **Example 3**: Hallucination detection
- **Example 4**: Bias and toxicity evaluation
- **Example 5**: Batch evaluation with multiple test cases

### 2. RAG Evaluation

Run the RAG evaluation examples:

```bash
python rag_evaluation.py
```

This script demonstrates:
- **Example 1**: Basic RAG system evaluation
- **Example 2**: Faithfulness of RAG responses
- **Example 3**: Contextual relevancy (are retrieved docs relevant?)
- **Example 4**: Contextual precision and recall
- **Example 5**: Comprehensive evaluation with multiple queries

## Project Structure

```
ai-evaluation/
├── .env                          # Environment configuration
├── requirements.txt              # Python dependencies
├── llm_as_judge_evaluation.py   # LLM as Judge examples
├── rag_evaluation.py             # RAG evaluation examples
├── deepeval_results/             # Evaluation results directory
└── README.md                     # This file
```

## Understanding the Code

### Custom Ollama Model Wrapper

Both scripts include a custom `OllamaModel` class that wraps the Ollama API to work with DeepEval:

```python
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
```

### DeepEval Metrics

The project uses various DeepEval metrics:

1. **AnswerRelevancyMetric**: Measures if the answer is relevant to the input
2. **FaithfulnessMetric**: Checks if the output is grounded in the provided context
3. **HallucinationMetric**: Detects hallucinations in the output
4. **BiasMetric**: Evaluates potential bias in responses
5. **ToxicityMetric**: Measures toxicity in responses
6. **ContextualRelevancyMetric**: Evaluates if retrieved context is relevant
7. **ContextualPrecisionMetric**: Measures precision of retrieved context
8. **ContextualRecallMetric**: Measures recall of expected context

### RAG System Components

The RAG system includes:

1. **Vector Store**: ChromaDB for document storage and retrieval
2. **Document Retrieval**: Semantic search using embeddings
3. **Response Generation**: Ollama LLM with retrieved context
4. **Evaluation**: DeepEval metrics for comprehensive assessment

## Example Output

### LLM as Judge Evaluation
```
================================================================================
EXAMPLE 1: Basic Answer Relevancy Evaluation
================================================================================

Input: What is the capital of France?
Actual Output: The capital of France is Paris. Paris is known for the Eiffel Tower...
Score: 0.95
Reason: The answer directly addresses the question and provides relevant information
Success: True
```

### RAG Evaluation
```
================================================================================
EXAMPLE 1: Basic RAG System Evaluation
================================================================================

Query: What is supervised learning?

Retrieved Documents:
  1. Supervised learning is a type of machine learning where the model is trained...

Generated Response: Supervised learning is a machine learning approach that uses labeled data...

--- Answer Relevancy Evaluation ---
Score: 0.92
Reason: The response accurately answers the query using the retrieved context
Success: True
```

## Customization

### Using Different Models

To use a different Ollama model, modify the model name:

```python
custom_model = OllamaModel(model_name="llama2:13b")
```

### Adding Custom Documents

In `rag_evaluation.py`, you can add your own documents:

```python
documents = [
    "Your custom document 1",
    "Your custom document 2",
    # ... more documents
]
rag_system.add_documents(documents)
```

### Adjusting Evaluation Thresholds

Modify the threshold parameter in metrics:

```python
metric = AnswerRelevancyMetric(
    threshold=0.8,  # Increase for stricter evaluation
    model=custom_model,
    include_reason=True
)
```

## Troubleshooting

### Ollama Connection Error
```
Error: Could not connect to Ollama
```
**Solution**: Make sure Ollama is running with `ollama serve`

### Model Not Found
```
Warning: Model 'gpt-oss:20b' not found
```
**Solution**: Pull the model with `ollama pull gpt-oss:20b`

### ChromaDB Errors
```
Error: Collection already exists
```
**Solution**: The script handles this automatically. If issues persist, restart Python.

### Slow Evaluation
The local LLM may take time for evaluation. Consider:
- Using a smaller model for faster results
- Reducing the number of test cases
- Running evaluations in smaller batches

## Performance Considerations

- **Model Size**: gpt-oss:20b is a large model. Ensure you have sufficient RAM (recommended: 16GB+)
- **Evaluation Time**: Local LLM evaluation takes longer than API-based evaluation
- **Batch Size**: Start with small batches to gauge performance

## Resources

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## License

This is a sample project for educational purposes.

## Contributing

Feel free to extend this project with additional examples or metrics!

## Next Steps

1. Try modifying the test cases with your own data
2. Experiment with different Ollama models
3. Add custom evaluation metrics
4. Integrate with your own RAG applications
5. Build automated evaluation pipelines

Happy evaluating!
