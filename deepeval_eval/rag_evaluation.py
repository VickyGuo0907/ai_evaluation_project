"""
RAG (Retrieval-Augmented Generation) Evaluation using DeepEval with Local Ollama Model
This script demonstrates how to build a simple RAG system and evaluate it
using deepeval metrics with a local Ollama model (gpt-oss:20b).
"""

import os
from dotenv import load_dotenv
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM
import ollama

# Load environment variables
load_dotenv()


class OllamaModel(DeepEvalBaseLLM):
    """Custom DeepEval model wrapper for Ollama"""

    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.model_name = model_name

    def load_model(self):
        """Ollama doesn't require explicit loading"""
        return self.model_name

    def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        """Async generation (uses sync for simplicity)"""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """Return the model name"""
        return self.model_name


class SimpleRAGSystem:
    """Simple RAG system using ChromaDB and Ollama"""

    def __init__(self, model_name: str = "gpt-oss:20b", collection_name: str = "rag_docs"):
        self.model_name = model_name
        self.collection_name = collection_name

        # Initialize ChromaDB with default embedding function
        self.client = chromadb.Client()

        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")

    def add_documents(self, documents: List[str], ids: List[str] = None):
        """Add documents to the vector store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            ids=ids
        )
        print(f"Added {len(documents)} documents to the collection")

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant documents for a query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using retrieved context"""
        context_str = "\n\n".join([f"Context {i + 1}: {ctx}" for i, ctx in enumerate(context)])

        prompt = f"""Based on the following context, please answer the question. Only use information from the provided 
        context. If the answer is not in the context, say so.{context_str}
        Question: {query}
        Answer:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def query(self, query: str, n_results: int = 3) -> tuple[str, List[str]]:
        """Full RAG pipeline: retrieve and generate"""
        retrieved_docs = self.retrieve(query, n_results)
        response = self.generate_response(query, retrieved_docs)
        return response, retrieved_docs


def setup_sample_knowledge_base():
    """Setup a sample knowledge base about machine learning"""
    rag_system = SimpleRAGSystem(model_name="gpt-oss:20b", collection_name="ml_knowledge")

    # Sample documents about machine learning
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on enabling systems to learn and improve from experience without being explicitly programmed.",
        "Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs.",
        "Unsupervised learning is a type of machine learning that uses unlabeled data. The algorithm tries to find patterns and structures in the data without predefined labels.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. It's particularly effective for processing images, speech, and text.",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to read, understand, and derive meaning from human languages.",
        "Computer vision is a field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.",
        "Transfer learning is a machine learning technique where a model trained on one task is adapted for use on a different but related task.",
        "A neural network is a computing system inspired by biological neural networks. It consists of interconnected nodes (neurons) organized in layers.",
        "Overfitting occurs when a machine learning model learns the training data too well, including noise and outliers, resulting in poor performance on new data."
    ]

    ids = [f"ml_doc_{i}" for i in range(len(documents))]

    rag_system.add_documents(documents, ids)
    return rag_system


def example_1_basic_rag_evaluation():
    """Example 1: Basic RAG evaluation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic RAG System Evaluation")
    print("=" * 80)

    rag_system = setup_sample_knowledge_base()
    custom_model = OllamaModel(model_name="gpt-oss:20b")

    # Query the RAG system
    query = "What is supervised learning?"
    response, retrieved_docs = rag_system.query(query)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved Documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc[:100]}...")
    print(f"\nGenerated Response: {response}")

    # Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=retrieved_docs
    )

    # Evaluate with Answer Relevancy
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=custom_model,
        include_reason=True
    )

    answer_relevancy_metric.measure(test_case)

    print(f"\n--- Answer Relevancy Evaluation ---")
    print(f"Score: {answer_relevancy_metric.score}")
    print(f"Reason: {answer_relevancy_metric.reason}")
    print(f"Success: {answer_relevancy_metric.is_successful()}")


def example_2_faithfulness_evaluation():
    """Example 2: Evaluate faithfulness of RAG responses"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: RAG Faithfulness Evaluation")
    print("=" * 80)

    rag_system = setup_sample_knowledge_base()
    custom_model = OllamaModel(model_name="gpt-oss:20b")

    query = "What is deep learning and how does it work?"
    response, retrieved_docs = rag_system.query(query)

    print(f"\nQuery: {query}")
    print(f"\nGenerated Response: {response}")

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=retrieved_docs
    )

    # Evaluate faithfulness (is the response grounded in retrieved context?)
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=custom_model,
        include_reason=True
    )

    faithfulness_metric.measure(test_case)

    print(f"\n--- Faithfulness Evaluation ---")
    print(f"Score: {faithfulness_metric.score}")
    print(f"Reason: {faithfulness_metric.reason}")
    print(f"Success: {faithfulness_metric.is_successful()}")


def example_3_contextual_relevancy():
    """Example 3: Evaluate contextual relevancy - are retrieved docs relevant?"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Contextual Relevancy Evaluation")
    print("=" * 80)

    rag_system = setup_sample_knowledge_base()
    custom_model = OllamaModel(model_name="gpt-oss:20b")

    query = "Explain reinforcement learning"
    response, retrieved_docs = rag_system.query(query)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved Documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc}")

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=retrieved_docs
    )

    # Evaluate if retrieved context is relevant to the input
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        model=custom_model,
        include_reason=True
    )

    contextual_relevancy_metric.measure(test_case)

    print(f"\n--- Contextual Relevancy Evaluation ---")
    print(f"Score: {contextual_relevancy_metric.score}")
    print(f"Reason: {contextual_relevancy_metric.reason}")
    print(f"Success: {contextual_relevancy_metric.is_successful()}")


def example_4_contextual_precision_recall():
    """Example 4: Evaluate contextual precision and recall"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Contextual Precision and Recall Evaluation")
    print("=" * 80)

    rag_system = setup_sample_knowledge_base()
    custom_model = OllamaModel(model_name="gpt-oss:20b")

    query = "What is the difference between supervised and unsupervised learning?"
    response, retrieved_docs = rag_system.query(query, n_results=5)

    print(f"\nQuery: {query}")
    print(f"\nGenerated Response: {response}")

    # Expected context should include info about both supervised and unsupervised learning
    expected_context = [
        "Supervised learning is a type of machine learning where the model is trained on labeled data.",
        "Unsupervised learning is a type of machine learning that uses unlabeled data."
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        expected_output="Supervised learning uses labeled data while unsupervised learning uses unlabeled data.",
        retrieval_context=retrieved_docs,
        expected_retrieval_context=expected_context
    )

    # Contextual Precision: measures if nodes in retrieval context are relevant
    precision_metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=custom_model,
        include_reason=True
    )

    # Contextual Recall: measures if expected context is in retrieved context
    recall_metric = ContextualRecallMetric(
        threshold=0.7,
        model=custom_model,
        include_reason=True
    )

    precision_metric.measure(test_case)
    recall_metric.measure(test_case)

    print(f"\n--- Contextual Precision Evaluation ---")
    print(f"Score: {precision_metric.score}")
    print(f"Reason: {precision_metric.reason}")
    print(f"Success: {precision_metric.is_successful()}")

    print(f"\n--- Contextual Recall Evaluation ---")
    print(f"Score: {recall_metric.score}")
    print(f"Reason: {recall_metric.reason}")
    print(f"Success: {recall_metric.is_successful()}")


def example_5_comprehensive_rag_evaluation():
    """Example 5: Comprehensive evaluation with multiple queries"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comprehensive RAG Evaluation")
    print("=" * 80)

    rag_system = setup_sample_knowledge_base()
    custom_model = OllamaModel(model_name="gpt-oss:20b")

    # Multiple test queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is overfitting in machine learning?",
        "How does transfer learning work?"
    ]

    test_cases = []

    print("\nGenerating responses for multiple queries...")
    for query in queries:
        response, retrieved_docs = rag_system.query(query)
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=retrieved_docs
        )
        test_cases.append(test_case)
        print(f"  - {query}")

    # Define multiple metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=custom_model, include_reason=True),
        FaithfulnessMetric(threshold=0.7, model=custom_model, include_reason=True),
        ContextualRelevancyMetric(threshold=0.7, model=custom_model, include_reason=True)
    ]

    # Batch evaluation
    print("\nEvaluating all test cases...")
    results = evaluate(test_cases=test_cases, metrics=metrics)

    print(f"\n--- Evaluation Results Summary ---")
    print(f"Total queries evaluated: {len(test_cases)}")
    print(f"Results: {results}")


if __name__ == "__main__":
    print("=" * 80)
    print("RAG EVALUATION WITH DEEPEVAL AND OLLAMA")
    print("=" * 80)

    # Check if Ollama is running
    try:
        ollama.list()
        print("\nOllama is running!")
    except Exception as e:
        print(f"\nWarning: Could not connect to Ollama. Make sure it's running: {e}")
        print("Start Ollama with: ollama serve")
        exit(1)

    # Check if model exists
    try:
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        if 'gpt-oss:20b' not in model_names:
            print(f"\nWarning: Model 'gpt-oss:20b' not found.")
            print(f"Available models: {model_names}")
            print("\nPull the model with: ollama pull gpt-oss:20b")
            exit(1)
    except Exception as e:
        print(f"Could not check available models: {e}")

    # Run examples
    try:
        example_1_basic_rag_evaluation()
        example_2_faithfulness_evaluation()
        example_3_contextual_relevancy()
        example_4_contextual_precision_recall()
        example_5_comprehensive_rag_evaluation()

        print("\n" + "=" * 80)
        print("ALL RAG EVALUATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
