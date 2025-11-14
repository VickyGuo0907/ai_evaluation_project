"""
Simple test script to verify Ollama integration with RAG system
"""

from llm_client import OllamaClient
from rag import ExampleRAG, SimpleKeywordRetriever

# Test documents
test_docs = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the human brain.",
]

print("=" * 60)
print("Testing Ollama Integration with RAG System")
print("=" * 60)

# Initialize Ollama client
print("\n1. Initializing Ollama client...")
try:
    llm_client = OllamaClient(model="llama3.2:3b", base_url="http://localhost:11434")
    print("   ✓ Ollama client initialized successfully")
except Exception as e:
    print(f"   ✗ Error initializing Ollama client: {e}")
    exit(1)

# Create RAG system
print("\n2. Creating RAG system...")
try:
    retriever = SimpleKeywordRetriever()
    rag_client = ExampleRAG(
        llm_client=llm_client,
        retriever=retriever,
        logdir="logs",
        model_name="llama3.2:3b",
    )
    print("   ✓ RAG system created successfully")
except Exception as e:
    print(f"   ✗ Error creating RAG system: {e}")
    exit(1)

# Add documents
print("\n3. Adding test documents...")
try:
    rag_client.add_documents(test_docs)
    print(f"   ✓ Added {len(test_docs)} documents")
except Exception as e:
    print(f"   ✗ Error adding documents: {e}")
    exit(1)

# Test query
print("\n4. Testing RAG query...")
test_query = "What is Python?"
print(f"   Query: '{test_query}'")

try:
    response = rag_client.query(test_query, top_k=2)
    print(f"   ✓ Query successful!")
    print(f"\n   Answer: {response['answer']}")
    print(f"\n   Log file: {response['logs']}")
except Exception as e:
    print(f"   ✗ Error during query: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Ollama integration is working correctly.")
print("=" * 60)
