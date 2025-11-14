import os
import sys
from pathlib import Path

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from llm_client import OllamaClient, OpenAIClient
from rag import default_rag_client


def _init_clients():
    """Initialize LLM clients and RAG system."""
    from openai import OpenAI

    # Choose which model to use for RAG generation
    # Option 1: Use Ollama for RAG (recommended for local testing)
    rag_model = "llama3.2:3b"  # Change to your preferred Ollama model
    rag_llm_client = OllamaClient(model=rag_model, base_url="http://localhost:11434")
    rag_client = default_rag_client(
        llm_client=rag_llm_client, model_name=rag_model, logdir="logs"
    )

    # Option 2: Use OpenAI for RAG (uncomment to use)
    # openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # rag_llm_client = OpenAIClient(client=openai_client, model="gpt-4o")
    # rag_client = default_rag_client(llm_client=rag_llm_client, model_name="gpt-4o", logdir="logs")

    # Choose which model to use for evaluation metrics
    # Option 1: Use Ollama for evaluation (recommended for local testing)
    ollama_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require a real API key
    )
    eval_llm = llm_factory("llama3.2:3b", client=ollama_client)

    # Option 2: Use Anthropic Claude for evaluation (uncomment to use)
    # from anthropic import Anthropic
    # anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # eval_llm = llm_factory("claude-3-5-sonnet-20241022", client=anthropic_client)

    # Option 3: Use Google Gemini for evaluation (uncomment to use)
    # import google.generativeai as genai
    # genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    # eval_llm = llm_factory("gemini-1.5-pro", client=genai)

    # Option 4: Use OpenAI for evaluation (uncomment to use)
    # openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # eval_llm = llm_factory("gpt-4o", client=openai_client)

    return rag_client, eval_llm


rag_client, llm = _init_clients()


def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )

    data_samples = [
        {
            "question": "What is ragas 0.3",
            "grading_notes": "- experimentation as the central pillar - provides abstraction for datasets, experiments and metrics - supports evals for RAG, LLM workflows and Agents",
        },
        {
            "question": "how are experiment results stored in ragas 0.3?",
            "grading_notes": "- configured using different backends like local, gdrive, etc - stored under experiments/ folder in the backend storage",
        },
        {
            "question": "What metrics are supported in ragas 0.3?",
            "grading_notes": "- provides abstraction for discrete, numerical and ranking metrics",
        },
    ]

    for sample in data_samples:
        row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
