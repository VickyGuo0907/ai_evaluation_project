import os
from typing import Any
import mlflow
from mlflow import MlflowClient
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from openai import OpenAI
from dotenv import load_dotenv
from dataset_utils import get_or_create_evaluation_dataset

# Load environment variables
load_dotenv()

RESULT_MODEL_NAME = os.getenv("RESULT_MODEL_NAME")
LLM_AS_JUDGE_MODEL_NAME = os.getenv("LLM_AS_JUDGE_MODEL_NAME")

# Set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# local Ollama model
openai_client = OpenAI(
    base_url=os.getenv("OLLAMA_API_BASE"),  # The local Ollama REST endpoint
    api_key=os.getenv(
        "OLLAMA_API_KEY"
    ),  # Required to instantiate OpenAI client, it can be a random string
)


def qa_predict_fn(question: str, context: str | None = None) -> str:
    """
    Simple Q&A prediction function using my Local LLM.

    This function can serve as:
      - a direct LLM call,
      - a RAG application call,
      - or a wrapper around your assistant API.

    MLflow will call this function for every test case.
    """

    # Build system prompt
    system_prompt = "You are a helpful assistant. Answer questions concisely and accurately."

    # If RAG context is provided, include it explicitly
    if context:
        system_prompt += f"\nRelevant Context:\n{context}\n"

    try:
        response = openai_client.chat.completions.create(
            model=RESULT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,  # low temp for reproducible evaluation
        )

        output = response.choices[0].message.content
        return output.strip()

    except Exception as e:
        # ALWAYS return a string — MLflow scorers expect text
        return f"ERROR: {e}"


@scorer(name="accuracy_scorer")
def accuracy_scorer(outputs: Any, expectations: dict[str, Any]):
    """
    Accuracy scorer that returns:
    - 1.0 for exact match
    - 0.5 for partial match (expected facts present)
    - 0.0 for incorrect
    """
    if not isinstance(outputs, str):
        return 0.0

    output = outputs.lower()
    expected = expectations.get("expected_response", "").lower()

    # 1. exact match
    if output.strip() == expected.strip():
        return 1.0

    # 2. partial evaluation using expected_facts
    expected_facts = expectations.get("expected_facts", [])

    if expected_facts:
        match_count = sum(1 for fact in expected_facts if fact.lower() in output)
        if match_count > 0:
            return 0.5

    return 0.0


def get_or_create_experiment(name: str) -> str:
    """Get or Create an MLflow experiment with the given name.
    Args:
        name: Name of the experiment
    Returns:
        str: The experiment ID
    """
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        # Create new experiment if it doesn't exist
        experiment_id = mlflow.create_experiment(name)
        print(f"Created new experiment: {name} (ID: {experiment_id})")
    elif experiment.lifecycle_stage == "deleted":
        # Restore the deleted experiment
        my_client = MlflowClient()
        experiment_id = experiment.experiment_id
        my_client.restore_experiment(experiment_id)
        print(f"Restored deleted experiment: {name} (ID: {experiment_id})")
    else:
        # Use existing active experiment
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {name} (ID: {experiment_id})")

    mlflow.set_experiment(name)
    return experiment_id


def run_evaluation(dataset):
    easy_to_understand = ("The response must use clear, concise language and structure responses logically. "
                          "It must avoid jargon or explain technical terms when used.")
    scorers = [
        # Use Local LLM as judge
        Correctness(model=LLM_AS_JUDGE_MODEL_NAME),
        Guidelines(
            model=LLM_AS_JUDGE_MODEL_NAME,
            guidelines=[
                "The response must be in English.",
                "The response must be clear, coherent, and concise.",
            ],
        ),
        # default use gpt-4o-mini as judge
        Guidelines(name="easy_to_understand", guidelines=easy_to_understand),
        # code based scorers
        accuracy_scorer,
    ]

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )


if __name__ == "__main__":
    # Evaluation dataset file path
    test_case_file_path = "eval_data/test_cases_1.yaml"
    experiment_name = "RAG Q&A Evaluation"
    dataset_name = "assistant_rag_test_set"

    # Step 1: Create or get experiment
    print("Step 1: Execute or Create experiment...")
    experiment_id = get_or_create_experiment(experiment_name)

    # Step 2: Create dataset using the utility function
    print("\nStep 2: Get or Create dataset from YAML file...")
    dataset, metadata = get_or_create_evaluation_dataset(
        experiment_ids=[experiment_id],
        dataset_name=dataset_name,
        tags={"type": "regression", "priority": "critical"},
        test_case_file_path=test_case_file_path,
    )

    # Step 3: Run evaluation
    print("\nStep 3: Running evaluation...")
    run_evaluation(dataset)
    print("\nEvaluation completed successfully!")
    print("Results saved to MLflow run. View details in the MLflow UI.")
