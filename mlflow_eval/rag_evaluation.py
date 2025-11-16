import os
from typing import Any
import mlflow
from mlflow import MlflowClient
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from openai import OpenAI
from dotenv import load_dotenv
from dataset_utils import create_evaluation_dataset

# Load environment variables
load_dotenv()

RESULT_MODEL_NAME = os.getenv("RESULT_MODEL_NAME")
LLM_AS_JUDGE_MODEL_NAME = os.getenv("LLM_AS_JUDGE_MODEL_NAME")

# Set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# local Ollama model
openai_client = OpenAI(
    base_url=os.getenv("OLLAMA_API_BASE"),  # The local Ollama REST endpoint
    api_key=os.getenv("OLLAMA_API_KEY"),  # Required to instantiate OpenAI client, it can be a random string
)


def qa_predict_fn(question: str, context: str = None) -> str:
    """Simple Q&A prediction function using OpenAI"""
    # Optionally use context in the system message if provided
    system_content = "You are a helpful assistant. Answer questions concisely."
    if context:
        system_content += f" Context: {context}"

    response = openai_client.chat.completions.create(
        model=RESULT_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,  # keep low for consistency
    )
    return response.choices[0].message.content


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

def create_experiment(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    elif experiment.lifecycle_stage == "deleted":
        # Restore the deleted experiment
        client = MlflowClient()
        client.restore_experiment(experiment.experiment_id)
        experiment_id = experiment.experiment_id
        print(f"Restored deleted experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def run_evaluation(dataset):
    scorers = [
        Correctness(model=LLM_AS_JUDGE_MODEL_NAME),
        Guidelines(
            model=LLM_AS_JUDGE_MODEL_NAME,
            guidelines=[
                "The response must be in English.",
                "The response must be clear, coherent, and concise.",
            ]),
        accuracy_scorer
    ]

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )


if __name__ == "__main__":
    test_case_file_path = "eval_data/test_cases_1.yaml"

    # Step 1: Create or get experiment
    print("Step 1: Creating experiment...")
    experiment_name = "RAG Evaluation"
    experiment_id = create_experiment(experiment_name)

    # Step 2: Create dataset using the utility function
    print("\nStep 2: Creating dataset from YAML file...")
    dataset, metadata = create_evaluation_dataset(
        experiment_ids=[experiment_id],
        dataset_name="assistant_rag_test_set",
        tags={"type": "regression", "priority": "critical"},
        test_case_file_path=test_case_file_path)

    print(f"Dataset version: {metadata.get('version', 'N/A')}")

    # Step 3: Run evaluation
    print("\nStep 3: Running evaluation...")
    run_evaluation(dataset)
    print("\nEvaluation completed successfully!")
    print("Results saved to MLflow run. View details in the MLflow UI.")


