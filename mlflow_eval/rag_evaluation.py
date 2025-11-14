import mlflow
import os
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from dotenv import load_dotenv

load_dotenv()

TEST_MODEL_NAME = os.getenv("TEST_MODEL_NAME")
LLM_AS_JUDGE_MODEL_NAME = os.getenv("LLM_AS_JUDGE_MODEL_NAME")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("RAG Evaluation Quickstart")

# local Ollama model
client = OpenAI(
    base_url=os.getenv("OLLAMA_API_BASE"),  # The local Ollama REST endpoint
    api_key=os.getenv("OLLAMA_API_KEY"),  # Required to instantiate OpenAI client, it can be a random string
)


def qa_predict_fn(question: str) -> str:
    """Simple Q&A prediction function using OpenAI"""
    response = client.chat.completions.create(
        model=TEST_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 5


if __name__ == "__main__":
    # Define a simple Q&A dataset with questions and expected answers
    eval_dataset = [
        {
            "inputs": {"question": "What is the capital of France?"},
            "expectations": {
                "expected_response": "Paris"
            },
        },
        {
            "inputs": {"question": "Who was the first person to build an airplane?"},
            "expectations": {
                "expected_response": "Wright Brothers"
            },
        },
        {
            "inputs": {"question": "Who wrote Romeo and Juliet?"},
            "expectations": {
                "expected_response": "William Shakespeare"
            },
        },
    ]

    scorers = [
        Correctness(model=LLM_AS_JUDGE_MODEL_NAME),
        Guidelines(
            model=LLM_AS_JUDGE_MODEL_NAME,
            guidelines=[
                "The answer should be factually correct.",
                "The response should be polite and concise.",
                "The tone should be professional."
            ]),
        Guidelines(name="english", guidelines=["The response must be in English"], ),
        Guidelines(name="clarify", guidelines=["The response must be clear, coherent, and concise"], ),
        is_concise
    ]

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )
