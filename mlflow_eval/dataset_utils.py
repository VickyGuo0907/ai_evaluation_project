"""Utility functions for dataset management in MLflow evaluations."""
from pathlib import Path
from typing import Tuple, Dict, Any
import yaml
from mlflow.genai.datasets import create_dataset, set_dataset_tags


def load_test_cases(file_path: str = "test_cases_2.yaml") -> Tuple[list, Dict[str, Any]]:
    """Load test cases and metadata from YAML file.
    
    Args:
        file_path: Path to the YAML file containing test cases
        
    Returns:
        Tuple of (test_cases, metadata) where test_cases is a list of test cases
        and metadata is a dictionary of additional information
    """
    script_dir = Path(__file__).parent
    yaml_path = script_dir / file_path

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return data['test_cases'], data.get('metadata', {})


def create_evaluation_dataset(
        experiment_ids: list,
        dataset_name: str = "evaluation_test_set",
        tags: dict = None,
        test_case_file_path: str = "test_cases_2.yaml"
):
    """Create and configure an evaluation dataset.
    
    Args:
        experiment_ids: List of experiment IDs to associate with the dataset
        dataset_name: Name for the new dataset
        tags: Additional tags to apply to the dataset
        test_case_file_path: Path to YAML file containing test cases
        
    Returns:
        Tuple of (dataset, metadata) where dataset is the created dataset
        and metadata is the loaded metadata from the test cases file
    """
    if tags is None:
        tags = {"type": "regression", "priority": "critical"}

    # Load test cases
    test_cases, metadata = load_test_cases(test_case_file_path)

    # Create dataset
    dataset = create_dataset(
        name=dataset_name,
        experiment_id=experiment_ids,
        tags=tags,
    )

    # Merge test cases into dataset
    dataset.merge_records(test_cases)

    # Set dataset tags with metadata
    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={
            "version": metadata.get("version", "1.0"),
            "last_updated": metadata.get("last_updated", ""),
            "description": metadata.get("description", ""),
            "coverage": "comprehensive",
            "includes_adversarial": "true",
            "record_count": str(len(dataset.records)),
        },
    )

    print(f"Created dataset: {dataset.dataset_id}")
    print(f"Loaded {len(test_cases)} test cases from YAML file")
    print(f"Dataset version: {metadata.get('version', 'N/A')}")

    return dataset, metadata
