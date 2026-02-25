"""
Data Loader

Loads datasets from HuggingFace or local files.
Adapted from ACES v1.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def load_from_huggingface(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "data",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load dataset from HuggingFace.
    
    Args:
        dataset_name: HF dataset name (e.g., "My-Custom-AI/ACE-BB")
        subset: Subset/configuration (e.g., "choice_behavior")
        split: Split to load (default: "data")
        limit: Maximum number of experiments to load
        
    Returns:
        List of experiment dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets package required. "
            "Install with: pip install datasets"
        )
    
    logger.info(f"Loading HF dataset: {dataset_name}, subset={subset}")
    
    # Load dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    logger.info(f"Loaded {len(dataset)} experiments from HuggingFace")
    
    # Convert to list of dicts
    experiments = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break
        experiments.append(dict(row))
    
    return experiments


def expand_experiment_row(row: Dict[str, Any]) -> pd.DataFrame:
    """
    Expand a HuggingFace experiment row into a DataFrame.
    
    In HF datasets, each row contains:
    - Scalar fields: query, experiment_label, experiment_number
    - List fields: All product attributes (title, price, etc.)
    
    This function expands lists into rows.
    """
    # Extract scalar fields
    scalars = {
        "query": row.get("query"),
        "experiment_label": row.get("experiment_label"),
        "experiment_number": row.get("experiment_number"),
    }
    
    # Find list fields (product attributes)
    list_fields = {}
    for key, value in row.items():
        if key not in scalars and isinstance(value, list):
            list_fields[key] = value
    
    if not list_fields:
        return pd.DataFrame()
    
    # Get number of products
    num_products = len(next(iter(list_fields.values())))
    
    # Build DataFrame
    data = {}
    
    # Add scalar fields (repeat for each product)
    for key, value in scalars.items():
        if value is not None:
            data[key] = [value] * num_products
    
    # Add list fields
    for key, values in list_fields.items():
        data[key] = values
    
    df = pd.DataFrame(data)
    
    # Sort by assigned_position if available
    if "assigned_position" in df.columns:
        df = df.sort_values("assigned_position").reset_index(drop=True)
    
    return df


def load_experiments_from_hf(
    dataset_name: str,
    subset: Optional[str] = None,
    limit: Optional[int] = None
) -> List[pd.DataFrame]:
    """
    Load and expand experiments from HuggingFace.
    
    Returns list of DataFrames, one per experiment.
    """
    rows = load_from_huggingface(dataset_name, subset, limit=limit)
    
    experiments = []
    for row in rows:
        df = expand_experiment_row(row)
        if not df.empty:
            experiments.append(df)
    
    logger.info(f"Expanded {len(experiments)} experiments")
    
    return experiments


def load_from_local(csv_path: str) -> pd.DataFrame:
    """
    Load dataset from local CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with experiment data
    """
    logger.info(f"Loading local dataset: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} rows from local file")
    
    return df
