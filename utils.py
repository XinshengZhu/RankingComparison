import h5py
import numpy as np
from typing import Dict, List, Tuple

def print_section_header(title: str) -> None:
    """
    Print a section header with a title centered in an 80-character wide line.
    Used for better output formatting and visual separation of different sections.

    Args:
        title: The text to be centered in the header
    """
    print("\n" + "-"*80)
    print(f" {title} ".center(80, "-"))
    print("-"*80 + "\n")

def load_h5_embeddings(h5_file_path: str,
                      id_key: str = 'id',
                      embedding_key: str = 'embedding') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and their corresponding IDs from an HDF5 file.

    Args:
        h5_file_path: Path to the HDF5 file containing embeddings
        id_key: Key for accessing IDs in the HDF5 file
        embedding_key: Key for accessing embeddings in the HDF5 file

    Returns:
        Tuple of (ids array, embeddings array)
    """
    with h5py.File(h5_file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(int)
        embeddings = np.array(f[embedding_key]).astype(np.float32)
    print(f"Loaded {len(ids)} embeddings from {h5_file_path}")
    return ids, embeddings

def load_query_relevance(tsv_file_path: str) -> np.ndarray:
    """
    Load query relevance judgments from a TSV file.

    Args:
        tsv_file_path: Path to TSV file containing relevance judgments

    Returns:
        Array of query relevance judgments
    """
    query_relevance = None
    if "dev" in tsv_file_path:
        # Dev format: directly load all columns
        query_relevance = np.loadtxt(tsv_file_path, delimiter='\t', dtype=int)
    elif "eval" in tsv_file_path:
        # Eval format: select columns [0, 2, 3] to skip the hardcoded 0 column

        query_relevance = np.loadtxt(tsv_file_path, delimiter='\t', dtype=int)[:, [0, 2, 3]]
    print(f"Loaded {len(query_relevance)} query relevance judgments from {tsv_file_path}")
    return query_relevance

def load_standard_result(tsv_file_path: str) -> Dict[int, List[int]]:
    """
    Load standard BM25 query result from a TSV file.

    Args:
        tsv_file_path: Path to the TSV file of standard BM25 query result

    Returns:
        Dictionary mapping query IDs to lists of passage IDs
    """
    # Load raw query-passage pairs
    standard_result = np.loadtxt(tsv_file_path, delimiter='\t', dtype=int)
    print(f"Loaded {len(standard_result)} standard BM25 query results from {tsv_file_path}")

    # Convert to dictionary format
    standard_result_dict = {}
    for query_id, passage_id in standard_result:
        if query_id not in standard_result_dict:
            standard_result_dict[query_id] = []
        standard_result_dict[query_id].append(passage_id)
    return standard_result_dict
