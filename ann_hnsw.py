import argparse
import faiss
import numpy as np
import os
import time
from typing import Dict, List, Tuple
from metrics import EvaluationMetrics
from utils import print_section_header, load_h5_embeddings, load_query_relevance

class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) index implementation for approximate nearest neighbor search.

    This class provides functionality to:
    1. Build an HNSW index from passage embeddings
    2. Perform efficient k-NN search using the built index
    3. Support configurable index parameters for quality vs speed tradeoff

    The HNSW algorithm creates a hierarchical graph structure that allows for
    efficient approximate nearest neighbor search in high dimensional spaces.
    """

    def __init__(self, M: int, efConstruction: int, efSearch: int):
        """
        Initialize HNSW index with configurable parameters.

        Args:
            M: Maximum number of connections per node in the graph structure.
               Higher values give better accuracy but increase memory usage.
            efConstruction: Size of dynamic candidate list during index construction.
                          Higher values give better index quality but slower build time.
            efSearch: Size of dynamic candidate list during search.
                     Higher values give better search accuracy but slower search speed.
        """
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.index = None   # FAISS HNSW index
        self.passage_ids = None # Original passage IDs for mapping results

    def build(self, passage_embeddings: np.ndarray,
              passage_ids: np.ndarray) -> None:
        """
        Build HNSW index from passage embeddings.

        Args:
            passage_embeddings: Matrix of shape (n_passages, embedding_dim) containing
                              the passage embedding vectors
            passage_ids: Array of passage IDs corresponding to the embeddings
                        for mapping search results back to original passages

        Notes:
            - Uses Faiss IndexHNSWFlat which stores raw vectors without compression
            - Measures and reports the index building time
            - Stores passage IDs for translating internal index IDs to passage IDs
        """
        # Measure index building time
        start_time = time.time()

        # Get embedding dimensionality
        d = passage_embeddings.shape[1]

        # Print index configuration
        print("Index Configuration:")
        print(f"  • Embedding dimension: {d}")
        print(f"  • Maximum connections (M): {self.M}")
        print(f"  • Construction quality (efConstruction): {self.efConstruction}")
        print(f"  • Search quality (efSearch): {self.efSearch}")
        print()

        # Initialize HNSW index with configured parameters
        self.index = faiss.IndexHNSWFlat(d, self.M)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch
        self.index.metric_type = faiss.METRIC_INNER_PRODUCT

        # Add vectors to build the hierarchical graph structure
        self.index.add(passage_embeddings)

        # Store passage IDs for result mapping
        self.passage_ids = passage_ids

        # Report build statistics
        elapsed_time = time.time() - start_time
        print(f"Index Build Statistics:")
        print(f"  • Total passages indexed: {len(passage_ids):,}")
        print(f"  • Build time: {elapsed_time}")

    def save(self, index_file_path: str) -> None:
        """
        Save the built HNSW index to disk.

        Args:
            index_file_path: Path to save the index to

        Notes:
            - Saves the index to disk for future loading
            - Requires the index to be built before saving
        """
        if self.index is None:
            print("Index not built")
            exit(1)

        # Save the index to disk
        faiss.write_index(self.index, index_file_path)
        print()
        print(f"Saved index to {index_file_path}")

    def load(self, index_file_path: str, passage_ids: np.ndarray) -> None:
        """
        Load an HNSW index from disk.

        Args:
            index_file_path: Path to load the index from

        Notes:
            - Loads the index and stores it in the instance
            - Loads the passage IDs for mapping search results
        """
        # Load the index from disk
        self.index = faiss.read_index(index_file_path)
        print(f"Loaded index from {index_file_path}")
        print()

        # Initialize the passage IDs for mapping search results
        self.passage_ids = passage_ids

        # Report load statistics
        print("Index Configuration:")
        print(f"  • Embedding dimension: {self.index.d}")
        print(f"  • Maximum connections (M): {self.M}")
        print(f"  • Construction quality (efConstruction): {self.efConstruction}")
        print(f"  • Search quality (efSearch): {self.efSearch}")

    def search(self, query_embeddings: np.ndarray,
               query_ids: np.ndarray, k: int = 10) -> Dict[int, List[int]]:
        """
        Perform k-nearest neighbor search using the built HNSW index.

        Args:
            query_embeddings: Matrix of shape (n_queries, embedding_dim) containing
                            the query embedding vectors
            query_ids: Array of query IDs corresponding to the query embeddings
            k: Number of nearest neighbors to retrieve for each query

        Returns:
            Dictionary mapping query IDs to lists of k nearest passage IDs

        Notes:
            - Measures and reports search time
            - Returns original passage IDs (not internal index IDs)
        """
        if self.index is None:
            print("Index not built or loaded")
            exit(1)

        # Measure search time
        start_time = time.time()

        # Perform k-NN search
        # D: distances (not used), I: indices of nearest neighbors
        D, I = self.index.search(query_embeddings, k)

        # Map internal index IDs to original passage IDs
        result = {query_id: [self.passage_ids[neighbor_id] for neighbor_id in neighbors]
                   for query_id, neighbors in zip(query_ids, I)}

        # Report search statistics
        elapsed_time = time.time() - start_time
        print(f"Index Search Statistics:")
        print(f"  • Total queries searched: {len(query_ids):,}")
        print(f"  • Nearest neighbors (k): {k}")
        print(f"  • Search time: {elapsed_time}")
        return result

def filter_query_embeddings(query_ids: np.ndarray,
                            query_embeddings: np.ndarray,
                            query_relevance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter query embeddings to only include those with relevance judgments.

    Args:
        query_ids: Array of all query IDs
        query_embeddings: Array of embedding vectors for all queries
        query_relevance: Array of relevance judgment judgments

    Returns:
        Tuple of (filtered query IDs array, filtered query embeddings array)
    """
    # Find indices of queries that have relevance judgments
    query_indices = np.isin(query_ids, np.unique(query_relevance[:, 0]))
    filtered_query_ids = query_ids[query_indices]
    filtered_query_embeddings = query_embeddings[query_indices]
    print(f"Filtered {len(filtered_query_ids)} query embeddings with relevance judgments")
    return filtered_query_ids, filtered_query_embeddings

if __name__ == '__main__':
    # Parse command line arguments for index parameters
    parser = argparse.ArgumentParser(description="ANN HNSW")
    parser.add_argument('--M', type=int, help="Maximum number of connections per layer")
    parser.add_argument('--efConstruction', type=int, help="Size of dynamic candidate list during construction")
    parser.add_argument('--efSearch', type=int, help="Size of dynamic candidate list during search")
    parser.add_argument('--indexPath', type=str, help="Path to save/load index")
    args = parser.parse_args()

    # Validate command line arguments
    has_index_path = args.indexPath is not None
    has_all_create_params = all([args.M is not None,
                             args.efConstruction is not None,
                             args.efSearch is not None])
    has_any_create_params = any([args.M is not None,
                                 args.efConstruction is not None,
                                 args.efSearch is not None])
    if has_index_path and not has_any_create_params:
        if not os.path.exists(args.indexPath):
            print(f"Index file not found: {args.indexPath}")
            exit(1)
    elif has_all_create_params and not has_index_path:
        if not all([args.M > 0, args.efConstruction > 0, args.efSearch > 0]):
            print("Invalid index parameters")
            exit(1)
    else:
        print("Must either provide index path or all creation parameters (M, efConstruction, efSearch)")
        exit(1)

    # Load passage and query embeddings
    print_section_header("Loading Passage and Query Embeddings")
    passage_ids, passage_embeddings = load_h5_embeddings('provided_data/msmarco_passages_embeddings_subset.h5')
    query_ids, query_embeddings = load_h5_embeddings('provided_data/msmarco_queries_dev_eval_embeddings.h5')

    # Load query relevance and filter query embeddings for different evaluation sets
    print_section_header("Loading Query Relevance and Filtering Query Embeddings")
    # Dev set
    query_relevance_dev = load_query_relevance('provided_data/qrels.dev.tsv')
    query_ids_dev, query_embeddings_dev = filter_query_embeddings(query_ids, query_embeddings, query_relevance_dev)
    # Eval one set
    query_relevance_eval_one = load_query_relevance('provided_data/qrels.eval.one.tsv')
    query_ids_eval_one, query_embeddings_eval_one = filter_query_embeddings(query_ids, query_embeddings, query_relevance_eval_one)
    # Eval two set
    query_relevance_eval_two = load_query_relevance('provided_data/qrels.eval.two.tsv')
    query_ids_eval_two, query_embeddings_eval_two = filter_query_embeddings(query_ids, query_embeddings, query_relevance_eval_two)

    if args.indexPath is not None and os.path.exists(args.indexPath):
        # Load HNSW index from disk
        print_section_header("Loading HNSW Index")
        M = int(args.indexPath.replace('.faiss', '').split('_')[-3])
        efConstruction = int(args.indexPath.replace('.faiss', '').split('_')[-2])
        efSearch = int(args.indexPath.replace('.faiss', '').split('_')[-1])
        hnsw_index = HNSWIndex(M=M, efConstruction=efConstruction, efSearch=efSearch)
        hnsw_index.load(args.indexPath, passage_ids)
    elif args.M is not None and args.efConstruction is not None and args.efSearch is not None:
        # Initialize and build HNSW index
        print_section_header("Building HNSW Index")
        hnsw_index = HNSWIndex(M=args.M, efConstruction=args.efConstruction, efSearch=args.efSearch)
        hnsw_index.build(passage_embeddings, passage_ids)
        hnsw_index.save('generated_data/hnsw_index_{}_{}_{}.faiss'.format(args.M, args.efConstruction, args.efSearch))
    else:
        print("Invalid command line arguments")
        exit(1)

    # Evaluate on dev set
    print_section_header("Dev Set Evaluation")
    # Search and compute metrics for k=10
    result_10_dev = hnsw_index.search(query_embeddings_dev, query_ids_dev, k=10)
    mrr_10_dev = EvaluationMetrics.mean_reciprocal_rank(result_10_dev, query_relevance_dev)
    print(f"  • MRR@10: {mrr_10_dev}")
    print()
    # Search and compute metrics for k=100
    result_100_dev = hnsw_index.search(query_embeddings_dev, query_ids_dev, k=100)
    recall_100_dev = EvaluationMetrics.recall(result_100_dev, query_relevance_dev)
    map_100_dev = EvaluationMetrics.mean_average_precision(result_100_dev, query_relevance_dev)
    print(f"  • Recall@100: {recall_100_dev}")
    print(f"  • MAP@100: {map_100_dev}")

    # Evaluate on eval set one
    print_section_header("Eval One Set Evaluation")
    # Search and compute metrics for k=10
    result_10_eval_one = hnsw_index.search(query_embeddings_eval_one, query_ids_eval_one, k=10)
    mrr_10_eval_one = EvaluationMetrics.mean_reciprocal_rank(result_10_eval_one, query_relevance_eval_one)
    ndcg_10_eval_one = EvaluationMetrics.normalized_discounted_cumulative_gain(result_10_eval_one, query_relevance_eval_one)
    print(f"  • MRR@10: {mrr_10_eval_one}")
    print(f"  • NDCG@10: {ndcg_10_eval_one}")
    print()
    # Search and compute metrics for k=100
    results_100_eval_one = hnsw_index.search(query_embeddings_eval_one, query_ids_eval_one, k=100)
    ndcg_100_eval_one = EvaluationMetrics.normalized_discounted_cumulative_gain(results_100_eval_one, query_relevance_eval_one)
    print(f"  • NDCG@100: {ndcg_100_eval_one}")

    # Evaluate on eval set two
    print_section_header("Eval Two Set Evaluation")
    # Search and compute metrics for k=10
    result_10_eval_two = hnsw_index.search(query_embeddings_eval_two, query_ids_eval_two, k=10)
    mrr_10_eval_two = EvaluationMetrics.mean_reciprocal_rank(result_10_eval_two, query_relevance_eval_two)
    ndcg_10_eval_two = EvaluationMetrics.normalized_discounted_cumulative_gain(result_10_eval_two, query_relevance_eval_two)
    print(f"  • MRR@10: {mrr_10_eval_two}")
    print(f"  • NDCG@10: {ndcg_10_eval_two}")
    print()
    # Search and compute metrics for k=100
    results_100_eval_two = hnsw_index.search(query_embeddings_eval_two, query_ids_eval_two, k=100)
    ndcg_100_eval_two = EvaluationMetrics.normalized_discounted_cumulative_gain(results_100_eval_two, query_relevance_eval_two)
    print(f"  • NDCG@100: {ndcg_100_eval_two}")

    print()
