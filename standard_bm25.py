import argparse
import numpy as np
from metrics import EvaluationMetrics
import subprocess
from utils import print_section_header, load_h5_embeddings, load_query_relevance, load_standard_result

def generate_collection_subset(source_tsv_file_path: str,
                               target_tsv_file_path: str,
                               passage_ids: np.ndarray) -> None:
    """
    Generate a subset of the collection based on passage IDs.

    Args:
        source_tsv_file_path: Path to the source collection TSV file
        target_tsv_file_path: Path to save the subset collection TSV file
        passage_ids: Array of passage IDs to filter the collection

    Notes:
        - Saves the filtered collection to a new TSV file
    """
    # Load full collection
    collection = np.loadtxt(source_tsv_file_path, dtype=str, delimiter='\t', comments=None, encoding='utf-8')
    print(f"Loaded {len(collection)} collection entries from {source_tsv_file_path}")

    # Filter collection based on passage IDs
    collection_ids = collection[:, 0].astype(int)
    collection_subset = collection[np.isin(collection_ids, passage_ids)]

    # Save filtered collection
    np.savetxt(target_tsv_file_path, collection_subset, fmt='%s', delimiter='\t')
    print(f"Saved {len(collection_subset)} collection entries to {target_tsv_file_path}")

def generate_query_subset(source_tsv_file_path: str,
                          target_tsv_file_path: str,
                            query_relevance: np.ndarray) -> None:
    """
    Generate a subset of the queries based on query relevance judgments.

    Args:
        source_tsv_file_path: Path to the source query TSV file
        target_tsv_file_path: Path to save the subset query TSV file
        query_relevance: Array of query relevance judgments

    Notes:
        - Saves the filtered queries to a new TSV file
    """
    # Load all queries
    queries = np.loadtxt(source_tsv_file_path, dtype=str, delimiter='\t', comments=None, encoding='utf-8')
    print(f"Loaded {len(queries)} queries from {source_tsv_file_path}")

    # Split query IDs and texts
    query_ids = queries[:, 0].astype(int)
    query_texts = queries[:, 1]

    # Filter queries based on relevance judgments
    query_indices = np.isin(query_ids, np.unique(query_relevance[:, 0]))
    query_ids = query_ids[query_indices]
    query_texts = query_texts[query_indices]
    query_subset = np.column_stack((query_ids, query_texts))

    # Save filtered queries
    np.savetxt(target_tsv_file_path, query_subset, fmt='%s', delimiter='\t')
    print(f"Saved {len(query_subset)} queries to {target_tsv_file_path}")

def generate_bm25_query_results() -> None:
    """
    Generate BM25 query results different evaluation sets.
    Execute the BM25 search pipeline using SearchSystem executables.

    Notes:
        - Saves the BM25 query result for each evaluation set to a TSV file
    """
    # Run SearchSystem components in sequence
    print("Running SearchSystem components...")

    # DataParser
    print()
    print("Parsing data...")
    subprocess.run(["SearchSystem_subset/build/DataParser"])

    # IndexBuilder
    print()
    print("Building index...")
    subprocess.run(["SearchSystem_subset/build/IndexBuilder"])

    # QueryProcessor
    print()
    print("Processing queries...")
    subprocess.run(["SearchSystem_subset/build/QueryProcessor"])

if __name__ == '__main__':
    # Parse command-line argument
    parser = argparse.ArgumentParser(description='Standard BM25')
    parser.add_argument('--generated', action='store_true', help='Use pre-generated results')
    args = parser.parse_args()

    # Validate command line arguments
    if args.generated is False:
        # Load passage embeddings
        print_section_header("Loading Passage Embeddings")
        passage_ids, passage_embeddings = load_h5_embeddings('provided_data/msmarco_passages_embeddings_subset.h5')

        # Generate a subset of the collection based on passage IDs
        print_section_header("Generating Collection Subset")
        generate_collection_subset('provided_data/collection.tsv',
                                   'generated_data/collection_subset.tsv',
                                   passage_ids)

    # Load query relevance for different evaluation sets
    print_section_header("Loading Query Relevance")
    # Dev set
    query_relevance_dev = load_query_relevance('provided_data/qrels.dev.tsv')
    # Eval one set
    query_relevance_eval_one = load_query_relevance('provided_data/qrels.eval.one.tsv')
    # Eval two set
    query_relevance_eval_two = load_query_relevance('provided_data/qrels.eval.two.tsv')

    if args.generated is False:
        # Generate a subset of the queries based on query relevance judgments for different evaluation sets
        print_section_header("Generating Query Subset")
        # Dev set
        generate_query_subset('provided_data/queries.dev.tsv',
                              'generated_data/queries.dev.subset.tsv',
                              query_relevance_dev)
        # Eval one set
        generate_query_subset('provided_data/queries.eval.tsv',
                              'generated_data/queries.eval.one.subset.tsv',
                              query_relevance_eval_one)
        # Eval two set
        generate_query_subset('provided_data/queries.eval.tsv',
                              'generated_data/queries.eval.two.subset.tsv',
                              query_relevance_eval_two)

        # Generate BM25 query results for different evaluation sets
        print_section_header("Generating BM25 Query Results")
        generate_bm25_query_results()

    # Load standard BM25 query results for different evaluation sets
    print_section_header("Loading Standard BM25 Query Results")
    # Dev set
    result_1000_dev = load_standard_result('generated_data/bm25_result_dev.tsv')
    # Eval one set
    result_1000_eval_one = load_standard_result('generated_data/bm25_result_eval_one.tsv')
    # Eval two set
    result_1000_eval_two = load_standard_result('generated_data/bm25_result_eval_two.tsv')

    # Evaluate on dev set
    print_section_header("Dev Set Evaluation")
    result_10_dev = {query_id: result_1000_dev[query_id][:10] for query_id in result_1000_dev.keys()}
    mrr_10_dev = EvaluationMetrics.mean_reciprocal_rank(result_10_dev, query_relevance_dev)
    print(f"  • MRR@10: {mrr_10_dev}")
    result_100_dev = {query_id: result_1000_dev[query_id][:100] for query_id in result_1000_dev.keys()}
    recall_100_dev = EvaluationMetrics.recall(result_100_dev, query_relevance_dev)
    map_100_dev = EvaluationMetrics.mean_average_precision(result_100_dev, query_relevance_dev)
    print(f"  • Recall@100: {recall_100_dev}")
    print(f"  • MAP@100: {map_100_dev}")

    # Evaluate on eval one set
    print_section_header("Eval One Set Evaluation")
    result_10_eval_one = {query_id: result_1000_eval_one[query_id][:10] for query_id in result_1000_eval_one.keys()}
    mrr_10_eval_one = EvaluationMetrics.mean_reciprocal_rank(result_10_eval_one, query_relevance_eval_one)
    ndcg_10_eval_one = EvaluationMetrics.normalized_discounted_cumulative_gain(result_10_eval_one, query_relevance_eval_one)
    print(f"  • MRR@10: {mrr_10_eval_one}")
    print(f"  • NDCG@10: {ndcg_10_eval_one}")
    result_100_eval_one = {query_id: result_1000_eval_one[query_id][:100] for query_id in result_1000_eval_one.keys()}
    ndcg_100_eval_one = EvaluationMetrics.normalized_discounted_cumulative_gain(result_100_eval_one, query_relevance_eval_one)
    print(f"  • NDCG@100: {ndcg_100_eval_one}")

    # Evaluate on eval two set
    print_section_header("Eval Two Set Evaluation")
    result_10_eval_two = {query_id: result_1000_eval_two[query_id][:10] for query_id in result_1000_eval_two.keys()}
    mrr_10_eval_two = EvaluationMetrics.mean_reciprocal_rank(result_10_eval_two, query_relevance_eval_two)
    ndcg_10_eval_two = EvaluationMetrics.normalized_discounted_cumulative_gain(result_10_eval_two, query_relevance_eval_two)
    print(f"  • MRR@10: {mrr_10_eval_two}")
    print(f"  • NDCG@10: {ndcg_10_eval_two}")
    result_100_eval_two = {query_id: result_1000_eval_two[query_id][:100] for query_id in result_1000_eval_two.keys()}
    ndcg_100_eval_two = EvaluationMetrics.normalized_discounted_cumulative_gain(result_100_eval_two, query_relevance_eval_two)
    print(f"  • NDCG@100: {ndcg_100_eval_two}")

    print()
