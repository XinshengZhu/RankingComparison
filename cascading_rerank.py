import argparse
import numpy as np
from metrics import EvaluationMetrics
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import Dict, List
from utils import print_section_header, load_h5_embeddings, load_query_relevance, load_standard_result

def rerank_query_result(query_result: Dict[int, List[int]],
                        query_ids: np.ndarray,
                        query_embeddings: np.ndarray,
                        passage_ids: np.ndarray,
                        passage_embeddings: np.ndarray,
                        k: int) -> Dict[int, List[int]]:
    """
    Rerank retrieved passages based on embedding cosine similarity with query.

    Args:
        query_result: Dictionary mapping query IDs to lists of retrieved passage IDs
        query_ids: Array of all query IDs
        query_embeddings: Query embedding matrix (shape: num_queries x embedding_dim)
        passage_ids: Array of all passage IDs
        passage_embeddings: Passage embedding matrix (shape: num_passages x embedding_dim)
        k: Number of top passages to rerank

    Returns:
        Dictionary mapping query IDs to lists of reranked passage IDs

    Notes:
        - Reranks passages using cosine similarity between query and passage embeddings
        - Maintains original order for passages with equal similarity scores
        - All passages in query_result are reranked
    """
    # Limit k number of retrieved passages to rerank
    query_result_k = {query_id: query_result[query_id][:k] for query_id in query_result.keys()}

    # Measure time taken to rerank
    start_time = time.time()

    # Rerank top-k passages for each query
    reranked_result = {}
    for result_query_id, result_passage_ids in query_result_k.items():
        # Find index of current query in query_ids array
        query_index = np.where(query_ids == result_query_id)[0][0]
        # Reshape query embedding to 2D array for similarity calculation
        query_embedding = query_embeddings[query_index].reshape(1, -1)

        # Get indices of retrieved passages in passage_ids array
        passage_indices = np.array([np.where(passage_ids == result_passage_id)[0][0]
                                    for result_passage_id in result_passage_ids])
        # Get embeddings for retrieved passages
        passage_embeddings_subset = passage_embeddings[passage_indices]

        # Calculate cosine similarities between query and passages
        similarities = cosine_similarity(query_embedding, passage_embeddings_subset).flatten()
        # Sort passage indices by similarity in descending order
        top_indices = np.argsort(similarities)[::-1]
        # Reorder passage IDs based on similarity ranking
        reranked_passages = [result_passage_ids[i] for i in top_indices]

        # Store reranked passages for current query
        reranked_result[result_query_id] = reranked_passages

    # Report elapsed time for reranking
    elapsed_time = time.time() - start_time
    print(f"Reranked top-{k} passages for {len(reranked_result)} queries in {elapsed_time} seconds")
    print()
    return reranked_result

if __name__ == '__main__':
    # Parse command-line argument
    parser = argparse.ArgumentParser(description='Cascading Rerank')
    parser.add_argument('--top', type=int, default=1000, help='Number of top passages to rerank')
    args = parser.parse_args()

    # Validate command line arguments
    if args.top < 100 or args.top > 1000:
        print("Invalid value for --top argument. Must be between 100 and 1000")
        exit(1)

    # Load passage and query embeddings
    print_section_header("Loading Passage and Query Embeddings")
    passage_ids, passage_embeddings = load_h5_embeddings('provided_data/msmarco_passages_embeddings_subset.h5')
    query_ids, query_embeddings = load_h5_embeddings('provided_data/msmarco_queries_dev_eval_embeddings.h5')

    # Load query relevance for different evaluation sets
    print_section_header("Loading Query Relevance")
    # Dev set
    query_relevance_dev = load_query_relevance('provided_data/qrels.dev.tsv')
    # Eval one set
    query_relevance_eval_one = load_query_relevance('provided_data/qrels.eval.one.tsv')
    # Eval two set
    query_relevance_eval_two = load_query_relevance('provided_data/qrels.eval.two.tsv')

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
    # Rerank top-k passages for dev set queries
    reranked_result_k_dev = rerank_query_result(result_1000_dev, query_ids, query_embeddings, passage_ids, passage_embeddings, args.top)
    # Evaluate metrics on reranked result
    reranked_result_10_dev = {query_id: reranked_result_k_dev[query_id][:10] for query_id in reranked_result_k_dev.keys()}
    mrr_10_dev = EvaluationMetrics.mean_reciprocal_rank(reranked_result_10_dev, query_relevance_dev)
    print(f"  • MRR@10: {mrr_10_dev}")
    reranked_result_100_dev = {query_id: reranked_result_k_dev[query_id][:100] for query_id in reranked_result_k_dev.keys()}
    recall_100_dev = EvaluationMetrics.recall(reranked_result_100_dev, query_relevance_dev)
    map_100_dev = EvaluationMetrics.mean_average_precision(reranked_result_100_dev, query_relevance_dev)
    print(f"  • Recall@100: {recall_100_dev}")
    print(f"  • MAP@100: {map_100_dev}")

    # Evaluate on eval one set
    print_section_header("Eval One Set Evaluation")
    # Rerank top-k passages for eval one set queries
    reranked_result_k_eval_one = rerank_query_result(result_1000_eval_one, query_ids, query_embeddings, passage_ids, passage_embeddings, args.top)
    # Evaluate metrics on reranked result
    reranked_results_10_eval_one = {query_id: reranked_result_k_eval_one[query_id][:10] for query_id in reranked_result_k_eval_one.keys()}
    mrr_10_eval_one = EvaluationMetrics.mean_reciprocal_rank(reranked_results_10_eval_one, query_relevance_eval_one)
    ndcg_10_eval_one = EvaluationMetrics.normalized_discounted_cumulative_gain(reranked_results_10_eval_one, query_relevance_eval_one)
    print(f"  • MRR@10: {mrr_10_eval_one}")
    print(f"  • NDCG@10: {ndcg_10_eval_one}")
    reranked_results_100_eval_one = {query_id: reranked_result_k_eval_one[query_id][:100] for query_id in reranked_result_k_eval_one.keys()}
    ndcg_100_eval_one = EvaluationMetrics.normalized_discounted_cumulative_gain(reranked_results_100_eval_one, query_relevance_eval_one)
    print(f"  • NDCG@100: {ndcg_100_eval_one}")

    # Evaluate on eval two set
    print_section_header("Eval Two Set Evaluation")
    # Rerank top-k passages for eval two set queries
    reranked_result_k_eval_two = rerank_query_result(result_1000_eval_two, query_ids, query_embeddings, passage_ids, passage_embeddings, args.top)
    # Evaluate metrics on reranked result
    reranked_results_10_eval_two = {query_id: reranked_result_k_eval_two[query_id][:10] for query_id in reranked_result_k_eval_two.keys()}
    mrr_10_eval_two = EvaluationMetrics.mean_reciprocal_rank(reranked_results_10_eval_two, query_relevance_eval_two)
    ndcg_10_eval_two = EvaluationMetrics.normalized_discounted_cumulative_gain(reranked_results_10_eval_two, query_relevance_eval_two)
    print(f"  • MRR@10: {mrr_10_eval_two}")
    print(f"  • NDCG@10: {ndcg_10_eval_two}")
    reranked_result_100_eval_two = {query_id: reranked_result_k_eval_two[query_id][:100] for query_id in reranked_result_k_eval_two.keys()}
    ndcg_100_eval_two = EvaluationMetrics.normalized_discounted_cumulative_gain(reranked_result_100_eval_two, query_relevance_eval_two)
    print(f"  • NDCG@100: {ndcg_100_eval_two}")

    print()
