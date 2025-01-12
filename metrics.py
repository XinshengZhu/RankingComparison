import numpy as np
from typing import Dict, List

class EvaluationMetrics:
    """
    Class containing methods to calculate evaluation metrics for information retrieval tasks.

    The methods in this class take two arguments:
    - result: Dictionary mapping query IDs to lists of retrieved passage IDs
    - relevance: Array of (query_id, passage_id, relevance_score) tuples

    The relevance_score is a numerical value indicating the relevance of a passage to a query.

    The methods return a float value representing the evaluation metric score.

    The following evaluation metrics are implemented:
    - Mean Reciprocal Rank (MRR)
    - Recall
    - Mean Average Precision (MAP)
    - Normalized Discounted Cumulative Gain (NDCG)
    """

    @staticmethod
    def mean_reciprocal_rank(result: Dict[int, List[int]],
                             relevance: np.ndarray) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        MRR is the average of the reciprocal ranks of the first relevant item for each query.

        :param result: Dictionary mapping query IDs to lists of retrieved passage IDs
        :param relevance: Array of (query_id, passage_id, relevance_score) tuples

        :return: MRR score between 0 and 1

        Example:

        result = {
            'query1': [doc3, doc1, doc4, doc2],  # doc1 is relevant, at position 2
            'query2': [doc5, doc6, doc7, doc8],  # no relevant docs
            'query3': [doc9, doc10, doc11, doc12] # doc9 is relevant, at position 1
        }
        relevance = [
            ('query1', doc1, 1),
            ('query3', doc9, 1)
        ]
        MRR = (1/2 + 1/1) / 2 = 0.75  # Average of [1/2, 1/1], query2 ignored
        """
        reciprocal_ranks = []
        for query_id, passage_ids in result.items():
            relevant_passage_ids = [entry[1] for entry in relevance
                             if entry[0] == query_id and entry[2] > 0]
            # Find position of first relevant document
            rank = next((i + 1 for i, passage_id in enumerate(passage_ids)
                         if passage_id in relevant_passage_ids), None)
            if rank is not None:
                reciprocal_ranks.append(1 / rank)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def recall(result: Dict[int, List[int]],
               relevance: np.ndarray) -> float:
        """
        Calculate Recall.
        Recall is the fraction of relevant documents retrieved among all relevant documents.

        :param result: Dictionary mapping query IDs to lists of retrieved passage IDs
        :param relevance: Array of (query_id, passage_id, relevance_score) tuples

        :return: Recall score between 0 and 1

        Example:

        result = {
            'query1': [doc3, doc1, doc4, doc2],  # Retrieved 2 out of 3 relevant docs
            'query2': [doc5, doc6, doc7, doc8]   # Retrieved 1 out of 2 relevant docs
        }
        relevance = [
            ('query1', doc1, 1),
            ('query1', doc2, 1),
            ('query1', doc5, 1),  # Not retrieved
            ('query2', doc7, 1),
            ('query2', doc9, 1)   # Not retrieved
        ]
        Recall = (2/3 + 1/2) / 2 = 0.583  # Average of [0.667, 0.5]
        """
        recalls = []
        for query_id, passage_ids in result.items():
            relevant_passsage_ids = [entry[1] for entry in relevance
                             if entry[0] == query_id and entry[2] > 0]
            if relevant_passsage_ids:
                recall = len(set(passage_ids) & set(relevant_passsage_ids)) / len(relevant_passsage_ids)
                recalls.append(recall)
        return np.mean(recalls) if recalls else 0.0

    @staticmethod
    def mean_average_precision(result: Dict[int, List[int]],
                               relevance: np.ndarray) -> float:
        """
        Calculate Mean Average Precision (MAP).
        MAP is the mean of average precision scores for each query.

        :param result: Dictionary mapping query IDs to lists of retrieved passage IDs
        :param relevance: Array of (query_id, passage_id, relevance_score) tuples

        :return: MAP score between 0 and 1

        Example:

        results = {
            'query1': [doc1, doc2, doc3, doc4],  # doc1 and doc4 are relevant
        }
        relevance = [
            ('query1', doc1, 1),  # Relevant doc at position 1
            ('query1', doc4, 1)   # Relevant doc at position 4
        ]
        Precision at rank 1 = 1/1 = 1.0     (1st doc is relevant)
        Precision at rank 2 = 1/2 = 0.5     (1 relevant out of 2)
        Precision at rank 3 = 1/3 ≈ 0.33    (1 relevant out of 3)
        Precision at rank 4 = 2/4 = 0.5     (2 relevant out of 4)
        AP = (1.0 + 0.5) / 2 = 0.75        (Average of precision at relevant positions)
        MAP = 0.75                          (Only one query in this example)
        """
        average_precisions = []
        for query_id, passage_ids in result.items():
            relevant_passage_ids = [entry[1] for entry in relevance
                             if entry[0] == query_id and entry[2] > 0]
            if relevant_passage_ids:
                # Calculate precision at each relevant document position
                precision = [
                    len(set(passage_ids[:i + 1]) & set(relevant_passage_ids)) / (i + 1)
                    if passage_id in relevant_passage_ids else 0
                    for i, passage_id in enumerate(passage_ids)
                ]
                average_precision = sum(precision) / len(relevant_passage_ids)
                average_precisions.append(average_precision)
        return np.mean(average_precisions) if average_precisions else 0.0

    @staticmethod
    def normalized_discounted_cumulative_gain(result: Dict[int, List[int]],
                                              relevance: np.ndarray) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).
        NDCG measures ranking quality considering relevance grades and positions.

        :param result: Dictionary mapping query IDs to lists of retrieved passage IDs
        :param relevance: Array of (query_id, passage_id, relevance_score) tuples

        :return: NDCG score between 0 and 1

        Example:

        result = {
            'query1': [doc1, doc2, doc3]  # System's ranking order
        }
        relevance = [
            ('query1', doc1, 1),  # Slightly relevant (relevance score 1)
            ('query1', doc2, 3),  # Highly relevant (relevance score 3)
            ('query1', doc3, 2)   # Moderately relevant (relevance score 2)
        ]

        DCG calculation for system's ranking:
        Position 1: (2¹ - 1) / log₂(1 + 1) = 1 / 1 = 1
        Position 2: (2³ - 1) / log₂(2 + 1) = 7 / 1.58 = 4.43
        Position 3: (2² - 1) / log₂(3 + 1) = 3 / 2 = 1.5
        DCG = 1 + 4.43 + 1.5 = 6.93

        IDCG calculation (reordering by relevance: 3,2,1):
        Position 1: (2³ - 1) / log₂(1 + 1) = 7 / 1 = 7        # doc2 (score 3)
        Position 2: (2² - 1) / log₂(2 + 1) = 3 / 1.58 = 1.89  # doc3 (score 2)
        Position 3: (2¹ - 1) / log₂(3 + 1) = 1 / 2 = 0.5      # doc1 (score 1)
        IDCG = 7 + 1.89 + 0.5 = 9.39

        NDCG = DCG/IDCG = 6.93/9.39 = 0.738
        """
        normalized_discounted_cumulative_gains = []
        for query_id, passage_ids in result.items():
            relevance_dict = {entry[1]: entry[2] for entry in relevance
                              if entry[0] == query_id}
            # Calculate DCG
            discounted_cumulative_gain = sum([(2 ** relevance_dict.get(passage_id, 0) - 1) / np.log2(i + 2)
                      for i, passage_id in enumerate(passage_ids)])
            # Calculate IDCG
            ideal_discounted_cumulative_gain = sum([(2 ** relevance_dict.get(passage_id, 0) - 1) / np.log2(i + 2)
                       for i, passage_id in enumerate(sorted(passage_ids,
                                 key=lambda passage_id: relevance_dict.get(passage_id, 0),
                                 reverse=True))])
            # Calculate NDCG
            normalized_discounted_cumulative_gain = discounted_cumulative_gain / ideal_discounted_cumulative_gain if ideal_discounted_cumulative_gain > 0 else 0.0
            normalized_discounted_cumulative_gains.append(normalized_discounted_cumulative_gain)
        return np.mean(normalized_discounted_cumulative_gains) if normalized_discounted_cumulative_gains else 0.0
