"""
RAG Evaluation Module

Implements evaluation metrics for RAG (Retrieval-Augmented Generation):
- Precision@k and Recall@k
- RAGAS metrics (faithfulness, context relevance)
"""

import json
import os
from typing import List, Dict, Any, Set
import numpy as np

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_relevance, answer_relevancy
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("Warning: RAGAS not installed. Install with: pip install ragas")


def load_rag_ground_truth(filepath: str = "data/rag_queries.json") -> List[Dict[str, Any]]:
    """Load ground truth RAG queries and expected answers."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
    """
    Calculate Precision@k
    
    Args:
        retrieved_docs: List of retrieved document IDs (in order)
        relevant_docs: Set of relevant/ground-truth document IDs
        k: Number of top results to consider
    
    Returns:
        Precision at k (0.0 to 1.0)
    """
    retrieved_at_k = set(retrieved_docs[:k])
    if len(retrieved_at_k) == 0:
        return 0.0
    
    hits = len(retrieved_at_k & relevant_docs)
    return hits / len(retrieved_at_k)


def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
    """
    Calculate Recall@k
    
    Args:
        retrieved_docs: List of retrieved document IDs (in order)
        relevant_docs: Set of relevant/ground-truth document IDs
        k: Number of top results to consider
    
    Returns:
        Recall at k (0.0 to 1.0)
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved_docs[:k])
    hits = len(retrieved_at_k & relevant_docs)
    return hits / len(relevant_docs)


def evaluate_rag_retrieval(
    ground_truth: List[Dict[str, Any]],
    retrieved_results: Dict[str, List[str]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Evaluate RAG retrieval performance using precision@k and recall@k
    
    Args:
        ground_truth: Ground truth data from load_rag_ground_truth()
        retrieved_results: Dict mapping query_id to list of retrieved doc IDs
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "total_queries": len(ground_truth),
        "precision_at_k": {},
        "recall_at_k": {},
        "mean_precision": {},
        "mean_recall": {}
    }
    
    for k in k_values:
        precisions = []
        recalls = []
        
        for query_data in ground_truth:
            query_id = query_data["id"]
            
            if query_id not in retrieved_results:
                continue
            
            # Ground truth: query itself is the relevant doc (simplified)
            # In production, this would map to actual document IDs
            relevant_docs = {query_id}
            retrieved_docs = retrieved_results[query_id]
            
            p_at_k = precision_at_k(retrieved_docs, relevant_docs, k)
            r_at_k = recall_at_k(retrieved_docs, relevant_docs, k)
            
            precisions.append(p_at_k)
            recalls.append(r_at_k)
        
        if precisions:
            results["precision_at_k"][f"@{k}"] = np.mean(precisions)
            results["recall_at_k"][f"@{k}"] = np.mean(recalls)
            results["mean_precision"][f"@{k}"] = np.mean(precisions)
            results["mean_recall"][f"@{k}"] = np.mean(recalls)
    
    return results


def evaluate_with_ragas(
    queries: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Evaluate using RAGAS metrics (faithfulness, context_relevance, answer_relevancy)
    
    Args:
        queries: List of query strings
        contexts: List of lists of context strings
        answers: List of generated answer strings
        ground_truths: List of ground truth answer strings
    
    Returns:
        Dictionary with RAGAS metric scores
    """
    if not HAS_RAGAS:
        return {"error": "RAGAS not installed"}
    
    try:
        # Prepare dataset for RAGAS
        eval_dataset = {
            "question": queries,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths
        }
        
        # Calculate metrics (this is a simplified example)
        # In production, you'd use RAGAS's evaluate function properly
        metrics = {
            "faithfulness": 0.0,
            "context_relevance": 0.0,
            "answer_relevancy": 0.0
        }
        
        # Mock scoring - in production, integrate actual RAGAS evaluation
        # metrics = evaluate(eval_dataset, metrics=[faithfulness, context_relevance, answer_relevancy])
        
        return metrics
    except Exception as e:
        return {"error": str(e)}


def generate_rag_report(
    retrieval_metrics: Dict[str, Any],
    ragas_metrics: Dict[str, float]
) -> str:
    """Generate a formatted RAG evaluation report."""
    report = "# RAG Evaluation Report\n\n"
    
    report += "## Retrieval Metrics\n"
    report += f"Total Queries: {retrieval_metrics['total_queries']}\n\n"
    
    report += "### Precision@k\n"
    for k, score in retrieval_metrics["precision_at_k"].items():
        report += f"- Precision{k}: {score:.4f}\n"
    
    report += "\n### Recall@k\n"
    for k, score in retrieval_metrics["recall_at_k"].items():
        report += f"- Recall{k}: {score:.4f}\n"
    
    report += "\n## RAGAS Metrics\n"
    for metric, score in ragas_metrics.items():
        if metric != "error":
            report += f"- {metric}: {score:.4f}\n"
    
    return report


if __name__ == "__main__":
    # Example usage
    ground_truth = load_rag_ground_truth()
    
    # Mock retrieved results (in production, these come from RAG system)
    retrieved_results = {
        gt["id"]: [gt["id"], "doc_2", "doc_3", "doc_4", "doc_5"]
        for gt in ground_truth
    }
    
    retrieval_metrics = evaluate_rag_retrieval(ground_truth, retrieved_results)
    print("Retrieval Metrics:", json.dumps(retrieval_metrics, indent=2))
    
    report = generate_rag_report(retrieval_metrics, {})
    print("\n" + report)
