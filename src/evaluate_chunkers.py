from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-large-en")

def precision_at_k(ground_truth, retrieved):
    gt = set([g.lower() for g in ground_truth])
    ret = set([r.lower() for r in retrieved])

    return len(gt.intersection(ret)) / len(retrieved)

def evaluate():
    ground_truth = ["Article 8", "Right to Fair Trial"]

    queries = [
        "Which European Convention article applies?",
        "What right was violated?"
    ]

    for q in queries:
        _ = model.encode([q])  # placeholder to simulate retrieval
        retrieved_chunks = ["Article 8", "Section 3"]

        score = precision_at_k(ground_truth, retrieved_chunks)
        print(f"{q}  --> Precision@k = {score}")

if __name__ == "__main__":
    evaluate()
