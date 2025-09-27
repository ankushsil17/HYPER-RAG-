from rank_bm25 import BM25Okapi
from collections import defaultdict


class BM25Index:
    def __init__(self, docs):
        # docs: list of {doc_id, text}
        self.doc_ids = [d["doc_id"] for d in docs]
        self.tokens = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(self.tokens)
    
    
    def search(self, query, top_k=10):
        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        # return list of (doc_id, score)
        pairs = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return pairs
