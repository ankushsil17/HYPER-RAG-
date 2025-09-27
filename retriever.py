from collections import defaultdict


# Reciprocal Rank Fusion (RRF)
# inputs: dict name-> list[(doc_id, score)] where score is ordered (higher=better)


def rrf_merge(bm25_list, dense_list, k=60):
    rank_scores = defaultdict(float)
    for results in [bm25_list, dense_list]:
        for rank, (doc_id, _) in enumerate(results, start=1):
            rank_scores[doc_id] += 1.0 / (k + rank)
    merged = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
    return merged
