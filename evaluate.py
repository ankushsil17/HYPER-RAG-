import re
import time
from rapidfuzz import fuzz
from statistics import harmonic_mean
from sentence_transformers import SentenceTransformer, util

_ws = re.compile(r"\s+")


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _ws.sub(" ", s).strip()
    return s


# SQuAD-style EM/F1 for string answers list


_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_sim(pred, golds):
    if not pred or not golds:
        return 0.0
    pred_emb = _embed_model.encode(pred, convert_to_tensor=True)
    gold_embs = _embed_model.encode(golds, convert_to_tensor=True)
    sims = util.cos_sim(pred_emb, gold_embs)
    return float(sims.max())  # best match

def f1_em(pred, gold_list):
    pred_n = normalize(pred)
    gold_norm = [normalize(g) for g in gold_list if g]
    if not gold_norm:
        return 0.0, 0.0
    em = max(1.0 if pred_n == g else 0.0 for g in gold_norm)
    def _f1(a, b):
        a_t = a.split(); b_t = b.split()
        common = set(a_t) & set(b_t)
        if not a_t or not b_t:
            return 0.0
        if not common:
            return 0.0
        prec = len(common)/len(a_t)
        rec = len(common)/len(b_t)
        return 2*prec*rec/(prec+rec)
    f1s = [_f1(pred_n, g) for g in gold_norm]
    return max(f1s), em


# Groundedness: is the prediction text supported by retrieved context?
# Fast heuristic: substring OR fuzzy match >= threshold


def groundedness(pred, context_text, fuzzy_threshold=80):
    pred_n = normalize(pred)
    ctx_n = normalize(context_text)
    if not pred_n:
        return 0.0
    if pred_n in ctx_n:
        return 1.0
    if fuzz.partial_ratio(pred_n, ctx_n) >= fuzzy_threshold:
        return 1.0
    return 0.0


# Composite: QAGL (harmonic mean of F1, Groundedness, 1/Latency_norm)


def harmonic_mean(vals):
    vals = [max(1e-8, v) for v in vals]
    return len(vals) / sum(1.0/v for v in vals)


class Timer:
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *exc):
        self.dt = max(1e-6, time.time() - self.t0)


# def qagl(f1, grounded, latency, lat_ref=1.0, weights=None):
#     # Normalize latency: lower is better -> inv_latency in [0, 1+] with cap
#     inv_lat = min(1.0, lat_ref / max(1e-6, latency))
#     w = weights or {"f1":1.0, "groundedness":1.0, "inv_latency":1.0}
#     return harmonic_mean([f1*w["f1"], grounded*w["groundedness"], inv_lat*w["inv_latency"]])

def qagl(f1, sem, latency, lat_ref=1.0, weights=None):
    """
    QAGL (Quality-Aware Generative Latency metric)
    Combines lexical accuracy (F1), semantic similarity, and efficiency (1/latency).
    """
    # Normalize latency: lower is better -> inv_latency ∈ [0, 1]
    inv_lat = min(1.0, lat_ref / max(1e-6, latency))

    # Default weights if none provided
    w = weights or {"f1": 1.0, "semantic": 1.0, "inv_latency": 1.0}

    return harmonic_mean([
        f1 * w["f1"],
        sem * w["semantic"],
        inv_lat * w["inv_latency"]
    ])
