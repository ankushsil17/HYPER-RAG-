from tqdm import tqdm
import random
import numpy as np


random.seed(42)
np.random.seed(42)


def build_corpus_map(docs):
    return {d["doc_id"]: d["text"] for d in docs}


def format_context(doc_map, doc_ids):
    pieces = [doc_map[did] for did in doc_ids if did in doc_map]
    return "\n---\n".join(pieces)


class Progress:
    def __init__(self, it, desc=""):
        self.tq = tqdm(it, desc=desc)
    def __iter__(self):
        for x in self.tq:
            yield x
