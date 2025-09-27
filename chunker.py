import re


_whitespace = re.compile(r"\s+")


def _split_words(text):
    return _whitespace.split(text.strip())


def chunk_text(text, chunk_size_words=512, overlap_words=0):
    words = _split_words(text)
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size_words - overlap_words)
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size_words])
        chunks.append(chunk)
        i += step
    return chunks


# Expand a corpus into chunked docs
# corpus: list of {doc_id, text}
# returns: list of {doc_id, text} with suffixes per chunk


def make_chunked_corpus(corpus, chunk_size_words=512, overlap_words=0):
    out = []
    for d in corpus:
        chunks = chunk_text(d["text"], chunk_size_words, overlap_words)
        for j, ch in enumerate(chunks):
            out.append({"doc_id": f"{d['doc_id']}__{j}", "text": ch})
    return out
