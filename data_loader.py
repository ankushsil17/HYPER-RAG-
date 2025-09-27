from datasets import load_dataset
import random

random.seed(42)


def load_squad(n=500):
    ds = load_dataset("squad", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    out = []
    for i, row in enumerate(ds):
        out.append({
            "id": f"squad_{i}",
            "question": row["question"],
            "context": row["context"],
            "answers": [a.strip() for a in row["answers"]["text"] if a.strip()]
        })
    return out


# def load_trivia_qa_rc(n=300):
#     # rc (reading comprehension) has evidence docs in context
#     ds = load_dataset("trivia_qa", "rc", split="train")
#     ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
#     out = []
#     for i, row in enumerate(ds):
#         # choose the first evidence doc_text if present
#         context = None
#         if row.get("evidence"):
#             ev = row["evidence"][0]
#             context = ev.get("doc_text") or ""
#         if not context:
#             context = row.get("search_results", [{}])[0].get("description", "")
#         out.append({
#             "id": f"trivia_{i}",
#             "question": row["question"],
#             "context": context,
#             "answers": [row["answer"]["value"]] if row.get("answer") else []
#         })
#     return out


def load_covid_qa(n=300):
    ds = load_dataset("covid_qa_deepset", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    out = []
    for i, row in enumerate(ds):
        ans = row.get("answers", []) or []
        if isinstance(ans, dict):  # handle HuggingFace style {"text": [...], "answer_start": [...]}
            ans = ans.get("text", [])
        elif isinstance(ans, str):  # fallback: single string
            ans = [ans]
        out.append({
            "id": f"covidqa_{i}",
            "question": row["question"],
            "context": row["context"] or "",
            "answers": [a for a in ans if a]
        })
    return out


def load_all(dspecs):
    corpora = {}
    queries = []
    for spec in dspecs:
        name = spec["name"]
        n = int(spec.get("subset_size", 200))
        if name == "covid_qa_deepset":
            rows = load_covid_qa(n)
        # elif name == "trivia_qa":
        #     rows = load_trivia_qa_rc(n)
        # elif name == "covid_qa_deepset":
        #     rows = load_covid_qa(n)
        else:
            raise ValueError(f"Unknown dataset {name}")

        # Build per-dataset corpus = contexts (de-duplicated)
        docs = []
        for r in rows:
            if r["context"]:
                docs.append({"doc_id": r["id"] + "_ctx", "text": r["context"]})
        corpora[name] = docs

        # Store queries with gold answers
        for r in rows:
            queries.append({
                "dataset": name,
                "qid": r["id"],
                "question": r["question"],
                "answers": r["answers"],
            })

    return corpora, queries
