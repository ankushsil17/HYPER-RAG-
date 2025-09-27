from datasets import load_dataset
import random

random.seed(42)


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
