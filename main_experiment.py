import os, yaml, time
import pandas as pd
from tqdm import tqdm

from data_loader import load_all
from chunker import make_chunked_corpus
from index_bm25 import BM25Index
from index_faiss import DenseIndex
from retriever import rrf_merge
from rerank import CrossEncoderReranker
from generator import LLMGenerator
from prompts import BASE_PROMPT
from evaluate import f1_em, groundedness, Timer, qagl, semantic_sim
from utils import build_corpus_map, format_context


def run():
    with open("rag-study/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1) Load datasets (corpora + queries)
    corpora, queries = load_all(cfg["datasets"])  # dict name -> [{doc_id,text}], queries list

    # 2) Prepare models
    gen = LLMGenerator(cfg["llm_model_name"], **cfg["gen"])  # 4-bit if CUDA
    reranker = None

    rows = []                 # aggregated results
    results_per_query = []    # detailed per-question results

    for dname, corpus in corpora.items():
        print(f"\n=== Dataset: {dname} | docs={len(corpus)} ===")
        for chunk_size in cfg["chunk_sizes"]:
            for overlap in cfg["chunk_overlap_words"]:
                print(f"Chunking: size={chunk_size}, overlap={overlap}")
                ch_docs = make_chunked_corpus(
                    corpus, chunk_size_words=chunk_size, overlap_words=overlap
                )
                doc_map = build_corpus_map(ch_docs)

                # Build indexes
                bm25 = BM25Index(ch_docs)
                dense = DenseIndex(ch_docs, embed_model_name=cfg["embed_model_name"])
                dense.build()

                # Subset queries for this dataset
                dq = [q for q in queries if q["dataset"] == dname]

                for retr_name in cfg["retrievers"]:
                    for use_rr in cfg["use_rerank"]:
                        if use_rr and reranker is None:
                            reranker = CrossEncoderReranker(cfg["rerank_model_name"])  # lazy init
                        for k in cfg["top_k_list"]:
                            f1_list, em_list, g_list, lat_list, sem_list = [], [], [], [], []
                            for q in tqdm(
                                dq, desc=f"{dname}|{retr_name}|rerank={use_rr}|k={k}"
                            ):
                                # Retrieve
                                with Timer() as t_retr:
                                    if retr_name == "bm25":
                                        cand_b = bm25.search(
                                            q["question"],
                                            top_k=max(cfg["rerank_depth"], k),
                                        )
                                        cand_pairs = cand_b
                                    elif retr_name == "dense":
                                        cand_d = dense.search(
                                            q["question"],
                                            top_k=max(cfg["rerank_depth"], k),
                                        )
                                        cand_pairs = cand_d
                                    else:  # hybrid
                                        cand_b = bm25.search(
                                            q["question"],
                                            top_k=max(cfg["rerank_depth"], k),
                                        )
                                        cand_d = dense.search(
                                            q["question"],
                                            top_k=max(cfg["rerank_depth"], k),
                                        )
                                        merged = rrf_merge(cand_b, cand_d)
                                        cand_pairs = merged

                                # Candidate doc ids & texts
                                cand_docs = [
                                    (doc_id, doc_map[doc_id])
                                    for (doc_id, _) in cand_pairs
                                    if doc_id in doc_map
                                ]

                                if use_rr:
                                    rr_top = max(k, cfg["rerank_depth"])
                                    cand_docs = cand_docs[:rr_top]
                                    cand_docs = reranker.rerank(
                                        q["question"], cand_docs, top_k=k
                                    )
                                    doc_ids = [d[0] for d in cand_docs]
                                else:
                                    doc_ids = [d[0] for d in cand_docs[:k]]

                                ctx = format_context(doc_map, doc_ids)

                                # Generate
                                with Timer() as t_gen:
                                    pred = gen.generate(
                                        q["question"], ctx, prompt_template=BASE_PROMPT
                                    )

                                # Score
                                f1, em = (
                                    f1_em(pred, q["answers"])
                                    if q["answers"]
                                    else (0.0, 0.0)
                                )
                                g = groundedness(pred, ctx)
                                sem = semantic_sim(pred, q["answers"])
                                latency = t_retr.dt + t_gen.dt

                                f1_list.append(f1)
                                em_list.append(em)
                                g_list.append(g)
                                lat_list.append(latency)
                                sem_list.append(sem)

                                # log detailed per-query results
                                retrieved_texts = [doc_map[doc_id] for doc_id in doc_ids]
                                results_per_query.append({
                                    "dataset": dname,
                                    "chunk_size": chunk_size,
                                    "overlap": overlap,
                                    "retriever": retr_name,
                                    "rerank": use_rr,
                                    "top_k": k,
                                    "qid": q["qid"],
                                    "question": q["question"],
                                    "gold_answers": "; ".join(q["answers"]),
                                    "prediction": pred,
                                    "retrieved_contexts": " || ".join(retrieved_texts),
                                    "f1": f1,
                                    "em": em,
                                    "semantic": sem,
                                    "groundedness": g,
                                    "latency": latency
                                })

                            # Aggregate
                            row = {
                                "dataset": dname,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "retriever": retr_name,
                                "rerank": use_rr,
                                "top_k": k,
                                "f1": sum(f1_list) / max(1, len(f1_list)),
                                "em": sum(em_list) / max(1, len(em_list)),
                                "groundedness": sum(g_list) / max(1, len(g_list)),
                                "latency": sum(lat_list) / max(1, len(lat_list)),
                                "semantic": sum(sem_list) / max(1, len(sem_list))
                            }
                            row["qagl"] = qagl(
                                row["f1"],
                                row["semantic"],
                                row["latency"],
                                lat_ref=1.0,
                                weights=cfg.get("weights")
                            )
                            rows.append(row)
                            print(
                                {k: v for k, v in row.items() if k not in ("dataset",)}
                            )

    # Save aggregated results
    df = pd.DataFrame(rows)
    df.to_csv(cfg["results_csv"], index=False)
    print("Saved:", cfg["results_csv"])

    # Save per-query results
    dfq = pd.DataFrame(results_per_query)
    dfq.to_csv("results_per_query.csv", index=False)
    print("Saved: results_per_query.csv")


if __name__ == "__main__":
    run()
