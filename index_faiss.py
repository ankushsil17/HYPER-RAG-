import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseIndex:
    def __init__(self, docs, embed_model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256, device=None):
        self.docs = docs
        self.doc_ids = [d["doc_id"] for d in docs]
        self.model = SentenceTransformer(embed_model_name, device=device)
        self.batch_size = batch_size
        self.index = None
        self.emb = None
    
    
    def build(self):
        texts = [d["text"] for d in self.docs]
        self.emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=self.batch_size, normalize_embeddings=True)
        dim = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.emb.astype(np.float32))
        
    
    def search(self, query, top_k=10):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q.astype(np.float32), top_k)
        idxs = I[0].tolist()
        sims = D[0].tolist()
        return [(self.doc_ids[i], sims[j]) for j, i in enumerate(idxs)]
