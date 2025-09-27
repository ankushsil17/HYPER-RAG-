from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    

    @torch.no_grad()
    def score(self, query, docs):
        # docs: list[(doc_id, text)]
        pairs = [(query, d[1]) for d in docs]
        toks = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs], truncation=True, padding=True, return_tensors="pt").to(self.device)
        logits = self.model(**toks).logits.squeeze(-1)
        scores = logits.detach().cpu().tolist()
        return list(zip([d[0] for d in docs], scores))
    
    
    def rerank(self, query, docs, top_k=5):
        scored = self.score(query, docs)
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
