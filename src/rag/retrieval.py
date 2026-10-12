"""
Retrieval engine for clinical guidelines and literature evidence based on model predictions and Grad-CAM salient regions.
"""

import json
import os

try:
    import numpy as np
except ImportError:
    np = None


class ClinicalGuidelineRetriever:
    """
    Indexes retinal clinical guidelines and retrieves top matching evidence passages
    given predicted disease grade, confidence score, and XAI spatial localization.
    """
    def __init__(self, guidelines_path=None, model_name="all-MiniLM-L6-v2"):
        if guidelines_path is None:
            base_dir = os.path.dirname(__file__)
            guidelines_path = os.path.join(base_dir, "knowledge_docs", "retinal_guidelines.json")

        with open(guidelines_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.model_name = model_name
        self.encoder = None
        self._init_encoder()

    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
            self.corpus_texts = [
                f"{doc['disease']} {doc['grade']}: {doc['criteria']} Management: {doc['clinical_management']}"
                for doc in self.documents
            ]
            self.corpus_embeddings = self.encoder.encode(self.corpus_texts, convert_to_numpy=True)
        except Exception:
            self.encoder = None

    def retrieve(self, query_text, top_k=2):
        """Retrieves top_k most relevant clinical documents for a query."""
        if self.encoder is not None:
            q_emb = self.encoder.encode([query_text], convert_to_numpy=True)[0]
            scores = np.dot(self.corpus_embeddings, q_emb) / (
                np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
            )
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                doc = copy_doc = dict(self.documents[idx])
                copy_doc["retrieval_score"] = float(scores[idx])
                results.append(copy_doc)
            return results

        # Keyword-based fallback if sentence-transformers not installed
        q_lower = query_text.lower()
        results = []
        for doc in self.documents:
            score = 0.0
            if doc["disease"].lower() in q_lower:
                score += 1.0
            if doc["grade"].lower() in q_lower:
                score += 2.0
            results.append((score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:top_k]]
