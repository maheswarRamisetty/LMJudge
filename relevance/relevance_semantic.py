import spacy
import re
from typing import List
from sentence_transformers import SentenceTransformer, util
import torch
from relevance.base_relevance import BaseRelevance
from utils import cos_sim


class SemanticRelevanceModule(BaseRelevance):
    def __init__(
        self,
        spacy_model="en_core_web_sm",
        embed_model="all-MiniLM-L6-v2",
        threshold=0.3,
    ):
        self.nlp = spacy.load(spacy_model)
        self.embedder = SentenceTransformer(embed_model)
        self.threshold = threshold
    def compute_relevance(
        self, conversation: str, summary: str, judgment: str
    ) -> float:

        J = self._extract(judgment)
        C = self._extract(conversation)
        S = self._extract(summary)

        if not J:
            return 0.0
        
        print(self.threshold)
        return (
            self.threshold * self._weighted_jaccard(J, C)
            + (1 - self.threshold) * self._weighted_jaccard(J, S)
        )

    def _weighted_jaccard(self, A: List[str], B: List[str]) -> float:
        if not A or not B:
            return 0.0

        A_emb = self.embedder.encode(A, convert_to_tensor=True)
        B_emb = self.embedder.encode(B, convert_to_tensor=True)

        sim = cos_sim(A_emb, B_emb)  
        max_sim, _ = torch.max(sim, dim=1)

        intersection = max_sim.sum().item()
        union = len(A) + len(B) - intersection

        return intersection / union if union > 0 else 0.0

    def _extract(self, text: str) -> List[str]:
        doc = self.nlp(text.lower())
        elements = set()

        for ent in doc.ents:
            elements.add(self._norm(ent.text, ent.label_))

        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 3 and not chunk.root.is_stop:
                norm = re.sub(r"[^\w\s-]", "", chunk.text)
                elements.add(norm.replace(" ", "_"))

        for token in doc:
            if (
                token.pos_ == "VERB"
                and not token.is_stop
                and token.lemma_ not in {"be", "have", "do"}
            ):
                elements.add(token.lemma_)

        return list(elements)

    def _norm(self, text: str, label: str) -> str:
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"\s+", "_", text)
        if label in {"MONEY", "CARDINAL"}:
            return f"amt_{text}"
        elif label == "DATE":
            return f"date_{text}"
        elif label in {"GPE", "LOC"}:
            return f"loc_{text}"
        elif label == "PERSON":
            return f"person_{text}"
        return text
