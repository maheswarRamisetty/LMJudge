import spacy
import re
from typing import List, Set
from relevance.base_relevance import BaseRelevance


class LexicalRelevanceModule(BaseRelevance):    
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def compute_relevance(self, conversation: str, judgment: str) -> float:
       
        ctx_elements = set(self._extract(conversation))
        judg_elements = set(self._extract(judgment))

        if not judg_elements or not ctx_elements:
            return 0.0

        intersection = len(judg_elements & ctx_elements)
        union = len(judg_elements | ctx_elements)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        return jaccard

    def _extract(self, text: str) -> List[str]:
        
        doc = self.nlp(text.lower())
        # print(doc)
        elements = set()

        for ent in doc.ents:
            elements.add(self._normalize_entity(ent.text, ent.label_))

        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 3:
                norm = re.sub(r"[^\w\s-]", "", chunk.text)
                norm = re.sub(r"\s+", "_", norm)
                elements.add(norm)

        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do"}:
                elements.add(token.lemma_)

        return list(elements)

    def _normalize_entity(self, text: str, label: str) -> str:
        
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"\s+", "_", text)

        if label in {"MONEY", "CARDINAL"}:
            return f"amt_{text}"
        elif label == "DATE":
            return f"date_{text}"
        elif label == "GPE":
            return f"loc_{text}"
        elif label == "PERSON":
            return f"person_{text}"
        return text