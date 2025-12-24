import spacy
import re
from typing import List
from relevance.base_relevance import BaseRelevance


class LexicalRelevanceModule(BaseRelevance):
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def compute_relevance(self, conversation: str, judgment: str) -> float:
        context_elements = self._extract(conversation)
        judgment_elements = self._extract(judgment)

        if not judgment_elements:
            return 0.0

        common = set(context_elements).intersection(set(judgment_elements))
        return len(common) / len(judgment_elements)

    def _extract(self, text: str) -> List[str]:
        doc = self.nlp(text.lower())
        elements = set()

        for ent in doc.ents:
            elements.add(self._norm(ent.text, ent.label_))

        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 3:
                norm = re.sub(r"[^\w\s-]", "", chunk.text)
                norm = re.sub(r"\s+", "_", norm)
                elements.add(norm)

        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do"}:
                elements.add(token.lemma_)

        return list(elements)

    def _norm(self, text: str, label: str) -> str:
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
