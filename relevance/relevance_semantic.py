import spacy
import re
from typing import List
from sentence_transformers import SentenceTransformer, util
from relevance.base_relevance import BaseRelevance


class SemanticRelevanceModule(BaseRelevance):
    def __init__(
        self,
        spacy_model="en_core_web_sm",
        embed_model="all-MiniLM-L6-v2",
        threshold=0.5,
    ):
        self.nlp = spacy.load(spacy_model)
        self.embedder = SentenceTransformer(embed_model)
        self.threshold=threshold

    def compute_relevance(self, conversation: str, judgment: str) -> float:
        context_elements = self._extract(conversation)
        judgment_elements = self._extract(judgment)

        element_score = 0.0
        if judgment_elements:
            ctx_emb = self.embedder.encode(context_elements, convert_to_tensor=True)
            jud_emb = self.embedder.encode(judgment_elements, convert_to_tensor=True)
            sim = util.cos_sim(jud_emb, ctx_emb)
            matched = 0
            for i in range(len(judgment_elements)):
                if float(sim[i].max()) >= 0.45:
                    matched += 1
            element_score = matched / len(judgment_elements)

        conv_sents = [sent.text for sent in self.nlp(conversation).sents]
        jud_sents = [sent.text for sent in self.nlp(judgment).sents]

        sentence_score = 0.0
        if jud_sents:
            conv_emb = self.embedder.encode(conv_sents, convert_to_tensor=True)
            jud_emb = self.embedder.encode(jud_sents, convert_to_tensor=True)
            sim = util.cos_sim(jud_emb, conv_emb)
            matched = 0
            for i in range(len(jud_sents)):
                if float(sim[i].max()) >= self.threshold:
                    matched += 1
            sentence_score = matched / len(jud_sents)

        return 0.6 * element_score + 0.4 * sentence_score

    def _extract(self, text: str) -> List[str]:
        doc = self.nlp(text.lower())
        elements = set()
        for ent in doc.ents:
            elements.add(self._norm(ent.text, ent.label_))
        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 3 and not chunk.root.is_stop:
                norm = re.sub(r"[^\w\s-]", "", chunk.text)
                norm = re.sub(r"\s+", "_", norm)
                elements.add(norm)
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do", "say", "go"} and not token.is_stop:
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
