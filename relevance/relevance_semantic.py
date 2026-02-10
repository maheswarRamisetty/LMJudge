import spacy
import re
from typing import List
from sentence_transformers import SentenceTransformer, util
import torch
from relevance.base_relevance import BaseRelevance


class SemanticRelevanceModule(BaseRelevance):

    def __init__(
        self,
        spacy_model="en_core_web_sm",
        embed_model="all-MiniLM-L6-v2",
        similarity_threshold=0.5,
    ):
       
        self.nlp = spacy.load(spacy_model)
        self.embedder = SentenceTransformer(embed_model)
        self.similarity_threshold = similarity_threshold

    def compute_relevance(self, conversation: str, judgment: str) -> float:
        conv_embedding = self.embedder.encode(conversation, convert_to_tensor=True)
        judg_embedding = self.embedder.encode(judgment, convert_to_tensor=True)
        
        doc_similarity = util.cos_sim(conv_embedding, judg_embedding).item()
        
        conv_elements = self._extract(conversation)
        judg_elements = self._extract(judgment)
        
        if not conv_elements or not judg_elements:
            return doc_similarity
        
        element_similarity = self._weighted_jaccard(judg_elements, conv_elements)
        
        relevance = 0.6 * doc_similarity + 0.4 * element_similarity
        
        return relevance

    def _weighted_jaccard(self, A: List[str], B: List[str]) -> float:
        if not A or not B:
            return 0.0

        A_emb = self.embedder.encode(A, convert_to_tensor=True)
        B_emb = self.embedder.encode(B, convert_to_tensor=True)

        sim_matrix = util.cos_sim(A_emb, B_emb)
        
        max_sim, _ = torch.max(sim_matrix, dim=1)
        
        intersection = torch.sum(max_sim > self.similarity_threshold).item()
        union = len(A) + len(B) - intersection

        return intersection / union if union > 0 else 0.0

    def _extract(self, text: str) -> List[str]:
        doc = self.nlp(text.lower())
        elements = set()

        for ent in doc.ents:
            elements.add(self._normalize_entity(ent.text, ent.label_))

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

    def _normalize_entity(self, text: str, label: str) -> str:
        
        text = re.sub(r"[^\w\s-]", "", text.lower())
        # print(text)
        text = re.sub(r"\s+", "_", text)
        # print(text)
        if label in {"MONEY", "CARDINAL"}:
            return f"amt_{text}"
        elif label == "DATE":
            return f"date_{text}"
        elif label in {"GPE", "LOC"}:
            return f"loc_{text}"
        elif label == "PERSON":
            return f"person_{text}"
        else:
            return text