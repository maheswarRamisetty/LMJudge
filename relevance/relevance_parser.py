from relevance.relevance_lexical import LexicalRelevanceModule
from relevance.relevance_semantic import SemanticRelevanceModule


class RelevanceParser:
    def __init__(self, mode="lexical", **kwargs):
        if mode == "lexical":
            self.module = LexicalRelevanceModule()
        elif mode == "semantic":
            self.module = SemanticRelevanceModule(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def compute(self, conversation: str, summary: str, judgment: str) -> float:
        return self.module.compute_relevance(conversation, summary, judgment)