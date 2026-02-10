from relevance.relevance_lexical import LexicalRelevanceModule
from relevance.relevance_semantic import SemanticRelevanceModule


class RelevanceParser:
    
    def __init__(self, mode="lexical", **kwargs):
   
        self.mode = mode
        
        if mode == "semantic":
            self.module = SemanticRelevanceModule(**kwargs)
        else:
            self.module = LexicalRelevanceModule(**kwargs)
    
    def compute(self, conversation: str, judgment: str) -> float:
        
        return self.module.compute_relevance(conversation, judgment)