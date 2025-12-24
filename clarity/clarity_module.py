import spacy
import re
from textstat import flesch_reading_ease
from typing import Dict, List
import numpy as np

class ClarityCalculator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.alpha = 0.25  
        self.beta = 0.25   
        self.gamma = 0.25  
        self.delta = 0.25  
    
    def compute_clarity_score(self, judgment_text: str) -> Dict:
        R = self._calculate_readability(judgment_text)
        S = self._calculate_syntactic_simplicity(judgment_text)
        D = self._calculate_disambiguation_index(judgment_text)
        P = self._calculate_pronoun_clarity(judgment_text)
        
        clarity = (
            self.alpha * R +
            self.beta * S +
            self.gamma * D +
            self.delta * P
        )
        
        component_scores = {
            'R': R,
            'S': S,
            'D': D,
            'P': P
        }
        
        weighted_components = {
            'alpha_R': self.alpha * R,
            'beta_S': self.beta * S,
            'gamma_D': self.gamma * D,
            'delta_P': self.delta * P
        }
        
        return {
            'clarity_score': clarity,
            'component_scores': component_scores,
            'weighted_components': weighted_components,
            'weights': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'delta': self.delta
            }
        }
    
    def _calculate_readability(self, text: str) -> float:
        try:
            flesch_score = flesch_reading_ease(text)
            normalized_score = max(0, min(1, flesch_score / 100))
            return normalized_score
        except:
            return 0.5
    
    def _calculate_syntactic_simplicity(self, text: str) -> float:
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0.5
        
        depths = []
        for sent in sentences:
            depth = self._get_tree_depth(sent.root)
            depths.append(depth)
        
        average_depth = sum(depths) / len(depths)
        simplicity = max(0, min(1, 1 - (average_depth / 10)))
        return simplicity
    
    def _get_tree_depth(self, node, current_depth=0) -> int:
        if not list(node.children):
            return current_depth
        
        max_depth = current_depth
        for child in node.children:
            child_depth = self._get_tree_depth(child, current_depth + 1)
            if child_depth > max_depth:
                max_depth = child_depth
        
        return max_depth
    
    def _calculate_disambiguation_index(self, text: str) -> float:
        doc = self.nlp(text)
        important_entity_types = {"MONEY", "DATE", "GPE"}
        
        found_entities = set()
        
        for ent in doc.ents:
            if ent.label_ in important_entity_types:
                found_entities.add(ent.label_)
        
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            found_entities.add("NUMERIC")
        
        currency_pattern = r'[₹$€£]\s*\d+'
        if re.search(currency_pattern, text):
            found_entities.add("MONEY")
        
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        if re.search(date_pattern, text):
            found_entities.add("DATE")
        
        return len(found_entities) / 3
    
    def _calculate_pronoun_clarity(self, text: str) -> float:
        doc = self.nlp(text)
        
        pronoun_list = []
        noun_list = []
        
        for token in doc:
            if token.pos_ == "PRON":
                pronoun_list.append(token)
            elif token.pos_ in ["NOUN", "PROPN"]:
                noun_list.append(token)
        
        if not pronoun_list:
            return 1.0
        
        ambiguous_count = 0
        for pronoun in pronoun_list:
            if not self._check_clear_antecedent(pronoun, noun_list, doc):
                ambiguous_count += 1
        
        clarity = 1 - (ambiguous_count / len(pronoun_list))
        return max(0, clarity)
    
    def _check_clear_antecedent(self, pronoun, nouns, doc) -> bool:
        pronoun_text = pronoun.text.lower()
        
        first_person = ["i", "me", "my", "mine", "myself"]
        second_person = ["you", "your", "yours", "yourself"]
        third_person_singular = ["he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself"]
        first_person_plural = ["we", "us", "our", "ours", "ourselves"]
        third_person_plural = ["they", "them", "their", "theirs", "themselves"]
        
        if pronoun_text in first_person:
            return True
        
        if pronoun_text in second_person:
            return True
        
        if pronoun_text in third_person_singular:
            for noun in nouns:
                if noun.sent == pronoun.sent:
                    return True
        
        if pronoun_text in first_person_plural:
            return True
        
        if pronoun_text in third_person_plural:
            for noun in nouns:
                if noun.sent == pronoun.sent:
                    return True
        
        return False

def main():
    calculator = ClarityCalculator()
    from cfg import DATA_PATH
    from data_loader import load_csv
    summaries,convs,judgments = load_csv(DATA_PATH) 
    
    for idx,c in enumerate(summaries):

        print(f"DOING {idx+1} : ",end="\t")

        result = calculator.compute_clarity_score(c)
        
        print(f"Clarity Score: {result['clarity_score']:.4f}")
        print(f"Component Scores:")
        print(f"  R (Readability): {result['component_scores']['R']:.4f}")
        print(f"  S (Syntactic Simplicity): {result['component_scores']['S']:.4f}")
        print(f"  D (Disambiguation Index): {result['component_scores']['D']:.4f}")
        print(f"  P (Pronoun Clarity): {result['component_scores']['P']:.4f}")
        
      
        print(f"\nTotal: {result['clarity_score']:.4f}")
        
    

if __name__ == "__main__":
    main()