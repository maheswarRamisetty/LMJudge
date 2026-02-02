import re
import spacy
import numpy as np
from typing import List, Dict
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

class SummaryCompletenessEvaluator:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", use_gpu=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        
        self.discourse_types = {
            'PROBLEM': ['problem', 'issue', 'complaint', 'error'],
            'CAUSE': ['cause', 'reason', 'because'],
            'SOLUTION': ['solution', 'fix', 'resolve'],
            'ACTION': ['action', 'step', 'measure', 'taken'],
            'DECISION': ['decision', 'conclusion', 'result'],
            'BACKGROUND': ['context', 'background'],
            'TIMELINE': ['time', 'date', 'when', 'duration']
        }
        
        self.weights = {'semantic': 0.4, 'entity': 0.4, 'discourse': 0.2}
    
    def evaluate_completeness(self, conversation, summary):
        sr = self._compute_semantic_recall(conversation, summary)
        er = self._compute_entity_recall(conversation, summary)
        dr = self._compute_discourse_recall(conversation, summary)
        
        completeness = (
            self.weights['semantic'] * sr +
            self.weights['entity'] * er +
            self.weights['discourse'] * dr
        )
        
        return {
            'completeness_score': completeness,
            'semantic_recall': sr,
            'entity_recall': er,
            'discourse_recall': dr
        }
    
    def _compute_semantic_recall(self, conversation, summary):
        conv_semantics = self._extract_semantic_frames(conversation)
        summary_semantics = self._extract_semantic_frames(summary)
        
        if not conv_semantics:
            return 0.0
        
        conv_embeddings = self.embed_model.encode(conv_semantics, convert_to_tensor=True)
        summary_embeddings = self.embed_model.encode(summary_semantics, convert_to_tensor=True)
        
        similarities = util.cos_sim(conv_embeddings, summary_embeddings)
        max_similarities = torch.max(similarities, dim=1).values
        
        matched = torch.sum(max_similarities > 0.7).item()
        return matched / len(conv_semantics)
    
    def _extract_semantic_frames(self, text):
        frames = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            subj = None
            verb = None
            obj = None
            
            for token in sent:
                if token.dep_ == "nsubj":
                    subj = token.text
                elif token.pos_ == "VERB":
                    verb = token.lemma_
                elif token.dep_ == "dobj":
                    obj = token.text
            
            if subj and verb:
                frames.append(f"{subj}_{verb}")
            if verb and obj:
                frames.append(f"{verb}_{obj}")
        
        return frames
    
    def _compute_entity_recall(self, conversation, summary):
        conv_entities = self._extract_entities(conversation)
        summary_entities = self._extract_entities(summary)
        
        if not conv_entities:
            return 0.0
        
        conv_norm = {}
        for ent in conv_entities:
            key = ent['normalized']
            conv_norm[key] = ent
        
        summary_norm = {}
        for ent in summary_entities:
            key = ent['normalized']
            summary_norm[key] = ent
        
        matched = 0
        for key in conv_norm:
            if key in summary_norm:
                matched += 1
        
        return matched / len(conv_norm)
    
    def _extract_entities(self, text):
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            normalized = self._normalize_entity(ent.text, ent.label_)
            entities.append({'text': ent.text, 'label': ent.label_, 'normalized': normalized})
        
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            entities.append({'text': num, 'label': 'CARDINAL', 'normalized': f"num_{num}"})
        
        currency_patterns = [(r'[₹$€£]\s*(\d+)', 'MONEY')]
        for pattern, label in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({'text': match.group(0), 'label': label, 'normalized': f"currency_{match.group(1)}"})
        
        return entities
    
    def _normalize_entity(self, text, label):
        normalized = text.lower().strip()
        
        if label == "MONEY":
            numbers = re.findall(r'\d+', normalized)
            if numbers:
                normalized = f"amount_{numbers[0]}"
        
        elif label == "DATE":
            months = {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
                     'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                     'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}
            
            for month_name, month_num in months.items():
                if month_name in normalized:
                    day_match = re.search(r'(\d{1,2})', normalized)
                    day = day_match.group(1) if day_match else "01"
                    normalized = f"date_{month_num}_{day.zfill(2)}"
                    break
        
        return normalized
    
    def _compute_discourse_recall(self, conversation, summary):
        conv_units = self._extract_discourse_units(conversation)
        summary_units = self._extract_discourse_units(summary)
        
        if not conv_units:
            return 0.0
        
        conv_by_type = defaultdict(list)
        for unit in conv_units:
            if unit['type'] != 'OTHER':
                conv_by_type[unit['type']].append(unit)
        
        summary_by_type = defaultdict(list)
        for unit in summary_units:
            if unit['type'] != 'OTHER':
                summary_by_type[unit['type']].append(unit)
        
        matched = 0
        for unit_type in conv_by_type:
            if unit_type in summary_by_type and summary_by_type[unit_type]:
                matched += 1
        
        return matched / len(conv_by_type) if conv_by_type else 0.0
    
    def _extract_discourse_units(self, text):
        units = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            unit_type = 'OTHER'
            
            for utype, keywords in self.discourse_types.items():
                if any(keyword in sent_text for keyword in keywords):
                    unit_type = utype
                    break
            
            units.append({'type': unit_type, 'content': sent.text})
        
        return units

def test_summary_completeness():
    evaluator = SummaryCompletenessEvaluator()
    from cfg import DATA_PATH
    from data_loader import load_csv
    summaries,convs,judgments = load_csv(DATA_PATH)
    for idx,(c,s) in enumerate(zip(convs[:10],summaries[:10])):
        result = evaluator.evaluate_completeness(c, s)
        print(f"Doing {idx+1} : ",end="\t")
        print(f"Completeness Score: {result['completeness_score']:.3f}")
        # print(f"Semantic Recall: {result['semantic_recall']:.3f}")
        # print(f"Entity Recall: {result['entity_recall']:.3f}")
        # print(f"Discourse Recall: {result['discourse_recall']:.3f}")
    
    return result

if __name__ == "__main__":

    test_summary_completeness()
