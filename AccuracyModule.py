import re
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rfuzz
import torch
import pandas as pd 
import json


@dataclass
class FactualElement:
    text: str
    normalized: str
    entity_type: str = None
    value: Any = None
    metadata: Dict = None

@dataclass
class Response:
    text:str 
    ok:bool=False

class AccuracyCalculator:    
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 use_gpu: bool = False):
        self.nlp = spacy.load(model_name)
        
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device=self.device,
                aggregation_strategy="simple"
            )
        except:
            self.ner_pipeline = None
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_facts(self, text: str, context_type: str = "conversation") -> List[FactualElement]:
        facts = []
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            fact = self._normalize_entity(ent.text, ent.label_)
            facts.append(fact)
        
        facts.extend(self._extract_with_patterns(text))
        
        facts.extend(self._extract_with_rules(text, doc))
        
        if self.ner_pipeline and len(text) < 512: 
            facts.extend(self._extract_with_transformers(text))
        
        facts = self._deduplicate_facts(facts)
        
        return facts
    
    def _normalize_entity(self, text: str, entity_type: str) -> FactualElement:

        speaker_norm = self._normalize_speaker(text)
        if speaker_norm:
            return FactualElement(text=text,normalized=speaker_norm,entity_type="SPEAKER",value=speaker_norm)

        normalized = text.lower().strip()
        
        if entity_type in ["MONEY", "CARDINAL"]:
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                value = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                currency_match = re.search(r'[₹$€£]', text)
                currency = currency_match.group() if currency_match else ""
                normalized = f"{value}{currency}".lower()
            else:
                value = text
        
        elif entity_type == "DATE":
            normalized = self._normalize_date(text)
            value = normalized
        
        elif entity_type in ["TIME", "PERCENT", "QUANTITY"]:
            value = text
        
        else:
            normalized = re.sub(r'[^\w\s-]', '', normalized)
            normalized = re.sub(r'\s+', '_', normalized)
            value = text
        
        return FactualElement(
            text=text,
            normalized=normalized,
            entity_type=entity_type,
            value=value
        )
    
    def _normalize_date(self, date_text: str) -> str:
        months = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
        
        date_lower = date_text.lower()
        for month_name, month_num in months.items():
            if month_name in date_lower:
                day_match = re.search(r'(\d{1,2})(?:\s|st|nd|rd|th)', date_text)
                day = day_match.group(1) if day_match else "01"
                day = day.zfill(2)
                return f"{month_num}-{day}"
        
        return date_text.lower()
    
    def _extract_with_patterns(self, text: str) -> List[FactualElement]:
        facts = []
        
        currency_patterns = [
            (r'[₹$€£]\s*(\d+\.?\d*)', "MONEY"),
            (r'(\d+\.?\d*)\s*(dollars|rupees|euros|pounds)', "MONEY"),
            (r'inr\s*(\d+)', "MONEY"),
        ]
        
        for pattern, entity_type in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                facts.append(FactualElement(
                    text=match.group(0),
                    normalized=match.group(0).lower(),
                    entity_type=entity_type,
                    value=match.group(1)
                ))
        
        time_patterns = [
            (r'(\d+)\s*(hours?|hrs?)', "TIME"),
            (r'(\d+)\s*(days?|days?)', "TIME"),
            (r'(\d+)\s*(minutes?|mins?)', "TIME"),
            (r'within\s*(\d+)\s*(?:hours?|days?)', "TIME"),
        ]
        
        for pattern, entity_type in time_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                facts.append(FactualElement(
                    text=match.group(0),
                    normalized=match.group(0).lower(),
                    entity_type=entity_type
                ))
        
        action_patterns = [
            (r'(?:raised|created|opened)\s*(?:a\s*)?ticket', "ACTION"),
            (r'(?:initiated|started|began)\s*(?:the\s*)?reversal', "ACTION"),
            (r'(?:disabled|turned\s*off|deactivated)', "STATUS"),
            (r'(?:enabled|turned\s*on|activated)', "STATUS"),
            (r'(?:confirmed|verified|checked)', "ACTION"),
            (r'(?:acknowledged|admitted|recognized)', "ACTION"),
        ]
        
        for pattern, entity_type in action_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                normalized = re.sub(r'\s+', '_', match.group(0).lower())
                facts.append(FactualElement(
                    text=match.group(0),
                    normalized=normalized,
                    entity_type=entity_type
                ))
        
        return facts
    
    def _extract_with_rules(self, text: str, doc) -> List[FactualElement]:
        facts = []
        speaker_norm = self._normalize_speaker(text)
        if speaker_norm:
            facts.append(FactualElement(
                text=text,
                normalized=speaker_norm,
                entity_type="SPEAKER"
            ))
        
        for token in doc:
            if token.like_num:
                for child in token.children:
                    if child.pos_ in ["NOUN", "PROPN"]:
                        fact_text = f"{token.text} {child.text}"
                        normalized = f"{token.text}_{child.text}".lower()
                        facts.append(FactualElement(
                            text=fact_text,
                            normalized=normalized,
                            entity_type="QUANTITY"
                        ))
        
        for token in doc:
            if token.dep_ == "neg":
                head = token.head
                fact_text = f"{token.text} {head.text}"
                normalized = f"not_{head.text}".lower()
                facts.append(FactualElement(
                    text=fact_text,
                    normalized=normalized,
                    entity_type="STATUS"
                ))
        
        return facts
    
    def _extract_with_transformers(self, text: str) -> List[FactualElement]:
        facts = []
        try:
            entities = self.ner_pipeline(text)
            for ent in entities:
                facts.append(FactualElement(
                    text=ent['word'],
                    normalized=ent['word'].lower(),
                    entity_type=ent['entity_group'],
                    value=ent['word']
                ))
        except Exception as e:
            pass
        
        return facts

    def _normalize_speaker(self, text: str) -> Optional[str]:
        match = re.search(r'person\s*[_\-#]*\s*(\d+)', text, re.IGNORECASE)
        if match:
            return f"person{match.group(1)}"
        return None

    
    def _deduplicate_facts(self, facts: List[FactualElement]) -> List[FactualElement]:
        seen = set()
        unique_facts = []
        
        for fact in facts:
            key = (fact.normalized, fact.entity_type)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        
        return unique_facts
    
    def match_facts(self, 
                   ref_facts: List[FactualElement], 
                   cand_facts: List[FactualElement],
                   similarity_threshold: float = 0.8) -> Tuple[List[Dict], float]:
        matches = []
        total_ref = len(ref_facts)
        
        if total_ref == 0:
            return matches, 0.0
            
        ref_texts = [fact.normalized for fact in ref_facts]
        cand_texts = [fact.normalized for fact in cand_facts]

        if len(cand_texts) == 0:
            matches = []
            for ref_fact in ref_facts:
                matches.append({
                    'reference': ref_fact,
                    'candidate': None,
                    'match_score': 0,
                    'match_type': 'none'
                })
            return matches, 0.0

        ref_embeddings = self.embedding_model.encode(ref_texts, convert_to_tensor=True)
        cand_embeddings = self.embedding_model.encode(cand_texts, convert_to_tensor=True)

        if ref_embeddings.size(0) == 0 or cand_embeddings.size(0) == 0:
            return [], 0.0

        from utils.utils import cosine_sim
        cosine_scores = cosine_sim(ref_embeddings, cand_embeddings)
    
        cand_matched = set()
        
        for i, ref_fact in enumerate(ref_facts):
            best_match = None
            best_score = 0
            best_j = -1
            
            for j, cand_fact in enumerate(cand_facts):
                if cand_fact.normalized == ref_fact.normalized:
                    best_match = cand_fact
                    best_score = 1.0
                    best_j = j
                    break
        
            if not best_match:
                for j, cand_fact in enumerate(cand_facts):
                    if j in cand_matched:
                        continue
                    
                    if (ref_fact.entity_type and cand_fact.entity_type and 
                        ref_fact.entity_type != cand_fact.entity_type):
                        continue
                    
                    similarity = rfuzz.ratio(ref_fact.normalized, cand_fact.normalized) / 100
                    
                    embedding_sim = cosine_scores[i][j].item()
                    
                    combined_score = 0.7 * embedding_sim + 0.3 * similarity
                    
                    if combined_score > best_score and combined_score >= similarity_threshold:
                        best_score = combined_score
                        best_match = cand_fact
                        best_j = j
            
            if best_match:
                cand_matched.add(best_j)
                matches.append({
                    'reference': ref_fact,
                    'candidate': best_match,
                    'match_score': best_score,
                    'match_type': 'exact' if best_score == 1.0 else 'fuzzy'
                })
            else:
                matches.append({
                    'reference': ref_fact,
                    'candidate': None,
                    'match_score': 0,
                    'match_type': 'none'
                })
        
        total_matched = sum(1 for m in matches if m['candidate'] is not None)
        accuracy = total_matched / total_ref
        
        return matches, accuracy
    
    def compute_accuracy(self, 
                        conversation_text: str, 
                        judgment_text: str,
                        include_prompt: bool = True,
                        prompt_text: str = None) -> Dict:
        
        reference_text = conversation_text
        if include_prompt and prompt_text:
            reference_text = f"{prompt_text}\n{conversation_text}"
        
        ref_facts = self.extract_facts(reference_text, "reference")
        cand_facts = self.extract_facts(judgment_text, "judgment")
        
        matches, accuracy_score = self.match_facts(ref_facts, cand_facts)
        
        fact_breakdown = []
        for match in matches:
            fact_breakdown.append({
                'reference_fact': match['reference'].text,
                'reference_normalized': match['reference'].normalized,
                'matched_candidate': match['candidate'].text if match['candidate'] else None,
                'match_score': match['match_score'],
                'match_type': match['match_type']
            })
        
        return {
            'accuracy_score': accuracy_score,
            'total_reference_facts': len(ref_facts),
            'matched_facts': sum(1 for m in matches if m['candidate'] is not None),
            'fact_breakdown': fact_breakdown,
            'reference_facts': [{'text': f.text, 'normalized': f.normalized} for f in ref_facts],
            'candidate_facts': [{'text': f.text, 'normalized': f.normalized} for f in cand_facts]
        }

if __name__ == "__main__":
    calculator = AccuracyCalculator()
    
    from cfg import DATA_PATH
    from data_loader import load_csv
    
    prompt = "Based on the above conversation, what is the decision regarding the customer's billing dispute?"
    
    
    summaries,call_convs,judgments = load_csv(DATA_PATH)


    scores = []
    for idx, (conversation, call_summary, judgment) in enumerate(zip(call_convs, summaries, judgments), start=1):
        print(f"\n===== Row {idx} =====")
        
        result = calculator.compute_accuracy(
            conversation_text=conversation,
            judgment_text=judgment,
            include_prompt=False,
            prompt_text=None    
        )
        
        # print(f"Accuracy Score: {result['accuracy_score']:.3f}")
        # print(f"Matched Facts: {result['matched_facts']}/{result['total_reference_facts']}")
        
        # print("\nReference Facts:")
        # for fact in result['reference_facts']: 
        #     print(f" - {fact['text']} -> {fact['normalized']}")
        
        # print("\nCandidate Facts:")
        # for fact in result['candidate_facts']:
        #     print(f" - {fact['text']} -> {fact['normalized']}")
        
        # print("\nMatching Breakdown:")
        # for match in result['fact_breakdown']:
        #     status = "✓" if match['matched_candidate'] else "✗"
        #     print(f" {status} {match['reference_fact']} -> {match['matched_candidate'] or 'No match'} (score: {match['match_score']:.2f})")

        scores.append(result['accuracy_score'])
    print(np.mean(scores))
