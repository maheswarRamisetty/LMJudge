# import re
# from typing import List, Dict, Set, Tuple, Any, Optional
# from dataclasses import dataclass
# import spacy
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from fuzzywuzzy import fuzz
# from rapidfuzz import fuzz as rfuzz
# import torch
# import pandas as pd 
# import json


# @dataclass
# class FactualElement:
#     text: str
#     normalized: str
#     entity_type: str = None
#     value: Any = None
#     metadata: Dict = None

# @dataclass
# class Response:
#     text:str 
#     ok:bool=False

# class AccuracyCalculator:    
#     def __init__(self, 
#                  model_name: str = "en_core_web_sm",
#                  use_gpu: bool = False):
#         self.nlp = spacy.load(model_name)
        
#         self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
#         try:
#             self.ner_pipeline = pipeline(
#                 "ner",
#                 model="dslim/bert-base-NER",
#                 device=self.device,
#                 aggregation_strategy="simple"
#             )
#         except:
#             self.ner_pipeline = None
        
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#     def extract_facts(self, text: str, context_type: str = "conversation") -> List[FactualElement]:
#         facts = []
        
#         doc = self.nlp(text)
        
#         for ent in doc.ents:
#             fact = self._normalize_entity(ent.text, ent.label_)
#             facts.append(fact)
        
#         facts.extend(self._extract_with_patterns(text))
        
#         facts.extend(self._extract_with_rules(text, doc))
        
#         if self.ner_pipeline and len(text) < 512: 
#             facts.extend(self._extract_with_transformers(text))
        
#         facts = self._deduplicate_facts(facts)
        
#         return facts
    
#     def _normalize_entity(self, text: str, entity_type: str) -> FactualElement:

#         speaker_norm = self._normalize_speaker(text)
#         if speaker_norm:
#             return FactualElement(text=text,normalized=speaker_norm,entity_type="SPEAKER",value=speaker_norm)

#         normalized = text.lower().strip()
        
#         if entity_type in ["MONEY", "CARDINAL"]:
#             numbers = re.findall(r'\d+\.?\d*', text)
#             if numbers:
#                 value = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
#                 currency_match = re.search(r'[₹$€£]', text)
#                 currency = currency_match.group() if currency_match else ""
#                 normalized = f"{value}{currency}".lower()
#             else:
#                 value = text
        
#         elif entity_type == "DATE":
#             normalized = self._normalize_date(text)
#             value = normalized
        
#         elif entity_type in ["TIME", "PERCENT", "QUANTITY"]:
#             value = text
        
#         else:
#             normalized = re.sub(r'[^\w\s-]', '', normalized)
#             normalized = re.sub(r'\s+', '_', normalized)
#             value = text
        
#         return FactualElement(
#             text=text,
#             normalized=normalized,
#             entity_type=entity_type,
#             value=value
#         )
    
#     def _normalize_date(self, date_text: str) -> str:
#         months = {
#             'jan': '01', 'january': '01',
#             'feb': '02', 'february': '02',
#             'mar': '03', 'march': '03',
#             'apr': '04', 'april': '04',
#             'may': '05', 
#             'jun': '06', 'june': '06',
#             'jul': '07', 'july': '07',
#             'aug': '08', 'august': '08',
#             'sep': '09', 'september': '09',
#             'oct': '10', 'october': '10',
#             'nov': '11', 'november': '11',
#             'dec': '12', 'december': '12'
#         }
        
#         date_lower = date_text.lower()
#         for month_name, month_num in months.items():
#             if month_name in date_lower:
#                 day_match = re.search(r'(\d{1,2})(?:\s|st|nd|rd|th)', date_text)
#                 day = day_match.group(1) if day_match else "01"
#                 day = day.zfill(2)
#                 return f"{month_num}-{day}"
        
#         return date_text.lower()
    
#     def _extract_with_patterns(self, text: str) -> List[FactualElement]:
#         facts = []
        
#         currency_patterns = [
#             (r'[₹$€£]\s*(\d+\.?\d*)', "MONEY"),
#             (r'(\d+\.?\d*)\s*(dollars|rupees|euros|pounds)', "MONEY"),
#             (r'inr\s*(\d+)', "MONEY"),
#         ]
        
#         for pattern, entity_type in currency_patterns:
#             for match in re.finditer(pattern, text, re.IGNORECASE):
#                 facts.append(FactualElement(
#                     text=match.group(0),
#                     normalized=match.group(0).lower(),
#                     entity_type=entity_type,
#                     value=match.group(1)
#                 ))
        
#         time_patterns = [
#             (r'(\d+)\s*(hours?|hrs?)', "TIME"),
#             (r'(\d+)\s*(days?|days?)', "TIME"),
#             (r'(\d+)\s*(minutes?|mins?)', "TIME"),
#             (r'within\s*(\d+)\s*(?:hours?|days?)', "TIME"),
#         ]
        
#         for pattern, entity_type in time_patterns:
#             for match in re.finditer(pattern, text, re.IGNORECASE):
#                 facts.append(FactualElement(
#                     text=match.group(0),
#                     normalized=match.group(0).lower(),
#                     entity_type=entity_type
#                 ))
        
#         action_patterns = [
#             (r'(?:raised|created|opened)\s*(?:a\s*)?ticket', "ACTION"),
#             (r'(?:initiated|started|began)\s*(?:the\s*)?reversal', "ACTION"),
#             (r'(?:disabled|turned\s*off|deactivated)', "STATUS"),
#             (r'(?:enabled|turned\s*on|activated)', "STATUS"),
#             (r'(?:confirmed|verified|checked)', "ACTION"),
#             (r'(?:acknowledged|admitted|recognized)', "ACTION"),
#         ]
        
#         for pattern, entity_type in action_patterns:
#             for match in re.finditer(pattern, text, re.IGNORECASE):
#                 normalized = re.sub(r'\s+', '_', match.group(0).lower())
#                 facts.append(FactualElement(
#                     text=match.group(0),
#                     normalized=normalized,
#                     entity_type=entity_type
#                 ))
        
#         return facts
    
#     def _extract_with_rules(self, text: str, doc) -> List[FactualElement]:
#         facts = []
#         speaker_norm = self._normalize_speaker(text)
#         if speaker_norm:
#             facts.append(FactualElement(
#                 text=text,
#                 normalized=speaker_norm,
#                 entity_type="SPEAKER"
#             ))
        
#         for token in doc:
#             if token.like_num:
#                 for child in token.children:
#                     if child.pos_ in ["NOUN", "PROPN"]:
#                         fact_text = f"{token.text} {child.text}"
#                         normalized = f"{token.text}_{child.text}".lower()
#                         facts.append(FactualElement(
#                             text=fact_text,
#                             normalized=normalized,
#                             entity_type="QUANTITY"
#                         ))
        
#         for token in doc:
#             if token.dep_ == "neg":
#                 head = token.head
#                 fact_text = f"{token.text} {head.text}"
#                 normalized = f"not_{head.text}".lower()
#                 facts.append(FactualElement(
#                     text=fact_text,
#                     normalized=normalized,
#                     entity_type="STATUS"
#                 ))
        
#         return facts
    
#     def _extract_with_transformers(self, text: str) -> List[FactualElement]:
#         facts = []
#         try:
#             entities = self.ner_pipeline(text)
#             for ent in entities:
#                 facts.append(FactualElement(
#                     text=ent['word'],
#                     normalized=ent['word'].lower(),
#                     entity_type=ent['entity_group'],
#                     value=ent['word']
#                 ))
#         except Exception as e:
#             pass
        
#         return facts

#     def _normalize_speaker(self, text: str) -> Optional[str]:
#         match = re.search(r'person\s*[_\-#]*\s*(\d+)', text, re.IGNORECASE)
#         if match:
#             return f"person{match.group(1)}"
#         return None

    
#     def _deduplicate_facts(self, facts: List[FactualElement]) -> List[FactualElement]:
#         seen = set()
#         unique_facts = []
        
#         for fact in facts:
#             key = (fact.normalized, fact.entity_type)
#             if key not in seen:
#                 seen.add(key)
#                 unique_facts.append(fact)
        
#         return unique_facts
    
#     def match_facts(self, 
#                    ref_facts: List[FactualElement], 
#                    cand_facts: List[FactualElement],
#                    similarity_threshold: float = 0.6) -> Tuple[List[Dict], float]:
#         matches = []
#         total_ref = len(ref_facts)
        
#         if total_ref == 0:
#             return matches, 0.0
            
#         ref_texts = [fact.normalized for fact in ref_facts]
#         cand_texts = [fact.normalized for fact in cand_facts]

#         if len(cand_texts) == 0:
#             matches = []
#             for ref_fact in ref_facts:
#                 matches.append({
#                     'reference': ref_fact,
#                     'candidate': None,
#                     'match_score': 0,
#                     'match_type': 'none'
#                 })
#             return matches, 0.0

#         ref_embeddings = self.embedding_model.encode(ref_texts, convert_to_tensor=True)
#         cand_embeddings = self.embedding_model.encode(cand_texts, convert_to_tensor=True)

#         if ref_embeddings.size(0) == 0 or cand_embeddings.size(0) == 0:
#             return [], 0.0

#         from utils.utils import cos_sim
#         cosine_scores = cos_sim(ref_embeddings, cand_embeddings)
    
#         cand_matched = set()
        
#         for i, ref_fact in enumerate(ref_facts):
#             best_match = None
#             best_score = 0
#             best_j = -1
            
#             for j, cand_fact in enumerate(cand_facts):
#                 if cand_fact.normalized == ref_fact.normalized:
#                     best_match = cand_fact
#                     best_score = 1.0
#                     best_j = j
#                     break
        
#             if not best_match:
#                 for j, cand_fact in enumerate(cand_facts):
#                     if j in cand_matched:
#                         continue
                    
#                     # if (ref_fact.entity_type and cand_fact.entity_type and 
#                     #     ref_fact.entity_type != cand_fact.entity_type):
#                     #     continue
                    
#                     similarity = rfuzz.ratio(ref_fact.normalized, cand_fact.normalized) / 100
                    
#                     embedding_sim = cosine_scores[i][j].item()
                    
#                     combined_score = 0.7 * embedding_sim + 0.3 * similarity
                    
#                     if combined_score > best_score and combined_score >= similarity_threshold:
#                         best_score = combined_score
#                         best_match = cand_fact
#                         best_j = j
            
#             if best_match:
#                 cand_matched.add(best_j)
#                 matches.append({
#                     'reference': ref_fact,
#                     'candidate': best_match,
#                     'match_score': best_score,
#                     'match_type': 'exact' if best_score == 1.0 else 'fuzzy'
#                 })
#             else:
#                 matches.append({
#                     'reference': ref_fact,
#                     'candidate': None,
#                     'match_score': 0,
#                     'match_type': 'none'
#                 })
        
#         total_matched = sum(1 for m in matches if m['candidate'] is not None)
#         accuracy = total_matched / total_ref
        
#         return matches, accuracy
#     def compute_accuracy(
#         self,
#         conversation_text: str,
#         judgment_text: str,
#         include_prompt: bool = False,
#         prompt_text: str = None
#     ) -> Dict:

#         context_text = conversation_text
#         if include_prompt and prompt_text:
#             context_text = f"{prompt_text}\n{conversation_text}"

#         context_facts = self.extract_facts(context_text, "context")
#         judge_facts = self.extract_facts(judgment_text, "judge")

#         context_facts = [
#             f for f in context_facts
#             if f.entity_type not in {"SPEAKER"}
#         ]

#         judge_facts = [
#             f for f in judge_facts
#             if f.entity_type not in {"SPEAKER"}
#         ]

#         matches, _ = self.match_facts(
#             ref_facts=context_facts,
#             cand_facts=judge_facts,
#             similarity_threshold=0.6
#         )

#         supported = sum(1 for m in matches if m["candidate"] is not None)
#         total = len(judge_facts)

#         accuracy = supported / max(1, total)

#         return {
#             "accuracy_score": round(accuracy, 3),
#             "judge_fact_count": total,
#             "supported_facts": supported,
#             "unsupported_facts": total - supported,
#             "support_breakdown": [
#                 {
#                     "judge_fact": m["candidate"].text if m["candidate"] else None,
#                     "matched_context_fact": m["reference"].text if m["candidate"] else None,
#                     "match_score": round(m["match_score"], 3),
#                     "status": "supported" if m["candidate"] else "hallucinated"
#                 }
#                 for m in matches
#                 if m["candidate"]
#             ]
#         }


import re
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rapidfuzz import fuzz as rfuzz
import torch


@dataclass
class FactualElement:
    text: str
    normalized: str
    entity_type: str = None
    value: Any = None
    context: str = None  # NEW: track source (conversation/response/judgment)
    semantic_frame: str = None  # NEW: for completeness module


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
        
    def extract_facts(self, text: str, context: str = "unknown") -> List[FactualElement]:
        """Extract factual elements with improved patterns and context tracking"""
        facts = []
        doc = self.nlp(text)
        
        # Spacy NER
        for ent in doc.ents:
            fact = self._normalize_entity(ent.text, ent.label_, context)
            facts.append(fact)
        
        # Pattern-based extraction (enhanced)
        facts.extend(self._extract_with_patterns(text, context))
        
        # Rule-based extraction
        facts.extend(self._extract_with_rules(text, doc, context))
        
        # Transformer NER (if available)
        if self.ner_pipeline and len(text) < 512: 
            facts.extend(self._extract_with_transformers(text, context))
        
        # Deduplicate
        facts = self._deduplicate_facts(facts)
        
        return facts
    
    def _normalize_entity(self, text: str, entity_type: str, context: str) -> FactualElement:
        """Enhanced normalization with semantic frame detection"""
        normalized = text.lower().strip()
        semantic_frame = None
        
        # Detect semantic frames for completeness module
        if entity_type in ["MONEY", "CARDINAL"]:
            semantic_frame = "FINANCIAL_TRANSACTION"
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                value = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                # Normalize currency symbols
                currency_match = re.search(r'[₹$€£]|INR|USD|EUR|GBP', text, re.IGNORECASE)
                currency = currency_match.group().upper() if currency_match else ""
                if currency == "₹":
                    currency = "INR"
                normalized = f"{value}_{currency}".lower() if currency else f"{value}"
            else:
                value = text
        
        elif entity_type == "DATE":
            semantic_frame = "TEMPORAL"
            normalized = self._normalize_date(text)
            value = normalized
        
        elif entity_type == "TIME":
            semantic_frame = "TEMPORAL"
            # Extract time duration
            time_match = re.search(r'(\d+)\s*(hours?|hrs?|days?|minutes?|mins?)', text, re.IGNORECASE)
            if time_match:
                num = time_match.group(1)
                unit = time_match.group(2).lower()
                # Normalize units
                if 'hour' in unit or 'hr' in unit:
                    normalized = f"{num}_hours"
                elif 'day' in unit:
                    normalized = f"{num}_days"
                elif 'min' in unit:
                    normalized = f"{num}_minutes"
            value = normalized
        
        else:
            normalized = re.sub(r'[^\w\s-]', '', normalized)
            normalized = re.sub(r'\s+', '_', normalized)
            value = text
        
        return FactualElement(
            text=text,
            normalized=normalized,
            entity_type=entity_type,
            value=value,
            context=context,
            semantic_frame=semantic_frame
        )
    
    def _normalize_date(self, date_text: str) -> str:
        """Enhanced date normalization"""
        months = {
            'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
            'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
            'may': '05', 'jun': '06', 'june': '06',
            'jul': '07', 'july': '07', 'aug': '08', 'august': '08',
            'sep': '09', 'september': '09', 'oct': '10', 'october': '10',
            'nov': '11', 'november': '11', 'dec': '12', 'december': '12'
        }
        
        date_lower = date_text.lower()
        
        # Check for date ranges (e.g., "June 10 and June 13")
        range_match = re.search(r'(\w+)\s+(\d+)\s+(?:and|to|-)\s+(\w+)?\s*(\d+)', date_text, re.IGNORECASE)
        if range_match:
            month1 = range_match.group(1).lower()
            day1 = range_match.group(2).zfill(2)
            month2 = range_match.group(3).lower() if range_match.group(3) else month1
            day2 = range_match.group(4).zfill(2)
            
            if month1 in months and month2 in months:
                return f"{months[month1]}_{day1}_to_{months[month2]}_{day2}"
        
        # Single date
        for month_name, month_num in months.items():
            if month_name in date_lower:
                day_match = re.search(r'(\d{1,2})(?:\s|st|nd|rd|th)', date_text)
                day = day_match.group(1).zfill(2) if day_match else "01"
                return f"{month_num}_{day}"
        
        return date_text.lower()
    
    def _extract_with_patterns(self, text: str, context: str) -> List[FactualElement]:
        """Enhanced pattern extraction for telecom domain"""
        facts = []
        
        # Currency patterns
        currency_patterns = [
            (r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', "MONEY", "INR"),
            (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', "MONEY", "USD"),
            (r'(?:INR|Rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)', "MONEY", "INR"),
        ]
        
        for pattern, entity_type, currency in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1).replace(',', '')
                facts.append(FactualElement(
                    text=match.group(0),
                    normalized=f"{value}_{currency.lower()}",
                    entity_type=entity_type,
                    value=float(value),
                    context=context,
                    semantic_frame="FINANCIAL_TRANSACTION"
                ))
        
        # Time/duration patterns
        time_patterns = [
            (r'(\d+)\s*(hours?|hrs?)', "TIME"),
            (r'(\d+)\s*(days?)', "TIME"),
            (r'(\d+)\s*(minutes?|mins?)', "TIME"),
            (r'within\s+(\d+)\s*(hours?|days?)', "TIME"),
        ]
        
        for pattern, entity_type in time_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num = match.group(1)
                unit = match.group(2).lower()
                normalized = f"{num}_{'hours' if 'hour' in unit or 'hr' in unit else 'days' if 'day' in unit else 'minutes'}"
                facts.append(FactualElement(
                    text=match.group(0),
                    normalized=normalized,
                    entity_type=entity_type,
                    context=context,
                    semantic_frame="TEMPORAL"
                ))
        
        # Action/decision patterns (important for judgments!)
        action_patterns = [
            (r'(?:raised|created|opened)\s+(?:a\s+)?ticket', "ACTION", "ticket_raised"),
            (r'(?:initiated|started|began)\s+(?:the\s+)?reversal', "ACTION", "reversal_initiated"),
            (r'(?:charges?|amount)\s+(?:will\s+be\s+)?(?:reversed|refunded)', "ACTION", "charges_reversed"),
            (r'(?:disabled|turned\s+off|deactivated)', "STATUS", "disabled"),
            (r'(?:enabled|turned\s+on|activated)', "STATUS", "enabled"),
            (r'(?:confirmed|verified|checked)', "ACTION", "confirmed"),
            (r'(?:acknowledged|admitted|recognized)', "ACTION", "acknowledged"),
            (r'roaming\s+(?:was\s+)?disabled', "STATUS", "roaming_disabled"),
            (r'international\s+roaming', "ENTITY", "international_roaming"),
            (r'erroneous(?:ly)?|incorrect(?:ly)?|wrong(?:ly)?', "STATUS", "error_charge"),
        ]
        
        for pattern, entity_type, normalized_base in action_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                facts.append(FactualElement(
                    text=match.group(0),
                    normalized=normalized_base,
                    entity_type=entity_type,
                    context=context,
                    semantic_frame="ACTION" if entity_type == "ACTION" else "STATE"
                ))
        
        return facts
    
    def _extract_with_rules(self, text: str, doc, context: str) -> List[FactualElement]:
        """Rule-based extraction with context"""
        facts = []
        
        # Number + Noun patterns
        for token in doc:
            if token.like_num:
                for child in token.children:
                    if child.pos_ in ["NOUN", "PROPN"]:
                        fact_text = f"{token.text} {child.text}"
                        normalized = f"{token.text}_{child.text}".lower()
                        facts.append(FactualElement(
                            text=fact_text,
                            normalized=normalized,
                            entity_type="QUANTITY",
                            context=context
                        ))
        
        # Negation patterns
        for token in doc:
            if token.dep_ == "neg":
                head = token.head
                fact_text = f"{token.text} {head.text}"
                normalized = f"not_{head.text}".lower()
                facts.append(FactualElement(
                    text=fact_text,
                    normalized=normalized,
                    entity_type="NEGATION",
                    context=context
                ))
        
        return facts
    
    def _extract_with_transformers(self, text: str, context: str) -> List[FactualElement]:
        """Transformer-based NER"""
        facts = []
        try:
            entities = self.ner_pipeline(text)
            for ent in entities:
                facts.append(FactualElement(
                    text=ent['word'],
                    normalized=ent['word'].lower().strip(),
                    entity_type=ent['entity_group'],
                    value=ent['word'],
                    context=context
                ))
        except Exception as e:
            pass
        
        return facts
    
    def _deduplicate_facts(self, facts: List[FactualElement]) -> List[FactualElement]:
        """Deduplicate with improved key generation"""
        seen = set()
        unique_facts = []
        
        for fact in facts:
            # Use normalized value and type as key, not context
            key = (fact.normalized, fact.entity_type)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        
        return unique_facts
    
    def match_facts(self, 
                   ref_facts: List[FactualElement], 
                   cand_facts: List[FactualElement],
                   similarity_threshold: float = 0.55,
                   strict_entity_types: bool = False,
                   allow_semantic_equivalence: bool = True) -> Tuple[List[Dict], float]:
        """
        Enhanced fact matching with semantic equivalence
        
        Args:
            ref_facts: Reference facts (source of truth)
            cand_facts: Candidate facts (to be validated)
            similarity_threshold: Minimum similarity score
            strict_entity_types: Require exact entity type match
            allow_semantic_equivalence: Allow semantically equivalent terms
        """
        matches = []
        total_ref = len(ref_facts)
        
        if total_ref == 0:
            return matches, 1.0  # No facts to validate = perfect
            
        ref_texts = [fact.normalized for fact in ref_facts]
        cand_texts = [fact.normalized for fact in cand_facts]

        if len(cand_texts) == 0:
            # All reference facts are unsupported
            for ref_fact in ref_facts:
                matches.append({
                    'reference': ref_fact,
                    'candidate': None,
                    'match_score': 0,
                    'match_type': 'none'
                })
            return matches, 0.0

        # Compute embeddings
        ref_embeddings = self.embedding_model.encode(ref_texts, convert_to_tensor=True)
        cand_embeddings = self.embedding_model.encode(cand_texts, convert_to_tensor=True)

        if ref_embeddings.size(0) == 0 or cand_embeddings.size(0) == 0:
            return [], 0.0

        # Cosine similarity matrix
        cosine_scores = util.cos_sim(ref_embeddings, cand_embeddings)
        
        # Semantic equivalence dictionary (for telecom domain)
        semantic_equivalents = {
            'reversal_initiated': {'charges_reversed', 'reversal', 'refund_initiated', 'initiated_reversal'},
            'charges_reversed': {'reversal_initiated', 'reversal', 'refund_initiated', 'initiated_reversal'},
            'error_charge': {'incorrect_charge', 'erroneous_charge', 'wrong_charge'},
            'incorrect_charge': {'error_charge', 'erroneous_charge', 'wrong_charge'},
            'roaming_disabled': {'roaming_turned_off', 'roaming_deactivated', 'disabled'},
            'ticket_raised': {'ticket_created', 'ticket_opened', 'raised_ticket'},
            'acknowledged': {'confirmed', 'verified', 'agent_acknowledged'},
        }
        
        cand_matched = set()
        
        for i, ref_fact in enumerate(ref_facts):
            best_match = None
            best_score = 0
            best_j = -1
            
            # 1. Exact match
            for j, cand_fact in enumerate(cand_facts):
                if cand_fact.normalized == ref_fact.normalized:
                    best_match = cand_fact
                    best_score = 1.0
                    best_j = j
                    break
            
            # 2. Semantic equivalence check
            if not best_match and allow_semantic_equivalence:
                ref_norm = ref_fact.normalized
                if ref_norm in semantic_equivalents:
                    for j, cand_fact in enumerate(cand_facts):
                        if j in cand_matched:
                            continue
                        if cand_fact.normalized in semantic_equivalents[ref_norm]:
                            best_match = cand_fact
                            best_score = 0.95  # High but not perfect
                            best_j = j
                            break
            
            # 3. Fuzzy + embedding similarity
            if not best_match:
                for j, cand_fact in enumerate(cand_facts):
                    if j in cand_matched:
                        continue
                    
                    # Optional: strict entity type matching
                    if strict_entity_types and (ref_fact.entity_type and cand_fact.entity_type and 
                        ref_fact.entity_type != cand_fact.entity_type):
                        continue
                    
                    # Fuzzy string similarity
                    string_sim = rfuzz.ratio(ref_fact.normalized, cand_fact.normalized) / 100
                    
                    # Embedding similarity
                    embedding_sim = cosine_scores[i][j].item()
                    
                    # Combined score (weighted)
                    combined_score = 0.4 * string_sim + 0.6 * embedding_sim
                    
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
                    'match_type': 'exact' if best_score == 1.0 else 'semantic' if best_score > 0.9 else 'fuzzy'
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
    
    def compute_accuracy(
        self,
        conversation_text: str,
        llm_response_text: str,
        judgment_text: str,
        prompt_text: str = None,
        mode: str = "precision"  # "precision", "recall", or "both"
    ) -> Dict:
        """
        REVISED: Compute accuracy for judge evaluation
        
        Args:
            conversation_text: Original conversation
            llm_response_text: LLM's response to evaluate
            judgment_text: Judge's evaluation
            prompt_text: Optional prompt
            mode: What to measure
                - "precision": Are judgment's facts grounded? (recommended for judges)
                - "recall": Does judgment capture all source facts?
                - "both": Compute both metrics
        
        Returns:
            Dictionary with accuracy metrics
        """
        # Extract facts from source materials (conversation + LLM response)
        conv_facts = self.extract_facts(conversation_text, "conversation")
        llm_facts = self.extract_facts(llm_response_text, "llm_response")
        
        # Combine source facts (deduplicated)
        source_facts = conv_facts + llm_facts
        source_facts = self._deduplicate_facts(source_facts)
        
        # Extract facts from judgment
        judge_facts = self.extract_facts(judgment_text, "judgment")
        
        # Remove speaker entities from all
        source_facts = [f for f in source_facts if f.entity_type not in {"SPEAKER"}]
        judge_facts = [f for f in judge_facts if f.entity_type not in {"SPEAKER"}]
        
        result = {}
        
        # PRECISION: Are judge's claims grounded in source materials?
        if mode in ["precision", "both"]:
            precision_matches, _ = self.match_facts(
                ref_facts=judge_facts,       # What judge claimed
                cand_facts=source_facts,     # What's in conversation + response
                similarity_threshold=0.55,
                allow_semantic_equivalence=True
            )
            
            grounded = sum(1 for m in precision_matches if m["candidate"] is not None)
            total_judge = len(judge_facts)
            precision = grounded / max(1, total_judge)
            
            hallucinations = [
                {
                    "text": m["reference"].text,
                    "normalized": m["reference"].normalized,
                    "type": m["reference"].entity_type,
                    "severity": "high" if m["match_score"] == 0 else "low"
                }
                for m in precision_matches
                if m["candidate"] is None
            ]
            
            result["precision"] = round(precision, 3)
            result["judge_fact_count"] = total_judge
            result["grounded_facts"] = grounded
            result["hallucinated_facts"] = len(hallucinations)
            result["hallucination_details"] = hallucinations
        
        # RECALL: Does judge capture important source facts?
        if mode in ["recall", "both"]:
            recall_matches, _ = self.match_facts(
                ref_facts=source_facts,      # What's in conversation + response
                cand_facts=judge_facts,      # What judge mentioned
                similarity_threshold=0.55,
                allow_semantic_equivalence=True
            )
            
            captured = sum(1 for m in recall_matches if m["candidate"] is not None)
            total_source = len(source_facts)
            recall = captured / max(1, total_source)
            
            result["recall"] = round(recall, 3)
            result["source_fact_count"] = total_source
            result["captured_facts"] = captured
            result["missed_facts"] = total_source - captured
        
        # F1 Score if both metrics computed
        if mode == "both":
            p = result["precision"]
            r = result["recall"]
            result["f1_score"] = round(2 * (p * r) / max(0.001, p + r), 3)
        
        # PRIMARY METRIC FOR JUDGES: Precision
        result["accuracy_score"] = result.get("precision", result.get("recall", 0))
        
        # Detailed breakdown
        if mode == "precision":
            result["fact_breakdown"] = [
                {
                    "judge_fact": m["reference"].text,
                    "matched_source": m["candidate"].text if m["candidate"] else None,
                    "match_score": round(m["match_score"], 3),
                    "match_type": m["match_type"],
                    "status": "grounded" if m["candidate"] else "hallucinated"
                }
                for m in precision_matches
            ]
        
        return result


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    calculator = AccuracyCalculator()
    
        
    from cfg import DATA_PATH
    from data_loader import load_csv
    
    prompt = "Based on the above conversation, what is the decision regarding the customer's billing dispute?"
    
    
    summaries,call_convs,judgments = load_csv(DATA_PATH)
    scores = []
    for idx,(c,s,j) in enumerate(zip(call_convs,summaries,judgments)):
    # Compute accuracy (PRECISION mode - recommended for judges)
        result = calculator.compute_accuracy(
            conversation_text=c,
            llm_response_text=s,
            judgment_text=j,
            mode="precision"  # Focus on: no hallucinations!
        )
        
        print("\n===== ACCURACY RESULTS =====")
        print(f"Accuracy (Precision): {result['accuracy_score']:.3f}")
        print(f"Grounded Facts: {result['grounded_facts']}/{result['judge_fact_count']}")
        print(f"Hallucinated Facts: {result['hallucinated_facts']}")
        
        if result['hallucinated_facts'] > 0:
            print("\nHallucinations Detected:")
            for hall in result['hallucination_details']:
                print(f"  - {hall['text']} ({hall['type']}, severity: {hall['severity']})")
        
        print("\nFact-level Breakdown:")
        for item in result['fact_breakdown']:
            status_symbol = "✓" if item['status'] == "grounded" else "✗"
            print(f"  {status_symbol} {item['judge_fact']}")
            if item['matched_source']:
                print(f"      → Matched: {item['matched_source']} (score: {item['match_score']:.2f}, type: {item['match_type']})")
        scores.append(result['accuracy_score'])

    print(np.mean(scores))
    

# if __name__ == "__main__":
#     calculator = AccuracyCalculator()
    
#     from cfg import DATA_PATH
#     from data_loader import load_csv
    
#     prompt = "Based on the above conversation, what is the decision regarding the customer's billing dispute?"
    
    
#     summaries,call_convs,judgments = load_csv(DATA_PATH)


#     scores = []
#     for idx, (conversation, call_summary, judgment) in enumerate(zip(call_convs, summaries, judgments), start=1):
#         print(f"\n===== Row {idx} =====")
        
#         result = calculator.compute_accuracy(
#             conversation_text=call_summary,
#             judgment_text=judgment,
#             include_prompt=False,
#             prompt_text=None    
#         )
        
#         # print(f"Accuracy Score: {result['accuracy_score']:.3f}")
#         # print(f"Matched Facts: {result['matched_facts']}/{result['total_reference_facts']}")
        
#         # print("\nReference Facts:")
#         # for fact in result['reference_facts']: 
#         #     print(f" - {fact['text']} -> {fact['normalized']}")
        
#         # print("\nCandidate Facts:")
#         # for fact in result['candidate_facts']:
#         #     print(f" - {fact['text']} -> {fact['normalized']}")
        
#         # print("\nMatching Breakdown:")
#         # for match in result['fact_breakdown']:
#         #     status = "✓" if match['matched_candidate'] else "✗"
#         #     print(f" {status} {match['reference_fact']} -> {match['matched_candidate'] or 'No match'} (score: {match['match_score']:.2f})")

#         scores.append(result['accuracy_score'])
#     print(np.mean(scores))