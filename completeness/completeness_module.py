import re
import spacy
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch


class CompletenessEvaluator:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", use_gpu=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        
        self.discourse_types = {
            'PROBLEM': ['problem', 'issue', 'complaint', 'error', 'charged', 'dispute', 'unexpected'],
            'CAUSE': ['cause', 'reason', 'because', 'due to', 'since'],
            'SOLUTION': ['solution', 'fix', 'resolve', 'reversal', 'refund', 'adjust', 'correct'],
            'ACTION': ['action', 'step', 'measure', 'taken', 'raised', 'initiated', 'confirmed', 'verified'],
            'DECISION': ['decision', 'conclusion', 'result', 'should', 'will', 'judgment', 'determined'],
            'BACKGROUND': ['context', 'background', 'had', 'was', 'customer', 'agent'],
            'TIMELINE': ['time', 'date', 'when', 'duration', 'hours', 'days', 'within', 'between'],
            'VERIFICATION': ['verify', 'check', 'confirm', 'shows', 'system', 'acknowledged']
        }
        
        self.semantic_equivalents = {
            'reversal': {'refund', 'reversed', 'reverse', 'reversal_initiated', 'charges_reversed'},
            'error': {'incorrect', 'erroneous', 'wrong', 'mistake', 'incorrectly'},
            'charge': {'fee', 'amount', 'cost', 'bill', 'charged'},
            'disabled': {'turned_off', 'deactivated', 'inactive', 'off'},
            'ticket': {'request', 'case', 'ticket_raised', 'raised_ticket'},
            'roaming': {'international_roaming', 'roaming_charges'},
        }
        
        self.weights = {'semantic': 0.35, 'entity': 0.35, 'discourse': 0.30}
    
    def evaluate_completeness(self, conversation: str, judgment: str) -> Dict:
        sr, sr_details = self._compute_semantic_recall(conversation, judgment)
        er, er_details = self._compute_entity_recall(conversation, judgment)
        dr, dr_details = self._compute_discourse_recall(conversation, judgment)
        
        completeness = (
            self.weights['semantic'] * sr +
            self.weights['entity'] * er +
            self.weights['discourse'] * dr
        )
        
        return {
            'completeness_score': round(completeness, 3),
            'semantic_recall': round(sr, 3),
            'entity_recall': round(er, 3),
            'discourse_recall': round(dr, 3),
            'details': {
                'semantic': sr_details,
                'entity': er_details,
                'discourse': dr_details
            }
        }
    
    def _compute_semantic_recall(self, conversation: str, judgment: str) -> Tuple[float, Dict]:
        conv_frames = self._extract_semantic_frames(conversation)
        judge_frames = self._extract_semantic_frames(judgment)
        
        if not conv_frames:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        
        if not judge_frames:
            return 0.0, {'matched': 0, 'total': len(conv_frames), 'missing': conv_frames}
        
        conv_embeddings = self.embed_model.encode(conv_frames, convert_to_tensor=True)
        judge_embeddings = self.embed_model.encode(judge_frames, convert_to_tensor=True)
        
        similarities = util.cos_sim(conv_embeddings, judge_embeddings)
        max_similarities = torch.max(similarities, dim=1).values
        
        matched = torch.sum(max_similarities > 0.65).item()
        missing_indices = (max_similarities <= 0.65).nonzero(as_tuple=True)[0].tolist()
        missing_frames = [conv_frames[i] for i in missing_indices]
        
        recall = matched / len(conv_frames)
        
        details = {
            'matched': matched,
            'total': len(conv_frames),
            'missing': missing_frames
        }
        
        return recall, details
    
    def _extract_semantic_frames(self, text: str) -> List[str]:
        frames = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            subj = None
            verb = None
            obj = None
            
            for token in sent:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    subj = token.text
                elif token.pos_ == "VERB":
                    verb = token.lemma_
                elif token.dep_ in ["dobj", "attr", "pobj"]:
                    obj = token.text
            
            if subj and verb:
                frames.append(f"{subj}_{verb}")
            if verb and obj:
                frames.append(f"{verb}_{obj}")
            
            if subj and verb and obj:
                frames.append(f"{subj}_{verb}_{obj}")
        
        return frames if frames else []
    
    def _compute_entity_recall(self, conversation: str, judgment: str) -> Tuple[float, Dict]:
        conv_entities = self._extract_entities(conversation)
        judge_entities = self._extract_entities(judgment)
        
        if not conv_entities:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        
        conv_norm = {}
        for ent in conv_entities:
            key = ent['normalized']
            conv_norm[key] = ent
        
        judge_norm = set()
        for ent in judge_entities:
            judge_norm.add(ent['normalized'])
        
        matched = 0
        missing = []
        
        for key, ent_data in conv_norm.items():
            if key in judge_norm:
                matched += 1
            else:
                is_semantically_matched = False
                base_key = key.split('_')[0] if '_' in key else key
                
                if base_key in self.semantic_equivalents:
                    for equiv in self.semantic_equivalents[base_key]:
                        if any(equiv in j_key for j_key in judge_norm):
                            matched += 1
                            is_semantically_matched = True
                            break
                
                if not is_semantically_matched:
                    missing.append({
                        'text': ent_data['text'],
                        'type': ent_data['label'],
                        'normalized': key
                    })
        
        recall = matched / len(conv_norm)
        
        details = {
            'matched': matched,
            'total': len(conv_norm),
            'missing': missing
        }
        
        return recall, details
    
    def _extract_entities(self, text: str) -> List[Dict]:
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            normalized = self._normalize_entity(ent.text, ent.label_)
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'normalized': normalized
            })
        
        currency_patterns = [
            (r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'MONEY', 'INR'),
            (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'MONEY', 'USD'),
            (r'(?:INR|Rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'MONEY', 'INR'),
        ]
        
        for pattern, label, currency in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1).replace(',', '')
                entities.append({
                    'text': match.group(0),
                    'label': label,
                    'normalized': f"amount_{value}"
                })
        
        time_patterns = [
            (r'(\d+)\s*(hours?|hrs?)', 'TIME'),
            (r'(\d+)\s*(days?)', 'TIME'),
        ]
        
        for pattern, label in time_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num = match.group(1)
                unit = 'hours' if 'hour' in match.group(2).lower() or 'hr' in match.group(2).lower() else 'days'
                entities.append({
                    'text': match.group(0),
                    'label': label,
                    'normalized': f"time_{num}_{unit}"
                })
        
        action_patterns = [
            (r'(?:raised|created|opened)\s+(?:a\s+)?ticket', 'ACTION', 'ticket_raised'),
            (r'(?:roaming|service)\s+(?:was\s+)?disabled', 'STATUS', 'roaming_disabled'),
            (r'reversal|reversed|refund', 'ACTION', 'reversal'),
        ]
        
        for pattern, label, normalized in action_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(0),
                    'label': label,
                    'normalized': normalized
                })
        
        return entities
    
    def _normalize_entity(self, text: str, label: str) -> str:
        normalized = text.lower().strip()
        
        if label == "MONEY":
            numbers = re.findall(r'\d+', normalized)
            if numbers:
                return f"amount_{numbers[0]}"
        
        elif label == "DATE":
            months = {
                'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
                'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
                'may': '05', 'jun': '06', 'june': '06',
                'jul': '07', 'july': '07', 'aug': '08', 'august': '08',
                'sep': '09', 'september': '09', 'oct': '10', 'october': '10',
                'nov': '11', 'november': '11', 'dec': '12', 'december': '12'
            }
            
            range_match = re.search(r'(\w+)\s+(\d+)\s+(?:and|to|-)\s+(\w+)?\s*(\d+)', text, re.IGNORECASE)
            if range_match:
                month1 = range_match.group(1).lower()
                day1 = range_match.group(2).zfill(2)
                month2 = range_match.group(3).lower() if range_match.group(3) else month1
                day2 = range_match.group(4).zfill(2)
                
                if month1 in months and month2 in months:
                    return f"date_{months[month1]}_{day1}_to_{months[month2]}_{day2}"
            
            for month_name, month_num in months.items():
                if month_name in normalized:
                    day_match = re.search(r'(\d{1,2})', text)
                    day = day_match.group(1).zfill(2) if day_match else "01"
                    return f"date_{month_num}_{day}"
        
        elif label == "TIME":
            time_match = re.search(r'(\d+)\s*(hours?|days?|minutes?)', normalized)
            if time_match:
                num = time_match.group(1)
                unit = time_match.group(2)
                if 'hour' in unit or 'hr' in unit:
                    return f"time_{num}_hours"
                elif 'day' in unit:
                    return f"time_{num}_days"
                elif 'min' in unit:
                    return f"time_{num}_minutes"
        
        return normalized.replace(' ', '_')
    
    def _compute_discourse_recall(self, conversation: str, judgment: str) -> Tuple[float, Dict]:
        conv_units = self._extract_discourse_units(conversation)
        judge_units = self._extract_discourse_units(judgment)
        
        if not conv_units:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        
        conv_by_type = defaultdict(list)
        for unit in conv_units:
            if unit['type'] != 'OTHER':
                conv_by_type[unit['type']].append(unit)
        
        if not conv_by_type:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        
        judge_by_type = defaultdict(list)
        for unit in judge_units:
            if unit['type'] != 'OTHER':
                judge_by_type[unit['type']].append(unit)
        
        matched = 0
        missing_types = []
        
        for unit_type in conv_by_type:
            if unit_type in judge_by_type and judge_by_type[unit_type]:
                matched += 1
            else:
                missing_types.append({
                    'type': unit_type,
                    'examples': [u['content'] for u in conv_by_type[unit_type][:2]]
                })
        
        recall = matched / len(conv_by_type)
        
        details = {
            'matched': matched,
            'total': len(conv_by_type),
            'missing': missing_types
        }
        
        return recall, details
    
    def _extract_discourse_units(self, text: str) -> List[Dict]:
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


def test_completeness():
    evaluator = CompletenessEvaluator()
    
    from cfg import DATA_PATH
    from data_loader import load_csv
    
    summaries, convs, judgments = load_csv(DATA_PATH)
    
    scores = []
    for idx, (conv, judgment) in enumerate(zip(convs[:10], judgments[:10]), 1):
        result = evaluator.evaluate_completeness(conv, judgment)
        
        print(f"\n===== Example {idx} =====")
        print(f"Completeness Score: {result['completeness_score']:.3f}")
        print(f"  Semantic Recall: {result['semantic_recall']:.3f}")
        print(f"  Entity Recall: {result['entity_recall']:.3f}")
        print(f"  Discourse Recall: {result['discourse_recall']:.3f}")
        
        if result['details']['entity']['missing']:
            print(f"\nMissing Entities ({len(result['details']['entity']['missing'])}):")
            for miss in result['details']['entity']['missing'][:3]:
                print(f"  - {miss['text']} ({miss['type']})")
        
        scores.append(result['completeness_score'])
    
    print(f"\n===== SUMMARY =====")
    print(f"Average Completeness: {np.mean(scores):.3f}")
    print(f"Std Deviation: {np.std(scores):.3f}")


if __name__ == "__main__":
    test_completeness()

