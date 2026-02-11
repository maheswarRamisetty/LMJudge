import re
import spacy
import math
import dateparser
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch
from rapidfuzz import fuzz

class CompletenessEvaluator:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", use_gpu=False, embed_threshold=0.65):
        self.nlp = spacy.load("en_core_web_sm")
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        self.embed_cache = {}
        self.embed_threshold = embed_threshold
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
            'roaming': {'international_roaming', 'roaming_charges', 'roaming_disabled'},
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
        c_embs = [self._embed_text(f) for f in conv_frames]
        j_embs = [self._embed_text(f) for f in judge_frames]
        sims = torch.zeros((len(c_embs), len(j_embs)))
        for i, ce in enumerate(c_embs):
            for j, je in enumerate(j_embs):
                sims[i, j] = util.cos_sim(ce, je)
        conv_best_vals, conv_best_idx = torch.max(sims, dim=1)
        matched = 0
        missing = []
        matched_pairs = []
        for i, val in enumerate(conv_best_vals):
            if val >= self.embed_threshold:
                matched += 1
                matched_pairs.append({'conv_frame': conv_frames[i], 'judge_frame': judge_frames[int(conv_best_idx[i])], 'sim': float(val)})
            else:
                fallback = False
                for jf in judge_frames:
                    if fuzz.partial_ratio(conv_frames[i], jf) > 75:
                        matched += 1
                        matched_pairs.append({'conv_frame': conv_frames[i], 'judge_frame': jf, 'sim': None, 'fuzzy': True})
                        fallback = True
                        break
                if not fallback:
                    missing.append(conv_frames[i])
        recall = matched / len(conv_frames)
        details = {'matched': matched, 'total': len(conv_frames), 'missing': missing, 'matched_pairs': matched_pairs}
        return recall, details

    def _extract_semantic_frames(self, text: str) -> List[str]:
        frames = []
        doc = self.nlp(text)
        for sent in doc.sents:
            subj = None
            verb = None
            obj = None
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subj = token.lemma_.lower()
                if token.pos_ == "VERB" and verb is None:
                    verb = token.lemma_.lower()
                if token.dep_ in ("dobj", "pobj", "attr", "dative") and obj is None:
                    obj = token.lemma_.lower()
            if subj and verb:
                frames.append(f"{subj}_{verb}")
            if verb and obj:
                frames.append(f"{verb}_{obj}")
            if subj and verb and obj:
                frames.append(f"{subj}_{verb}_{obj}")
        noun_chunks = [nc.text.lower().strip().replace(" ", "_") for nc in doc.noun_chunks if len(nc.text.strip())>2]
        frames.extend(noun_chunks[:3])
        seen = set()
        out = []
        for f in frames:
            if f and f not in seen:
                out.append(f)
                seen.add(f)
        return out

    def _compute_entity_recall(self, conversation: str, judgment: str) -> Tuple[float, Dict]:
        conv_entities = self._extract_entities(conversation)
        judge_entities = self._extract_entities(judgment)
        if not conv_entities:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        conv_map = {e['normalized']: e for e in conv_entities}
        judge_set = {e['normalized'] for e in judge_entities}
        matched = []
        missing = []
        for key, ent in conv_map.items():
            if key in judge_set:
                matched.append((ent['text'], key))
            else:
                base = key.split('_')[0]
                found = False
                if base in self.semantic_equivalents:
                    for equiv in self.semantic_equivalents[base]:
                        if any(equiv in j for j in judge_set):
                            matched.append((ent['text'], equiv))
                            found = True
                            break
                if not found:
                    ent_emb = self._embed_text(ent['text'])
                    for je in judge_entities:
                        je_emb = self._embed_text(je['text'])
                        sim = float(util.cos_sim(ent_emb, je_emb))
                        if sim >= self.embed_threshold:
                            matched.append((ent['text'], je['normalized']))
                            found = True
                            break
                if not found:
                    fuzzy_found = False
                    for je in judge_entities:
                        if fuzz.partial_ratio(ent['text'], je['text']) > 80:
                            matched.append((ent['text'], je['normalized']))
                            fuzzy_found = True
                            break
                    if not fuzzy_found:
                        missing.append({'text': ent['text'], 'type': ent.get('label'), 'normalized': key})
        recall = len(matched) / len(conv_map) if conv_map else 1.0
        details = {'matched': matched, 'total': len(conv_map), 'missing': missing}
        return recall, details

    def _extract_entities(self, text: str) -> List[Dict]:
        entities = []
        doc = self.nlp(text)
        for ent in doc.ents:
            normalized = self._normalize_entity(ent.text, ent.label_)
            entities.append({'text': ent.text, 'label': ent.label_, 'normalized': normalized})
        currency_patterns = [
            (r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'MONEY'),
            (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'MONEY'),
            (r'(?:INR|Rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'MONEY')
        ]
        for pattern, label in currency_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                v = m.group(1).replace(',', '')
                entities.append({'text': m.group(0), 'label': label, 'normalized': f'amount_{v}'})
        time_patterns = [(r'(\d+)\s*(hours?|hrs?)', 'TIME'), (r'(\d+)\s*(days?)', 'TIME')]
        for pattern, label in time_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                num = m.group(1)
                unit = 'hours' if 'hour' in m.group(2).lower() or 'hr' in m.group(2).lower() else 'days'
                entities.append({'text': m.group(0), 'label': label, 'normalized': f'time_{num}_{unit}'})
        action_patterns = [
            (r'(?:raised|created|opened)\s+(?:a\s+)?ticket', 'ACTION', 'ticket_raised'),
            (r'(?:roaming|service)\s+(?:was\s+)?disabled', 'STATUS', 'roaming_disabled'),
            (r'reversal|reversed|refund', 'ACTION', 'reversal')
        ]
        for pattern, label, norm in action_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({'text': m.group(0), 'label': label, 'normalized': norm})
        date_matches = re.findall(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{1,2}\b', text, flags=re.IGNORECASE)
        for dm in date_matches:
            dp = dateparser.parse(dm)
            if dp:
                entities.append({'text': dm, 'label': 'DATE', 'normalized': dp.strftime("date_%Y_%m_%d")})
            else:
                entities.append({'text': dm, 'label': 'DATE', 'normalized': dm.replace(' ', '_')})
        dedup = {}
        for e in entities:
            key = e.get('normalized') or e['text'].lower().strip()
            if key not in dedup:
                dedup[key] = e
        return list(dedup.values())

    def _normalize_entity(self, text: str, label: str) -> str:
        txt = text.lower().strip()
        if label == "MONEY":
            digits = re.findall(r'\d+', txt)
            if digits:
                return f"amount_{digits[0]}"
        if label == "DATE":
            dp = dateparser.parse(text)
            if dp:
                return dp.strftime("date_%Y_%m_%d")
            else:
                return txt.replace(' ', '_')
        if label == "TIME":
            tm = re.search(r'(\d+)\s*(hours?|days?)', txt)
            if tm:
                num = tm.group(1)
                unit = 'hours' if 'hour' in tm.group(2) else 'days'
                return f"time_{num}_{unit}"
        return txt.replace(' ', '_')

    def _compute_discourse_recall(self, conversation: str, judgment: str) -> Tuple[float, Dict]:
        conv_units = self._extract_discourse_units(conversation)
        judge_units = self._extract_discourse_units(judgment)
        if not conv_units:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        conv_by_type = defaultdict(list)
        for u in conv_units:
            if u['type'] != 'OTHER':
                conv_by_type[u['type']].append(u)
        if not conv_by_type:
            return 1.0, {'matched': 0, 'total': 0, 'missing': []}
        judge_by_type = defaultdict(list)
        for u in judge_units:
            if u['type'] != 'OTHER':
                judge_by_type[u['type']].append(u)
        matched = 0
        missing_types = []
        matched_pairs = []
        for unit_type, units in conv_by_type.items():
            if unit_type in judge_by_type and judge_by_type[unit_type]:
                matched += 1
                matched_pairs.append(unit_type)
            else:
                missing_types.append({'type': unit_type, 'examples': [u['content'] for u in units[:2]]})
        recall = matched / len(conv_by_type)
        details = {'matched': matched, 'total': len(conv_by_type), 'missing': missing_types, 'matched_types': matched_pairs}
        return recall, details

    def _extract_discourse_units(self, text: str) -> List[Dict]:
        units = []
        doc = self.nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            st = s.lower()
            typ = 'OTHER'
            for k, kw in self.discourse_types.items():
                if any(w in st for w in kw):
                    typ = k
                    break
            units.append({'type': typ, 'content': s})
        return units

    def _embed_text(self, text: str):
        if not text:
            return self.embed_model.encode("", convert_to_tensor=True, normalize_embeddings=True)
        k = text
        if k in self.embed_cache:
            return self.embed_cache[k]
        emb = self.embed_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        self.embed_cache[k] = emb
        return emb

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
