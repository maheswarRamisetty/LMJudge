import spacy
import numpy as np
import re
from typing import List,Tuple


class RelevanceModule:
	def __init__(self,model="en_core_web_sm"):
		self.nlp=spacy.load(model)

	def compute_relevance(self,conversation:str,judgment:str)->float:
		context_element = self._extract_and_norm_eles(conversation)
		judgment_element = self._extract_and_norm_eles(judgment)

		if not context_element and not judgment_element:
			return 0.0

		common = set(context_element).intersection(set(judgment_element))
		union = set(context_element).union(set(judgment_element))
		relevance = len(common)/len(judgment_element) if judgment_element else 0.0
		return relevance

	def _extract_and_norm_eles(self,text:str)->List[str]:
		doc = self.nlp(text.lower())
		elements = set()

		for ent in doc.ents:
			normalized = self._normalized_ent(ent.text,ent.label_)
			elements.add(normalized)

		for chunk in doc.noun_chunks:
			if 1<=len(chunk.text.split())<=3:
				normalized=re.sub(r'[^\w\s-]', '', chunk.text)
				normalized = re.sub(r'\s+', '_', normalized)
				elements.add(normalized)

		for token in doc:
			if token.pos_=="VERB" and token.lemma_ not in ['be','have','do']:
				elements.add(token.lemma_)

		nums = re.findall(r'\b\d+\b', text)
		for num in nums:
			elements.add(f"num_{num}")

		return list(elements)


	def _normalized_ent(self,text:str,label:str)->str:

		normalized = text.lower().strip()
		normalized = re.sub(r'[^\w\s-]', '', normalized)
		normalized = re.sub(r'\s+', '_', normalized)

		if label in ["MONEY","CARDINAL"]:
			return f"amt_{normalized}"


		elif label == "DATE":
			return f"date_{normalized}"

		elif label == "GPE":
			return f"loc_{normalized}"

		elif label == "PERSON":
			return f"person_{normalized}"

		else:
			return normalized


if __name__=="__main__":
	from cfg import DATA_PATH
	from data_loader import load_csv
	summaries,convs,judgments = load_csv(DATA_PATH)
	rV = RelevanceModule()
	for idx,(c,j) in enumerate(zip(convs[:10],judgments[:10])):
		ans = rV.compute_relevance(c,j)
		print(ans)







