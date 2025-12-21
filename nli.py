import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class NLIModel:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)


    def entailment_prob(self, premises, hypotheses, batch_size=16):
        probs = []

        for i in tqdm(range(0, len(premises), batch_size), desc="NLI batches"):
            p_batch = premises[i:i + batch_size]
            h_batch = hypotheses[i:i + batch_size]

            enc = self.tokenizer(
                p_batch,
                h_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
                softmax = torch.softmax(logits, dim=-1)
                entailment = softmax[:, 2] 

            probs.extend(entailment.cpu().numpy())

        return np.array(probs)
