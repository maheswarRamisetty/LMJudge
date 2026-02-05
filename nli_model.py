from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from cfg import NLI_MODEL, DEVICE, NLI_BATCH_SIZE
from tqdm import tqdm
from data_loader import load_csv

class NLIModel:
    def __init__(self, model_name=NLI_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device

    def entailment_score(self, premises, hypotheses, batch_size=NLI_BATCH_SIZE, max_length=512):
        assert len(premises) == len(hypotheses)
        scores = []
        for i in tqdm(range(0, len(premises), batch_size), desc="NLI batches"):
            batch_p = premises[i:i+batch_size]
            batch_h = hypotheses[i:i+batch_size]
            enc = self.tokenizer(batch_p, batch_h, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
            enc = {k:v.to(self.device) for k,v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits 
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                entail = probs[:, 2]
                scores.extend(list(entail))
        return np.array(scores, dtype=float)
    
    
if __name__ == "__main__":

    model = NLIModel()
    summaries, judgments = load_csv("../df_infer_informativeness_9.csv")
    print("--------------- SUMMARIES ---------------")
    for x in summaries:
        print("Summary:", x)

    print("--------------- JUDGMENTS ---------------")

    for x in judgments:
        print("Judgment:", x)   
        
        
        
    scores = model.entailment_score(summaries, judgments)
    print(scores)