import numpy as np
from sentence_transformers import SentenceTransformer
from utils import cosine_sim
from tqdm import tqdm
from cfg import EMBEDDING_MODEL
class EmbeddingSimilarity:
    
    def __init__(self,model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        
    def embed(self,texts):
        return self.model.encode(texts,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
    
    def summary_to_chunk_similarity(self,summary,chunks):
        
        embedded_summary = self.embed(summary)
        embedded_chunks = self.embed(chunks)
        
        sims = []
        
        for chunk in embedded_chunks:
            sims.append(cosine_sim(embedded_summary,chunk))
            
        f_s = float(max(sims)) if sims else 0.0
        return sims,f_s 