import numpy as np
from jcjs_coherence.utils.utils import cosine_sim
from cfg import ALPHA_S, ALPHA_N
from tqdm import tqdm
from embedding_model import Embedder

def compute_embed_sim(embeddings_a, embeddings_b):
    sims = []
    for a, b in zip(embeddings_a, embeddings_b):
        sims.append(cosine_sim(a, b))
    return np.array(sims, dtype=float)

def compute_cpath(embed_sim_scores, nli_entailment_scores, alpha_s=ALPHA_S, alpha_n=ALPHA_N):
    return alpha_s * embed_sim_scores + alpha_n * nli_entailment_scores



if __name__ == "__main__":
    embedding = Embedder()    
    u=embedding.embed_texts(["The company reported increased revenue this quarter.", "This quarter, the firm saw a rise in its earnings."])
    cos = cosine_sim(u[0],u[1])
    print(cos)