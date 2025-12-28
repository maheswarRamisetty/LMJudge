EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "roberta-large-mnli"  
ALPHA_S = 0.5   
ALPHA_N = 0.5   
EMBED_BATCH_SIZE = 64
NLI_BATCH_SIZE = 16
DATA_PATH=--------
TOKENIZER_NAME ="sentence-transformers/all-mpnet-base-v2"
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 0.5
