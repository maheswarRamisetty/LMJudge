EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "roberta-large-mnli"  
ALPHA_S = 0.5   
ALPHA_N = 0.5   
EMBED_BATCH_SIZE = 64
prompt = "Based on the above conversation, what is the decision regarding the customer's billing dispute?"
NLI_BATCH_SIZE = 16
DATA_PATH="C:\Users\RMSTVNMFST\mahesh\judge\output_on_summaries.csv"
TOKENIZER_NAME ="sentence-transformers/all-mpnet-base-v2"
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.5
