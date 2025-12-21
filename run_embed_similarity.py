import pandas as pd
from conv_chunker import ConvChunker
from tqdm import tqdm
from embedding_similarity import EmbeddingSimilarity

INPUT_CSV = "../data/Data-master/sft_informativeness_batch_1.csv"

def main():
    
    df = pd.read_csv(INPUT_CSV)
    chunker = ConvChunker(max_tokens=300,overlap_tokens=50)
    embed_sims = EmbeddingSimilarity()
    
    a_s=[]
    total_f_s =0.0
    for idx,row in tqdm(df.iterrows(),total=len(df),desc="Calculating Embed Similarity"):
        conv = str(row['call_conversation'])
        summary = str(row['call_summary'])
        
        chunks = chunker._build_chunks(conv)
        sims, f_s = embed_sims.summary_to_chunk_similarity(summary, chunks)
        a_s.append({
            'num_chunks':len(chunks),
            "chunk_sims":sims,
            "embed_s":f_s
        })
        total_f_s += f_s    
    
    return total_f_s/len(a_s)

if __name__=="__main__":
    print(main())
        
        
