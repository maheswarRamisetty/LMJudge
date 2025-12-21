import argparse
from ast import List
from typing import Tuple
import pandas as pd
from data_loader import load_csv
from text_processing import minimal_clean
from embedding_model import Embedder
from cfg import ALPHA,ALPHA_N,ALPHA_S
from nli_model import NLIModel
from coherence import compute_embed_sim, compute_cpath
from utils import to_pairs
from cfg import DEVICE
from tqdm import tqdm
from itertools import zip_longest
import numpy as np

def main(input_csv, output_csv, force_cols=None):
    # df, summary_col, judgment_col = load_csv(input_csv)
    # print(f"Detected summary column: {summary_col}; judgment column: {judgment_col}")

    # df['_summary_text'] = df['_summary_text'].apply(minimal_clean)
    # df['_judgment_text'] = df['_judgment_text'].apply(minimal_clean)
            
    # embedder = Embedder()
    # nli = NLIModel()

    # summaries = df['_summary_text'].tolist()
    # judgments = df['_judgment_text'].tolist()

    # print("Computing embeddings for summaries...")
    # sum_emb = embedder.embed_texts(summaries)
    # print("Computing embeddings for judgments...")
    # jud_emb = embedder.embed_texts(judgments)

    # print("Computing EmbedSim (cosine)...")
    # embed_sim_scores = compute_embed_sim(sum_emb, jud_emb)

    # print("Computing NLI entailment scores...")
    # nli_entail_scores = nli.entailment_score(summaries, judgments)

    # cpath = compute_cpath(embed_sim_scores, nli_entail_scores)


    # df['embed_sim'] = embed_sim_scores
    # df['nli_entailment'] = nli_entail_scores
    # df['cpath'] = cpath

    # df.to_csv(output_csv, index=False)
    # print(f"Saved output with embed_sim/nli_entailment/cpath to {output_csv}")
    
    pass


def calculate_coherence(csv_path):
    summaries,judgments = load_csv(csv_path)
    embedder = Embedder()
    scores = []
    for summary,judgment in tqdm(zip(summaries,judgments),desc="Calculating Coherence",total=len(summaries)):
        sum_emb = embedder.embed_texts([summary])[0]
        jud_emb = embedder.embed_texts([judgment])[0]
        sim_score = compute_embed_sim([sum_emb],[jud_emb])[0]
        scores.append(sim_score)
    return scores


def calculate_nli():
    from split_judgment import JudgmentAlignment
    jA = JudgmentAlignment()
    return jA.evaluate()


def compute_coherence(pos_flat, neg_flat, alpha=0.5):
    pos_pairs = to_pairs(pos_flat)
    neg_pairs = to_pairs(neg_flat)
    neg_dict = {idx: vals for idx, vals in neg_pairs}
    results = {}

    for idx, pos_vals in pos_pairs:
        neg_vals = neg_dict.get(idx, [])

        pos_mean = np.mean(pos_vals) if pos_vals else 0.0
        neg_mean = np.mean(neg_vals) if neg_vals else 0.0
    #0.5
        score = alpha * pos_mean + (1 - alpha) * (1 - neg_mean)
        results[idx] = round(float(score), 4)

    return results


if __name__ == "__main__":  
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", "-i", required=True, help="Input CSV path")
    # parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    # args = parser.parse_args()
    # main(args.input, args.output)
    ans = calculate_coherence("../df_infer_informativeness_9.csv")
    ans1,ans2 = calculate_nli()
    # print("Coherence Scores:", ans1,ans2)
    final_coherence = compute_coherence(ans1,ans2,alpha=ALPHA)
    c_path = ALPHA_S * np.array(ans) + ALPHA_N * np.array(list(final_coherence.values()))
    print(np.sum(c_path)/len(c_path))
    


        
