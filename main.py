import pandas as pd
from cfg import DATA_PATH, ALPHA, ALPHA_N, ALPHA_S
from data_loader import load_csv
from conv_chunker import ConvChunker
from summary_extract import SummaryExtractor
from nli import NLIModel
from embedding_similarity import EmbeddingSimilarity
from completeness.completeness_module import SummaryCompletenessEvaluator
from AccuracyModule import AccuracyCalculator
from nli_entailment import compute_nli_entailment
from relevance.relevance_parser import RelevanceParser
from clarity.clarity_module import ClarityCalculator
import tqdm
import numpy as np
import os

weights = {
    'wA': 0.25,
    'wR': 0.25,
    'wC': 0.25,
    'wCL': 0.25
}

def run_evaluation(csv_path):
    summaries, conversations, judgments = load_csv(csv_path)

    chunker = ConvChunker(max_tokens=300, overlap_tokens=50)
    fact_extractor = SummaryExtractor()
    nli_model = NLIModel()
    embed_sims = EmbeddingSimilarity()
    relevance_module = RelevanceParser(mode="semantic", threshold=0.6)

    results = []

    for idx, (s, c, j) in enumerate(zip(summaries, conversations, judgments), start=1):

        summary = str(s)
        judgment = str(j)

        # Embedding Similarity
        judgment_chunks = chunker._build_chunks(judgment)
        _, f_s = embed_sims.summary_to_chunk_similarity(
            summary, judgment_chunks
        )

        # NLI Entailment
        chunks = chunker._build_chunks(c)
        facts = fact_extractor.extract_fs(summary)
        nli_score, _ = compute_nli_entailment(
            facts=facts,
            conversation_chunks=chunks,
            nli_model=nli_model
        )

        # Accuracy
        acc = AccuracyCalculator().compute_accuracy(
            conversation_text=c,
            judgment_text=judgment,
            include_prompt=False
        )['accuracy_score']

        # Completeness
        comp = SummaryCompletenessEvaluator().evaluate_completeness(
            conversation=c,
            summary=summary
        )['completeness_score']

        # Relevance
        rel = relevance_module.compute(
            conversation=c,
            summary=summary,
            judgment=judgment
        )

        # Clarity
        clarity = ClarityCalculator().compute_clarity_score(c)['clarity_score']

        C_Path = ALPHA_S * f_s + ALPHA_N * nli_score
        J_Score = (
            weights["wA"] * acc +
            weights["wR"] * rel +
            weights["wC"] * comp +
            weights["wCL"] * clarity
        )

        JCJS = ALPHA * C_Path + (1 - ALPHA) * J_Score

        results.append({
            "Example": idx,
            "Embedding": round(f_s, 4),
            "NLI": round(nli_score, 4),
            "Accuracy": round(acc, 4),
            "Completeness": round(comp, 4),
            "Relevance": round(rel, 4),
            "Clarity": round(clarity, 4),
            "JCJS": round(JCJS, 4)
        })

    return results


def evaluate_single(summary, conversation, judgment):
    from cfg import ALPHA, ALPHA_N, ALPHA_S

    chunker = ConvChunker(max_tokens=300, overlap_tokens=50)
    fact_extractor = SummaryExtractor()
    nli_model = NLIModel()
    embed_sims = EmbeddingSimilarity()
    relevance_module = RelevanceParser(mode="semantic", threshold=0.6)

    # Embedding Similarity
    judgment_chunks = chunker._build_chunks(judgment)
    _, f_s = embed_sims.summary_to_chunk_similarity(
        summary, judgment_chunks
    )

    # NLI
    chunks = chunker._build_chunks(conversation)
    facts = fact_extractor.extract_fs(summary)
    nli_score, _ = compute_nli_entailment(
        facts=facts,
        conversation_chunks=chunks,
        nli_model=nli_model
    )

    # Accuracy
    acc = AccuracyCalculator().compute_accuracy(
        conversation_text=conversation,
        judgment_text=judgment,
        include_prompt=False
    )['accuracy_score']

    # Completeness
    comp = SummaryCompletenessEvaluator().evaluate_completeness(
        conversation=conversation,
        summary=summary
    )['completeness_score']

    # Relevance
    rel = relevance_module.compute(
        conversation=conversation,
        summary=summary,
        judgment=judgment
    )

    # Clarity
    clarity = ClarityCalculator().compute_clarity_score(
        conversation
    )['clarity_score']

    C_Path = ALPHA_S * f_s + ALPHA_N * nli_score
    J_Score = (
        0.25 * acc +
        0.25 * rel +
        0.25 * comp +
        0.25 * clarity
    )

    JCJS = ALPHA * C_Path + (1 - ALPHA) * J_Score

    return {
        "Embedding": round(f_s, 4),
        "NLI": round(nli_score, 4),
        "Accuracy": round(acc, 4),
        "Completeness": round(comp, 4),
        "Relevance": round(rel, 4),
        "Clarity": round(clarity, 4),
        "JCJS": round(JCJS, 4)
    }


def run_and_append(csv_path, output_csv):
    results = run_evaluation(csv_path)
    df = pd.DataFrame(results)

    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)

    print(f"Appended {len(df)} rows to {output_csv}")


if __name__=="__main__":
    OUTPUT_CSV = "output.csv"
    run_and_append(DATA_PATH,OUTPUT_CSV)
    
    