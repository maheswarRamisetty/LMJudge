
import pandas as pd
from conv_chunker import ConvChunker
from summary_extract import SummaryExtractor
from nli import NLIModel
from nli_entailment import compute_nli_entailment
from tqdm import tqdm

INPUT_CSV = "../data/Data-master/data.csv"
OUTPUT_CSV = "nli_entailment_output.csv"


def main():
    from cfg import DATA_PATH
    df = pd.read_csv(DATA_PATH)

    chunker = ConvChunker(max_tokens=300, overlap_tokens=50)
    fact_extractor = SummaryExtractor()
    nli_model = NLIModel()

    nli_scores = []
    fact_level_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="NLI entailment"):
        conversation = str(row["call_conversation"])
        summary = str(row["call_summary"])

        chunks = chunker._build_chunks(conversation)
        facts = fact_extractor.extract_fs(summary)

        nli_score, per_fact = compute_nli_entailment(
            facts=facts,
            conversation_chunks=chunks,
            nli_model=nli_model
        )

        nli_scores.append(nli_score)
        fact_level_scores.append(per_fact)

    df["nli_entailment"] = nli_scores
    df["fact_entailment_scores"] = fact_level_scores

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved NLI results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()