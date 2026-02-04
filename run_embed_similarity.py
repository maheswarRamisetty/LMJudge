import pandas as pd
from tqdm import tqdm
from conv_chunker import ConvChunker
from embedding_similarity import EmbeddingSimilarity
from cfg import DATA_PATH
from data_loader import load_csv


def main():
    summaries, conversations, judgments = load_csv(DATA_PATH)

    chunker = ConvChunker(max_tokens=300, overlap_tokens=50)
    embed_sims = EmbeddingSimilarity()


    #[... ... ... ...] -> []-> 768
    #[.... ....] -> 768


    
    total_s = 0.0
    count = 0

    for summary, judgment in tqdm(
        zip(summaries, judgments),
        total=len(judgments),
        desc="Jusdment-Summary Similarity"
    ):
        summary = str(summary)
        judgment = str(judgment)

        judgment_chunks = chunker._build_chunks(judgment)

        _, f_s = embed_sims.summary_to_chunk_similarity(
            summary,
            judgment_chunks
        )

        total_s += f_s
        count += 1

    return total_s / count


if __name__ == "__main__":
    print(main())