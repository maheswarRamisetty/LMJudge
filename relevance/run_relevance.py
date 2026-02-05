import argparse
from relevance.relevance_parser import RelevanceParser
from cfg import DATA_PATH
from data_loader import load_csv
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        
        choices=["lexical", "semantic"],
        default="lexical"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3
    )

    args = parser.parse_args()

    summaries, conversations, judgments = load_csv(DATA_PATH)
    scores =[]
    if args.mode == "semantic":
        relevance = RelevanceParser(
            mode="semantic",
            threshold=args.threshold
        )
    else:
        relevance = RelevanceParser(mode="lexical")

    for i, (s,c,j) in enumerate(
        zip(summaries, conversations,judgments), start=1
    ):
        score = relevance.compute(c,s,j)
        scores.append(score)
        print(f"Doing.. {i}: {score:.7f}")
    print("Overall : ",np.mean(scores))

if __name__ == "__main__":
    main()