import re
from nli_model import NLIModel
from data_loader import load_csv
from claim_rewrite import ClaimRewriter
import numpy as np
from cfg import ALPHA,ALPHA_N,ALPHA_S

class JudgmentParser:
    def __init__(self):
        self.positive = [
            r".*effectively conveys\s+(.*)",
            r".*effectively captures\s+(.*)",
            r".*includes\s+(.*)",
            r".*mentions\s+(.*)",
            r".*covers\s+(.*)",
            r".*describes\s+(.*)",
            r".*highlights\s+(.*)"
        ]

        self.negative = [
            r".*omits\s+(.*)",
            r".*does not mention\s+(.*)",
            r".*fails to include\s+(.*)",
            r".*misses\s+(.*)",
            r".*lacks\s+(.*)",
            r".*neglects\s+(.*)"
        ]

    def extract_claims(self, judgment):
        positives = []
        negatives = []

        sentences = re.split(r'[.!?]+', judgment)
        sentences = [s.strip() for s in sentences if s.strip()]

        for s in sentences:
            for p in self.positive:
                m = re.match(p, s, re.IGNORECASE)
                if m:
                    positives.append(self._clean(m.group(1)))

            for p in self.negative:
                m = re.match(p, s, re.IGNORECASE)
                if m:
                    negatives.append(self._clean(m.group(1)))

        return list(set(positives)), list(set(negatives))

    def _clean(self, text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text[0].upper() + text[1:] if text else text


class JudgmentAlignment:
    def __init__(self):
        self.nli = NLIModel()
        self.parser = JudgmentParser()
        self.rewriter = ClaimRewriter()
        self.summaries, self.judgments = load_csv("../df_infer_informativeness_9.csv")
        self.final_pos=[]
        self.final_neg=[]

    def evaluate(self):
        for i, (summary, judgment) in enumerate(zip(self.summaries, self.judgments)):
            pos, neg = self.parser.extract_claims(judgment)

            pos_h = [self.rewriter.rewrite_positive(c) for c in pos]
            neg_h = [self.rewriter.rewrite_negative(c) for c in neg]

            if pos_h:
                pos_scores = self.nli.entailment_score(
                    [summary] * len(pos_h),
                    pos_h
                )
            else:
                pos_scores = np.array([])

            if neg_h:
                neg_scores = self.nli.entailment_score(
                    [summary] * len(neg_h),
                    neg_h
                )
            else:
                neg_scores = np.array([])

            print(f"\nExample {i+1}")
            print("Positive hypotheses:", pos_h)
            print("Positive entailment:", pos_scores)
            print("Negative hypotheses:", neg_h)
            print("Negative entailment:", neg_scores)
            
            self.final_pos.extend([i,pos_scores.tolist()])
            self.final_neg.extend([i,neg_scores.tolist()])

            final_score = self.aggregate(pos_scores, neg_scores)
            print("Final alignment score:", round(final_score, 3))
            
        return self.final_pos,self.final_neg

    def aggregate(self, pos_scores, neg_scores):
        pos_mean = pos_scores.mean() if len(pos_scores) else 0.0
        neg_penalty = neg_scores.mean() if len(neg_scores) else 0.0
        return max(pos_mean - neg_penalty, 0.0)


# def call() -> None:
#     ja = JudgmentAlignment()
#     return ja.evaluate()
    


if __name__ == "__main__":
    ja = JudgmentAlignment()
    ja.evaluate()
