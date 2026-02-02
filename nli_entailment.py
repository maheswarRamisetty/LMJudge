import numpy as np
def compute_nli_entailment(facts, conversation_chunks, nli_model):
    if not facts or not conversation_chunks:
        return 0.0, []

    fact_entailments = []

    for fact in facts:
        premises = conversation_chunks
        hypotheses = [fact] * len(conversation_chunks)

        scores = nli_model.entailment_prob(premises, hypotheses)
        max_score = float(np.max(scores))
        fact_entailments.append(max_score)

    final_nli = float(np.mean(fact_entailments))
    return final_nli, fact_entailments