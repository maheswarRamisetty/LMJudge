from cfg import DATA_PATH
from data_loader import load_csv
from cfg import EMBED_BATCH_SIZE
from cfg import NLI_BATCH_SIZE
from cfg import EMBEDDING_MODEL
from conv_chunker import ConvChunker
from summary_extract import SummaryExtractor
from nli import NLIModel
from embedding_similarity import EmbeddingSimilarity
from completeness.completeness_module import SummaryCompletenessEvaluator
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from utils.logger import _l
from AccuracyModule import AccuracyCalculator
from nli_entailment import compute_nli_entailment
from cfg import prompt
from relevance.relevance_parser import RelevanceParser
from clarity.clarity_module import ClarityCalculator
from cfg import ALPHA,ALPHA_N,ALPHA_S

logger = _l("EVAL")
weights = {
    'wA':0.25,
    'wR':0.25,
    'wC':0.25,
    'wCL':0.25
}

def main():
    scores = 0.0
    logger.info("Loading Data")
    ok=False
    summaries, conversations, judgments = load_csv(DATA_PATH)
    # logger.info("Initializing Embedding Module")
    chunker = ConvChunker(max_tokens=300, overlap_tokens=50)
    l_d = len(judgments)
    fact_extractor = SummaryExtractor()
    nli_model = NLIModel()
    embed_sims = EmbeddingSimilarity()
    relevance_module = RelevanceParser(mode="semantic", threshold=0.6)
    
    for idx,(s,c,j) in tqdm(
		enumerate(zip(summaries, conversations, judgments), start=1),
		total=len(judgments),
		desc="Judgment-Summary Similarity"
	):
        
        logger.info(f"Processing Example {idx}"
                    )
        
        logger.info("Turn for Embedding Similarity Module")
        summary = str(s)
        judgment = str(j)
        judgment_chunks = chunker._build_chunks(judgment)
        #f_s -> embedding sim..
        _, f_s = embed_sims.summary_to_chunk_similarity(
			summary,
			judgment_chunks
		)
        
        #turn for nli-entailment
        logger.info("Turn for NLI Entailment Module")
        
        chunks = chunker._build_chunks(c)
        facts = fact_extractor.extract_fs(summary)
        
        nli_score, per_fact = compute_nli_entailment(
            facts=facts,
            conversation_chunks=chunks,
            nli_model=nli_model
        )
        
        # turn for Accuracy Module
        logger.info("Turn for Accuracy Module")
        
        aM = AccuracyCalculator()
        result = aM.compute_accuracy(
            conversation_text=c,
            judgment_text=judgment,
            include_prompt=False,
            prompt_text=None    
        )['accuracy_score']
        
        logger.info("Turn for Completeness Module")
        
        # turn for completeness module
        ela = SummaryCompletenessEvaluator()
        COMP_RES =  ela.evaluate_completeness(
            conversation=c,
            summary=summary
        )['completeness_score']
        
        
        
        # turn for relevance module
        logger.info("Turn for Relevance Module")        
        
        rel_score = relevance_module.compute(
            conversation_text=c,
            summary_text=summary,
            judgment_text=judgment
        )

        logger.info("Turn for Clarity Module")
        
        calc = ClarityCalculator()
        clarity_score = calc.compute_clarity_score(c)['clarity_score']
        
        # print(
        #     f"Example {idx} Scores:\n"
        #     f"Embedding Similarity Score: {f_s:.4f}\n"
        #     f"NLI Entailment Score: {nli_score:.4f}\n"
        #     f"Accuracy Score: {result:.4f}\n"
        #     f"Completeness Score: {COMP_RES:.4f}\n"
        #     f"Relevance Score: {rel_score:.4f}\n"
        #     f"Clarity Score: {clarity_score:.4f}\n"
        #     "---------------------------------------"
        # )
        
        
        C_Path = ALPHA_S*f_s + ALPHA_N*nli_score
        
        J_Score = weights["wA"]*result + weights["wR"]*rel_score + weights["wC"]*COMP_RES + weights["wCL"]*clarity_score
        
        JCJS = ALPHA*C_Path + (1-ALPHA)*J_Score
        
        print(
            f"Example {idx} JCJS Score: {JCJS:.4f}\n"
            "---------------------------------------"
        )
        
        scores += JCJS
        
        
        
    return scores,l_d

     
		

if __name__=="__main__":
    total_score, length = main()
    print(f"Final JCJS Score over {length} examples: {total_score/length:.4f}")  