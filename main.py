from cfg import DATA_PATH
from data_loader import load_csv
from conv_chunker import ConvChunker
from summary_extract import SummaryExtractor
from nli import NLIModel
from embedding_similarity import EmbeddingSimilarity
from completeness.completeness_module import SummaryCompletenessEvaluator
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from utils.logger import _l
from nli_entailment import compute_nli_entailment
from cfg import prompt
from AccuracyModule import AccuracyModule

logger = _l("EVAL")

def main():
    logger.info("Loading Data")
    ok=False
    summaries, conversations, judgments = load_csv(DATA_PATH)
    logger.info("Initializing Embedding Module")
    chunker = ConvChunker(max_tokens=300, overlap_tokens=50)
    fact_extractor = SummaryExtractor()
    nli_model = NLIModel()
    embed_sims = EmbeddingSimilarity()
    for idx,(s,c,j) in tqdm(
		enumerate(zip(summaries, conversations, judgments), start=1),
		total=len(judgments),
		desc="Judgment-Summary Similarity"
	):
        summary = str(s)
        judgment = str(j)
        judgment_chunks = chunker._build_chunks(judgment)
        #f_s -> embedding sim..
        _, f_s = embed_sims.summary_to_chunk_similarity(
			summary,
			judgment_chunks
		)
        
        #turn for nli-entailment
        
        chunks = chunker._build_chunks(c)
        facts = fact_extractor.extract_fs(summary)
        
        nli_score, per_fact = compute_nli_entailment(
            facts=facts,
            conversation_chunks=chunks,
            nli_model=nli_model
        )
        
        # turn for Accuracy Module
        
        aM = AccuracyModule()
        result = aM.compute_accuracy(
            conversation_text=c,
            judgment_text=judgment,
            include_prompt=False,
            prompt_text=None    
        )

        acc_score = result['accuracy_score']
        
        # turn for completeness module
        ela = SummaryCompletenessEvaluator()
        COMP_RES =  ela.evaluate_completeness(
            conversation=c,
            summary=summary
        )
        
        comp_score = COMP_RES['completeness_score']
        
        # turn for relevance module
        
        

  
        
     
     
		

if __name__=="__main__":
	main()
