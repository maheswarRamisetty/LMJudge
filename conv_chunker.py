import numpy as np
import re
from cfg import TOKENIZER_NAME
from transformers import AutoTokenizer

class ConvChunker:
    def __init__(
        self,
        tokenizer_name=TOKENIZER_NAME,
        max_tokens=300,
        overlap_tokens=50
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
    def _split_persons(self,text):
        
        t = re.split(r'(?=#Person\d+#:)', text)
        return [i for i in t if i.strip()]
    
    def _token_length(self,text):
        return len(self.tokenizer.encode(text,add_special_tokens=False))
    
    def _build_chunks(self,text):
        
        t=self._split_persons(text)
        chunks = []
        current_chunk = []
        current_tokens =0
        
        for turn in t:
            turn_tokens = self._token_length(turn)
            if current_tokens + turn_tokens <= self.max_tokens:
                chunk_text = "".join(current_chunk)
                chunks.append(chunk_text)
                
                overlap_text = "".join(current_chunk[:-2]) if len(current_chunk) >2 else ""
                current_chunk = [overlap_text, turn] if overlap_text else [turn]
                current_tokens = self._token_length("".join(current_chunk))
                
            else:
                current_chunk.append(turn)
                current_tokens += turn_tokens
        if current_chunk:
            chunks.append("".join(current_chunk))
    
        return chunks
        
        
        
        
if __name__ == "__main__":
    pass
import numpy as np
import re
from cfg import TOKENIZER_NAME
from transformers import AutoTokenizer

class ConvChunker:
    def __init__(
        self,
        tokenizer_name=TOKENIZER_NAME,
        max_tokens=300,
        overlap_tokens=50
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
    def _split_persons(self,text):
        
        t = re.split(r'(?=#Person\d+#:)', text)
        return [i for i in t if i.strip()]
    
    def _token_length(self,text):
        return len(self.tokenizer.encode(text,add_special_tokens=False))
    
    def _build_chunks(self,text):
        
        t=self._split_persons(text)
        chunks = []
        current_chunk = []
        current_tokens =0
        
        for turn in t:
            turn_tokens = self._token_length(turn)
            if current_tokens + turn_tokens <= self.max_tokens:
                chunk_text = "".join(current_chunk)
                chunks.append(chunk_text)
                
                overlap_text = "".join(current_chunk[:-2]) if len(current_chunk) >2 else ""
                current_chunk = [overlap_text, turn] if overlap_text else [turn]
                current_tokens = self._token_length("".join(current_chunk))
                
            else:
                current_chunk.append(turn)
                current_tokens += turn_tokens
        if current_chunk:
            chunks.append("".join(current_chunk))
    
        return chunks
        
        
        
        
if __name__ == "__main__":
    pass