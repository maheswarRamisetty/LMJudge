import re
import spacy

class SummaryExtractor:
    def __init__(self,model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        
    def extract_fs(self,text):
        doc = self.nlp(text)
        facts =[]
        for x in doc.sents:
            sent = x.text.strip()
            if len(sent.split())>=5:
                facts.append(sent)
        return facts
    
if __name__=="__main__":
    sE = SummaryExtractor()
    print(sE.extract_fs("Judy Liao visited in person to confirm the receipt of her application for the accounts assistant position. The representative confirmed that her resume, cover letter, and letters of recommendation were received and that no additional documents were needed. Judy inquired about the timeline for interviews and was informed that the company is still accepting applications and may begin scheduling interviews in a week or two. The conversation concluded with an offer to assist further if needed."))