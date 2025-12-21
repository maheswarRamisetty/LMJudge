import pandas as pd
import json
import string
from utils import simple_normalize

POSSIBLE_SUMMARY_COLS = ['summary', 'context', 'conversation_summary', 'prompt_summary']
POSSIBLE_JUDGMENT_COLS = ['reasoning', 'judgment', 'judgement', 'judge', 'decision', 'response_reasoning']

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    summary_col = None
    judgment_col = None

    for cand in POSSIBLE_SUMMARY_COLS:
        for c in df.columns:
            if cand in c.lower():
                summary_col = c
                break
        if summary_col:
            break

    for cand in POSSIBLE_JUDGMENT_COLS:
        for c in df.columns:
            if cand in c.lower():
                judgment_col = c
                break
        if judgment_col:
            break
    if summary_col is None:
        summary_col = df.columns[0]
    if judgment_col is None:
        if 'reasoning' in cols:
            judgment_col = df.columns[cols.index('reasoning')]
        else:
            judgment_col = df.columns[-1]

    return summary_col, judgment_col


def get_summary(text):
    start = "Conversation Summary:"
    if start not in text:
        return None
    
    next = text.split(start, 1)[1]
    end_marker = "<|eot_id|>"
    if end_marker in next:
        next = next.split(end_marker, 1)[0]
    
    return next.strip()
    

def load_csv(path):
    df = pd.read_csv(path)
    # print(df.columns)
    inputs = df["input"]
    c=0
    summaries=[]
    generated = df["generated"]
    for x in inputs[:4]:
        if 'Conversation Summary:' in x:
            summary = get_summary(x)
            summaries.append(summary)
            # print("Summary:", summary)
        else:
            print("No")
            
    judgments=[]
            
    for idx,j in enumerate(generated[:4]):
        try:
            data = json.loads(j)
            judgments.append(data.get("reasoning", ""))
            
        except Exception as e:
            print(f"Row {idx} - JSON parsing error:", e)
            judgments.append("")
            
            
    return summaries, judgments
    # for p in judgments:
    #     print("Judgment:", p)

if __name__ == "__main__":
                
    print(load_csv("../df_infer_informativeness_9.csv"))