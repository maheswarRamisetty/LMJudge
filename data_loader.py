import pandas as pd
import json
import string
from utils.utils import simple_normalize
from utils.utils import parse_json

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

    inputs = df["input"]
    generated = df["generated"]
    call_convs_col = df["call_conversation"]

    summaries = []
    call_convs = []
    judgments = []

    for x in inputs:
        if isinstance(x, str) and "Conversation Summary:" in x:
            summaries.append(get_summary(x))
        else:
            summaries.append("")

    for conv in call_convs_col:
        call_convs.append(conv if isinstance(conv, str) else "")

    for idx, j in enumerate(generated):
        reasoning = parse_json(j)
        if not reasoning:
            print(f"Row {idx} - JSON parsing error (salvaged empty)")
        judgments.append(reasoning)

    return summaries, call_convs, judgments

    # for p in judgments:
    #     print("Judgment:", p)

# def load_data(path):
#     df = pd.load_csv(path)

#     row = df.iloc[0]

#     summary = row['summary']
#     conversaton = row['call_conversation']
#     generated=row['generated']

#     try:

#         js = json.loads(generated)
#         judgment = js.get('reasoning',"")

#     except Exception as e:
#         print(f"Row {idx}")


if __name__ == "__main__":
    from cfg import DATA_PATH
    summaries , call_convs,judgments = load_csv(DATA_PATH)
    print(summaries[:1])
    print(call_convs[:1])
    print(judgments[:1])