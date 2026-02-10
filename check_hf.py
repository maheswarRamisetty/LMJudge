import json
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm
import uuid

SUMMARY_PROMPT = """
You are given a conversation transcript.

Write a clear, coherent, and informative summary in 4–6 sentences.

The summary should:
- capture the main topics, goals, and context of the discussion,
- include important actions taken, clarifications made, and problems addressed,
- mention any conclusions, solutions, or next steps,
- avoid irrelevant details or speculation,
- use only information explicitly stated in the transcript.

Conversation Transcript:
{conversation}
"""

gpt_5creds_path = r"C:\Users\RMSTVNMFST\creds\api_creds.json"
with open(gpt_5creds_path, "r") as file:
    gpt_5_creds = json.load(file)

client = AzureOpenAI(
    azure_endpoint=gpt_5_creds["endpoint"],
    api_key=gpt_5_creds["api_key"],
    api_version=gpt_5_creds["model_version"],
)

deployment_name = "gpt-5"

df = pd.read_csv("./data/three/three.csv")

if 'summary' not in df.columns:
    df['summary'] = pd.NA

def generate_summary(full_topic, prompt, completion_text):
    conversation = f"Topic: {full_topic}\n\nUser: {prompt}\n\nAssistant: {completion_text}"
    final_prompt = SUMMARY_PROMPT.format(conversation=conversation)

    messages = [
        {"role": "system", "content": "You are a precise conversation summarizer."},
        {"role": "user", "content": final_prompt + f"\n\n[uid={uuid.uuid4()}]"}
    ]

    completion = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        stream=False
    )

    return completion.choices[0].message.content.strip()

for idx, row in tqdm(df.head(200).iterrows(), total=200, desc="Generating summaries"):
    if pd.isna(df.at[idx, 'summary']):
        try:
            summary = generate_summary(row['full_topic'], row['prompt'], row['completion'])
            df.at[idx, 'summary'] = summary
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            df.at[idx, 'summary'] = pd.NA

df.to_csv("datasummareis.csv", index=False)
print("Summaries generated and saved!")
