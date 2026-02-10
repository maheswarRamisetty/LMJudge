import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_CSV = "../data.csv"
OUTPUT_DIR = "topic_datasets"
PLOT_FILE = "topic_distribution.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV)

df = df.dropna(subset=["topic"])

topic_counts = (
    df["topic"]
    .value_counts()
    .sort_values(ascending=False)
)

print("Topic Distribution:")
print(topic_counts)

plt.figure(figsize=(12, 6))
topic_counts.plot(kind="bar")

plt.title("Distribution of Conversations over Unique Topics")
plt.xlabel("Topic")
plt.ylabel("Number of Conversations")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(PLOT_FILE)
plt.show()

data1 = df[df['topic']=='Hobbies']
data2 = df[df['topic']=='Transportation']
data3 = df[df['topic']=='astronomy']
data4 = df[df['topic']=='ecosystems']

combined = pd.concat([data1,data2,data3,data4],ignore_index=False)
combined.to_csv("three.csv",index=False)

# for topic, count in topic_counts.items():
#     topic_df = df[df["topic"] == topic]
#     safe_topic_name = topic.replace(" ", "_").lower()

#     output_path = os.path.join(
#         OUTPUT_DIR,
#         f"{safe_topic_name}_dataset.csv"
#     )

#     topic_df.to_csv(output_path, index=False)

# print(f"\nCreated {len(topic_counts)} topic-wise datasets in '{OUTPUT_DIR}'")
