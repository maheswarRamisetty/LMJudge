import pandas as pd
from cfg import DATA_PATH, CALL_CONV_PATH


def load_csv(main_path, conv_path):
    df_main = pd.read_csv(main_path)
    df_conv = pd.read_csv(conv_path)

    # print(df_conv['completion'].head())
    
    df_main['call_conversation'] = df_conv['text']
    df_main.to_csv(DATA_PATH, index=False)
    return df_main

def x(path):
    df = pd.read_csv(path)
    embed_avg = df['Embedding'].mean()
    nli_avg = df['NLI'].mean()
    print(f"Average Embedding Score: {embed_avg:.4f}")
    print(f"Average NLI Score: {nli_avg:.4f}")
    accuracy_avg = df['Accuracy'].mean()
    completeness_avg = df['Completeness'].mean()
    relevance_avg = df['Relevance'].mean()
    clarity_avg = df['Clarity'].mean()
    jcjs_avg = df['JCJS'].mean()
    print(f"Average Accuracy: {accuracy_avg:.4f}")
    print(f"Average Completeness: {completeness_avg:.4f}")
    print(f"Average Relevance: {relevance_avg:.4f}")
    print(f"Average Clarity: {clarity_avg:.4f}")
    print(f"Average JCJS: {jcjs_avg:.4f}")

if __name__ == "__main__":
    df = load_csv(DATA_PATH, CALL_CONV_PATH)
    print(df.head())

    df = pd.read_csv(DATA_PATH)
    df.to_csv("data_with_call_merged_two.csv",index=False)
    print("File Saved")

    # path1 = "./data/one/output.csv"
    # # df2 = pd.read_csv("./data/two/two_output.csv")
    # path2 = "./data/three/three_output.csv"
    # path3 = "./data/four/four_output.csv"
    # path4 = "./data/five/five_output.csv"

    # print("Dataset One:",x(path1))
    # print("Dataset Two:",x(path2))
    # print("Dataset Three:",x(path3))
    # print("Dataset Four:",x(path4))
    # df = pd.read_csv("../data")
    # unique_topics = df['topic'].dropna().unique()
    # # print(len(unique_topics))

    # unique_topic_chains = (
    # df[["topic", "subtopic", "subsubtopic"]]
    # .dropna(how="all")
    # .drop_duplicates())

    # print(f"\nTotal unique topic hierarchies: {len(unique_topic_chains)}")
    # print(unique_topic_chains.head())