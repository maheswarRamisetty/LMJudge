import pandas as pd
from cfg import DATA_PATH, CALL_CONV_PATH


def load_csv(main_path, conv_path):
    df_main = pd.read_csv(main_path)
    df_conv = pd.read_csv(conv_path)

    # print(df_conv['completion'].head())
    
    df_main['call_conversation'] = df_conv['completion']
    df_main.to_csv(DATA_PATH, index=False)
    return df_main

if __name__ == "__main__":
    df = load_csv(DATA_PATH, CALL_CONV_PATH)
    print(df.head())