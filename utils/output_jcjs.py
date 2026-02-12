import pandas as pd
import numpy as np

df = pd.read_csv("../data/three/three_output.csv")
print(np.mean(df['JCJS']))