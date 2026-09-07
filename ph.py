import pandas as pd
df = pd.read_csv(r"C:\Users\stme\Desktop\phase3\data\healthfc\healthfc.csv")
print(df['evidence_tier'].head(20).tolist())
print(df['evidence_tier'].unique())