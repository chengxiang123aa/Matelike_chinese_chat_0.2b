import pandas as pd

df = pd.read_parquet('full_sft_dataset_3m_based_belle.parquet')
print(len(df))