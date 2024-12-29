import os
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# import kaggle_evaluation.jane_street_inference_server

input_path = 'data'
feature_path = "features.csv"
responders_path = "responders.csv"
sample_submission_path = "sample_submission.csv"

feature_df = pd.read_csv(os.path.join(input_path, feature_path))
responders_df = pd.read_csv(os.path.join(input_path, responders_path))
sample_submission_data = pd.read_csv(os.path.join(input_path, sample_submission_path))
df_partition0 = pd.read_parquet(os.path.join(input_path, 'train.parquet/partition_id=0'))

# 可视化
responder_6_data = df_partition0['responder_6']
plt.plot(responder_6_data[:1000])
plt.savefig('viz/responder_6.jpg')

print(df_partition0['symbol_id'].unique())

print(sample_submission_data.shape)