import sys

import pandas as pd

raw_data_filepath = sys.argv[1]
data = pd.read_csv(f"{raw_data_filepath}")

# print(data)

print(f"Loaded data of shape: {data.shape}")

texts = list(set(data["text"]))

print(f"Total number of samples processed: {len(texts)}")

file_id = raw_data_filepath.split(".csv")[0]
file_name = f"{file_id}_p.csv"
with open(file_name, "w") as f:
    f.write(" |EndOfText|\n".join(texts))
