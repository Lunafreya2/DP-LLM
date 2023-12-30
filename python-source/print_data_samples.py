import sys

import pandas as pd

data = pd.read_csv(f"{sys.argv[1]}")

print(data)
