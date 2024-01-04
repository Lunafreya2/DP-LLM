import random

import numpy as np
import pandas as pd
import torch

# from transformers import (AdamW, BertConfig, BertForSequenceClassification,
#                           BertTokenizer, get_linear_schedule_with_warmup)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 19

## Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if device == torch.device("cuda:0"):
    torch.cuda.manual_seed_all(SEED)


SYN_DATA_DIR = "./data/synthetic"
ORG_DATA_DIR = "./data/org"

df_train_syn = pd.read_csv(
    f"{SYN_DATA_DIR}/train_16k.csv", delimiter=",", names=["text", "label"]
)
df_train_org = pd.read_csv(
    f"{ORG_DATA_DIR}/train_16k.csv", delimiter=",", names=["text", "label"]
)

df_test = pd.read_csv(
    f"{ORG_DATA_DIR}/test_2k.csv", delimiter=",", names=["text", "label"]
)
