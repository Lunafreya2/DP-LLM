# import datasets
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion = sys.argv[1]
# dataset_e = datasets.load_dataset("dair-ai/emotion", name="split", split="test")

model = GPT2LMHeadModel.from_pretrained("./finetunedmodel")
tokenizer = GPT2Tokenizer.from_pretrained("./finetunedtokenizer")

model.to(device)
model.eval()

input_ids = (
    torch.tensor(tokenizer.encode(f"[BOS]{emotion}[SEP]{sys.argv[2]}"))
    .unsqueeze(0)
    .to(device)
)  # Empty string as a prompt

temp = float(sys.argv[3])
top_k = int(sys.argv[4])

output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,  # Number of sequences to return
    num_beams=10,  # Number of beams for beam search
    temperature=temp,
    top_k=top_k,
    no_repeat_ngram_size=3,
    do_sample=True,
    early_stopping=True,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))
