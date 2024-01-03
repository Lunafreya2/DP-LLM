import sys

import datasets
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp = float(sys.argv[1])
top_k = int(sys.argv[2])

LABELS = ["sadness", "joy", "love", "anger", "fear", "suprise"]
# emotion = sys.argv[1]
dataset_e = datasets.load_dataset("dair-ai/emotion", name="split", split="test")

model = GPT2LMHeadModel.from_pretrained("./finetunedmodel")
tokenizer = GPT2Tokenizer.from_pretrained("./finetunedtokenizer")

model.to(device)
model.eval()

for idx, row in enumerate(dataset_e["text"]):
    emotion = LABELS[dataset_e["label"][0]]
    # text_break = row[0:10]
    text_break = " ".join(row.split(" ")[0:3])

    input_ids = (
        torch.tensor(tokenizer.encode(f"[BOS]{emotion}[SEP]{text_break}"))
        .unsqueeze(0)
        .to(device)
    )  # Empty string as a prompt

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

    if idx > 9:
        break
